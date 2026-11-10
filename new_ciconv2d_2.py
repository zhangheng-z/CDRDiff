# Import general dependencies
import math
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import random

eps = 1e-4


# ==================================
# ======== Gaussian filter =========
# ==================================

# This code is based on https://github.com/Attila94/CIConv/blob/main/method/ciconv2d.py

def save_img(img, img_name):
    # img = img.squeeze(0)
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    str = f"./feature_result/{img_name}.png"
    # cv2.imwrite(str, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str, img)


def adjust_brightness_tensor(image_tensor, brightness_factor=0.6):
    device = image_tensor.device
    hsv_tensor = rgb_to_hsv(image_tensor)
    h, s, v = torch.chunk(hsv_tensor, 3, dim=1)
    v_adjusted = v * brightness_factor
    v_adjusted = torch.clamp(v_adjusted, 0, 1)
    hsv_adjusted = torch.cat([h, s, v_adjusted], dim=1)
    rgb_adjusted = hsv_to_rgb(hsv_adjusted)
    return rgb_adjusted


def rgb_to_hsv(image):
    img = image * 255  # 转换为0-255范围处理
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc, _ = torch.max(img, dim=1)
    minc, _ = torch.min(img, dim=1)
    diff = maxc - minc + 1e-8
    # 计算Hue
    h = torch.zeros_like(maxc)
    rc = (maxc - r) / diff
    gc = (maxc - g) / diff
    bc = (maxc - b) / diff
    h = torch.where(maxc == r, bc - gc, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0  # 映射到[0,1)
    s = diff / (maxc + 1e-8)
    v = maxc / 255.0
    return torch.stack([h, s, v], dim=1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h = h * 6.0
    i = torch.floor(h)
    f = h - i
    i = i.to(dtype=torch.int32) % 6
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    rgb = torch.zeros_like(hsv)
    v = v.unsqueeze(1)
    t = t.unsqueeze(1)
    p = p.unsqueeze(1)
    q = q.unsqueeze(1)
    idx = i.unsqueeze(1)
    rgb = torch.where(idx == 0, torch.cat([v, t, p], dim=1), rgb)
    rgb = torch.where(idx == 1, torch.cat([q, v, p], dim=1), rgb)
    rgb = torch.where(idx == 2, torch.cat([p, v, t], dim=1), rgb)
    rgb = torch.where(idx == 3, torch.cat([p, q, v], dim=1), rgb)
    rgb = torch.where(idx == 4, torch.cat([t, p, v], dim=1), rgb)
    rgb = torch.where(idx == 5, torch.cat([v, p, q], dim=1), rgb)
    return torch.clamp(rgb, 0, 1)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()

        sequence = []

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]

        self.model = nn.Sequential(*sequence)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        out = self.model(x) + self.conv(x)

        return out


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class MSG_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # PatchGAN分支
        self.patch_net = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            Res_block(64, 64),
            Res_block(64, 64),
            # AttnBlock(64),
            nn.Conv2d(64, 64, 3, stride=(2, 2), padding=1),
            nn.LeakyReLU(),
            Res_block(64, 128),
            Res_block(128, 128),
            # AttnBlock(128),
            nn.Conv2d(128, 128, 3, stride=(2, 2), padding=1),
            nn.LeakyReLU(),
            Res_block(128, 256),
            Res_block(256, 256),
            # AttnBlock(256),
            nn.Conv2d(256, 256, 3, stride=(2, 2), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, 3, padding=1),  # 输出概率图
        )
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(
            3, 1, 1, 1)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(
            3, 1, 1, 1)

    def compute_gradient(self, x):
        # 计算水平和垂直梯度
        grad_x = nn.functional.conv2d(x, self.sobel_x.to(x.device), padding=1, groups=3)
        grad_y = nn.functional.conv2d(x, self.sobel_y.to(x.device), padding=1, groups=3)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # 梯度幅值
        return gradient.mean(dim=1, keepdim=True)

    def forward(self, x):
        gradient = self.compute_gradient(x)
        x_with_grad = torch.cat([x, gradient], dim=1)
        patch_out = self.patch_net(x_with_grad)

        return torch.sigmoid(patch_out)


class L_net(nn.Module):
    def __init__(self, num=64):
        super(L_net, self).__init__()
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.L_net(input))


class R_net(nn.Module):
    def __init__(self, num=64):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.R_net(input))


class N_net(nn.Module):
    def __init__(self, num=64):
        super(N_net, self).__init__()
        self.N_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.N_net(input))


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.L_net = L_net(num=64)
        self.R_net = R_net(num=64)
        self.N_net = N_net(num=64)

    def forward(self, input):
        x = self.N_net(input)
        L = self.L_net(x)
        R = self.R_net(x)
        return L, R, x


def add_noise_to_tensor(tensor, noise_type='gaussian', noise_level=0.1, device='cuda'):
    """
    向张量添加不同类型的噪声 (纯 PyTorch 实现)

    参数:
        tensor: 输入张量 (b, 1, h, w) 或 (b, h, w) [单通道]
        noise_type: 噪声类型 ('gaussian', 'salt_pepper', 'poisson', 'speckle')
        noise_level: 噪声强度 (0-1)
        device: 使用的设备 ('cuda' 或 'cpu')

    返回:
        添加噪声后的张量 (b, 1, h, w)
    """
    # 确保是单通道张量 (b, 1, h, w)
    if tensor.dim() == 3:  # (b, h, w)
        tensor = tensor.unsqueeze(1)  # -> (b, 1, h, w)
    elif tensor.dim() == 4 and tensor.shape[1] != 1:
        # 如果多通道，取平均转换为单通道
        tensor = tensor.mean(dim=1, keepdim=True)

    # 确保张量在正确的设备上
    tensor = tensor.to(device)

    # 保存原始值范围
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # 归一化到 [0,1]
    normalized_tensor = tensor
    if max_val > min_val:
        normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # 添加不同类型的噪声
    noisy = normalized_tensor.clone()
    batch_size, _, height, width = noisy.shape

    if noise_type == 'gaussian':
        # 高斯噪声
        gauss = torch.randn_like(noisy) * noise_level
        noisy = torch.clamp(noisy + gauss, 0, 1)

    elif noise_type == 'salt_pepper':
        # 椒盐噪声
        s_vs_p = 0.5  # 盐和胡椒的比例
        amount = noise_level

        # 创建随机掩码
        random_mask = torch.rand_like(noisy)

        # 计算需要修改的像素数量
        num_pixels = int(amount * height * width)
        if num_pixels <= 0:  # 避免零像素
            num_pixels = 1

        # 创建盐噪声掩码
        salt_mask = (random_mask < num_pixels / (height * width)) & (random_mask < s_vs_p)
        # 创建椒噪声掩码
        pepper_mask = (random_mask < num_pixels / (height * width)) & (random_mask >= s_vs_p)

        # 应用盐噪声
        noisy[salt_mask] = 1.0
        # 应用椒噪声
        noisy[pepper_mask] = 0.0

    elif noise_type == 'poisson':
        # 泊松噪声
        # 由于泊松噪声在 PyTorch 中没有直接实现，我们使用近似
        vals = len(torch.unique(noisy))
        vals = 2 ** torch.ceil(torch.log2(torch.tensor(vals, device=device)))
        noisy = torch.poisson(noisy * vals) / vals

    elif noise_type == 'speckle':
        # 乘性噪声
        speckle = torch.randn_like(noisy)
        noisy = torch.clamp(noisy + noisy * speckle * noise_level, 0, 1)

    # 恢复原始值范围
    if max_val > min_val:
        noisy = min_val + noisy * (max_val - min_val)

    return noisy


def create_gaussian_kernel(kernel_size=5, sigma=1.0, device='cuda'):
    """
    创建高斯卷积核

    参数:
        kernel_size: 卷积核大小
        sigma: 高斯标准差
        device: 使用的设备

    返回:
        高斯卷积核 (1, 1, kernel_size, kernel_size)
    """
    # 生成坐标网格
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(x, x, indexing='ij')

    # 计算二维高斯分布
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma ** 2))

    # 归一化
    kernel = kernel / kernel.sum()

    # 添加维度: (1, 1, kernel_size, kernel_size)
    return kernel.view(1, 1, kernel_size, kernel_size)


def extract_high_frequency_grayscale(tensor, kernel_size=5, sigma=1.0, device='cuda'):
    """
    提取单通道灰度张量的高频信息 (纯 PyTorch 实现)

    参数:
        tensor: 单通道输入张量，形状为 (b, 1, h, w) 或 (b, h, w)
        kernel_size: 高斯核大小，建议为奇数
        sigma: 高斯核标准差
        device: 使用的设备 ('cuda' 或 'cpu')

    返回:
        高频信息张量，形状与输入相同 (b, 1, h, w)
    """
    # 确保是单通道张量 (b, 1, h, w)
    if tensor.dim() == 3:  # (b, h, w)
        tensor = tensor.unsqueeze(1)  # -> (b, 1, h, w)
    elif tensor.dim() == 4 and tensor.shape[1] != 1:
        # 如果多通道，取平均转换为单通道
        tensor = tensor.mean(dim=1, keepdim=True)

    # 确保张量在正确的设备上
    tensor = tensor.to(device)

    # 创建高斯卷积核
    kernel = create_gaussian_kernel(kernel_size, sigma, device).to(dtype=torch.float16)

    # 应用卷积实现高斯模糊
    padding = kernel_size // 2
    low_frequency = F.conv2d(tensor, kernel, padding=padding)

    # 高频信息 = 原始图像 - 低频模糊图像
    high_frequency = tensor - low_frequency

    return high_frequency


def rgb_to_grayscale(tensor):
    """
    将RGB张量转换为灰度张量 (b, c, h, w) -> (b, 1, h, w)

    使用公式: 0.2989 * R + 0.5870 * G + 0.1140 * B

    参数:
        tensor: RGB图像张量 (b, 3, h, w)

    返回:
        灰度图像张量 (b, 1, h, w)
    """
    # 确保输入是RGB图像 (b, 3, h, w)
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have 3 channels for RGB conversion")

    # RGB转灰度公式
    r, g, b = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # 添加通道维度 (b, h, w) -> (b, 1, h, w)
    return gray.unsqueeze(1)


def linear_stretch(high_freq, min_percentile=1, max_percentile=99):
    """
    对高频信息进行线性拉伸 (纯 PyTorch 实现)

    参数:
        high_freq: 高频信息张量 (b, 1, h, w)
        min_percentile: 最小值百分位
        max_percentile: 最大值百分位

    返回:
        拉伸后的高频信息张量 (b, 1, h, w)
    """
    # 确保有通道维度 (b, 1, h, w)
    if high_freq.dim() == 3:  # (b, h, w)
        high_freq = high_freq.unsqueeze(1)  # -> (b, 1, h, w)

    # 为批次中的每个图像应用线性拉伸
    stretched_tensors = []
    for i in range(high_freq.shape[0]):
        # 提取单张图像 (1, h, w)
        img_tensor = high_freq[i]

        # 获取通道数据 (h, w)
        channel = img_tensor.squeeze(0)

        # 展平并计算分位数
        flat = channel.flatten().to(dtype=torch.float32)
        min_val = torch.quantile(flat, min_percentile / 100.0)
        max_val = torch.quantile(flat, max_percentile / 100.0)

        # 处理特殊情况（所有值相同）
        if min_val == max_val:
            stretched_tensor = img_tensor
        else:
            # 应用线性拉伸
            stretched_channel = (channel - min_val) / (max_val - min_val)
            stretched_channel = torch.clamp(stretched_channel, 0, 1)
            stretched_tensor = stretched_channel.unsqueeze(0).to(dtype=torch.float32)  # (1, h, w)

        stretched_tensors.append(stretched_tensor)

    # 将列表连接为批次张量 (b, 1, h, w)
    return torch.stack(stretched_tensors)


class PriorConv2d(nn.Module):
    def __init__(self, invariant, k=3, scale=0.0):

        super(PriorConv2d, self).__init__()
        self.use_cuda = torch.cuda.is_available()

        # Constants
        self.gcm = torch.nn.Parameter(torch.tensor([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]))
        self.k = k
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            torch.nn.Conv2d(16, 1, 3, padding=1)
        )
        self.extract_net = net()
        # self.phrase_correct_net=PhaseCorrectionNet()

    def forward(self, batch):
        # Make sure scale does not explode: clamp to max abs value of 2.5
        # self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5)

        batch_t = batch.clone()

        # 提取高频细节特征
        gray_batch = rgb_to_grayscale(batch)
        high_freq = extract_high_frequency_grayscale(gray_batch, kernel_size=5, sigma=1.5)
        save_img(high_freq[0],"high_freq")
        stretched_high_freq = linear_stretch(high_freq, min_percentile=1, max_percentile=99)

        if(self.training):
            noise_level = 0.08
            noise_type = 'gaussian'
            random_num = random.random()
            if (random_num > 0.5):
                stretched_high_freq = add_noise_to_tensor(
                    stretched_high_freq,
                    noise_type=noise_type,
                    noise_level=noise_level
                )

        mean = torch.mean(batch)
        batch = batch.to(dtype=torch.float32)
        if (mean.item() > 0.1):
            batch = adjust_brightness_tensor(batch, 0.06/mean)
        batch = batch.to(dtype=torch.float16)

        # 测试阶段每个batch只运行一次extract_net
        if(self.training == False and hasattr(self, 'R') and hasattr(self, 'batch') and batch.shape == self.batch.shape and torch.allclose(batch, self.batch)):
            R = self.R
            L = self.L
            X = self.X
            self.batch = batch
        else:
            L, R, X = self.extract_net(batch)
            print("=============================")
            if(self.training == False):
                self.L,self.R,self.X = L,R,X
                self.batch = batch

        # total_params = sum(p.numel() for p in self.extract_net.parameters())
        # print(total_params)

        save_img(L[0], "L")
        save_img(X[0], "X")

        # features = torch.cat([stretched_high_freq, stretched_high_freq, stretched_high_freq, R], dim=1)
        features = torch.cat([R, R], dim=1)

        save_img(stretched_high_freq[0], "stretched_high_freq")
        save_img(R[0], "R")
        # save_img(fourier[0],name+"fourier")


        return features.to(dtype=torch.float16)
