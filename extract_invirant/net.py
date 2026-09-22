import torch
import torch.nn as nn

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
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_

class MSG_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # PatchGAN分支
        self.patch_net = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            Res_block(64,64),
            Res_block(64,64),
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
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def compute_gradient(self, x):
        # 计算水平和垂直梯度
        grad_x = nn.functional.conv2d(x, self.sobel_x.to(x.device), padding=1,groups=3)
        grad_y = nn.functional.conv2d(x, self.sobel_y.to(x.device), padding=1,groups=3)
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
