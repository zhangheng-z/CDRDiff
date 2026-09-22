from cldm.hack import disable_verbosity, enable_sliced_attention

disable_verbosity()

import cv2
import einops
import numpy as np
import torch
import random
import glob
import os
import argparse
import img_name
import json
import time

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddim import DDIMSampler
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from extract_invirant.net import net

# 4090: 14G

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default='./checkpoints/My-Checkpoint1.ckpt', type=str)
parser.add_argument("--same_folder", default='output', type=str)
parser.add_argument("--input_folder", default='test_data', type=str)

parser.add_argument("--gt_folder", default=None, type=str)

parser.add_argument("--use_float16", default=True, type=bool)
parser.add_argument("--save_memory", default=False, type=bool)  # Cannot use. Has bugs


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


if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_file = args.checkpoint

    if args.save_memory:
        enable_sliced_attention()

    print("====== Load parameters ======")

    # Load pretrained Stable Diffusion v1.5
    model = create_model('./models/cldm_v15.yaml').cpu()

    state_dict = load_state_dict('./models/control_sd15_ini.ckpt', location='cpu')
    new_state_dict = {}
    for s in state_dict:
        if "cond_stage_model.transformer" not in s:
            new_state_dict[s] = state_dict[s]
    model.load_state_dict(new_state_dict)

    # Insert new layers in ControlNet (sorry for the ugliness
    model.add_new_layers()

    # Load trained checkpoint
    state_dict = load_state_dict(checkpoint_file, location='cpu')

    # for key in model.state_dict().keys():
    #     print(key)
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        print(sd_name)
        # if '_forward_module.control_model' in sd_name:
        if 'module.control_model' in sd_name:
            new_state_dict[sd_name.replace('module.control_model.', '')] = sd_param
    # model.control_model.load_state_dict(new_state_dict, strict=False)
    model.control_model.load_state_dict(new_state_dict)
    # Load bypass decoder
    ae_checkpoint = './checkpoints/main-epoch=00-step=7000.ckpt'
    model.change_first_stage(ae_checkpoint)

    print("====== Finish loading parameters ======")

    if args.use_float16:
        model = model.cuda().to(dtype=torch.float16)
    else:
        model = model.cuda()

    DDIM_sampler = DDIMSampler(model)
    diffusion_sampler = DPMSolverSampler(model)

    # DCE_net = enhance_net_nopool().cuda()
    # DCE_net.load_state_dict(torch.load('./Invirants/Epoch99.pth'))
    # DCE_net=DCE_net.to(torch.float16)
    # for param in DCE_net.parameters():
    #     param.requires_grad = False

    enhanced_net = net().cuda()
    enhanced_net.load_state_dict(torch.load('./checkpoints/generatorTrain_epoch_5.pth'))
    enhanced_net = enhanced_net.to(torch.float16)
    for param in enhanced_net.parameters():
        param.requires_grad = False

    enhanced_net.eval()
    model.eval()
    os.makedirs("feature_result", exist_ok=True)

    def process(input_image,high_img, prompt="", num_samples=1, image_resolution=512, diffusion_steps=10, guess_mode=False,
                strength=1.0, scale=9.0, seed=0, eta=0.0):
        with torch.no_grad():
            detected_map = resize_image(HWC3(input_image), image_resolution)
            H, W, C = detected_map.shape

            if args.use_float16:
                control = torch.from_numpy(detected_map.copy()).cuda().to(dtype=torch.float16) / 255.0
            else:
                control = torch.from_numpy(detected_map.copy()).cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            high_img = resize_image(HWC3(high_img), image_resolution)
            if args.use_float16:
                high_img = torch.from_numpy(high_img.copy()).cuda().to(dtype=torch.float16) / 255.0
            else:
                high_img = torch.from_numpy(high_img.copy()).cuda() / 255.0
            high_img = torch.stack([high_img for _ in range(num_samples)], dim=0)
            high_img = einops.rearrange(high_img, 'b h w c -> b c h w').clone()

            mean = torch.mean(control)
            control = control.to(dtype=torch.float32)
            if (mean.item() > 0.1):
                control = adjust_brightness_tensor(control, 0.06 / mean)
            control = control.to(dtype=torch.float16)
            L,R,X = enhanced_net(control)
            save_img(R[0],"R1")
            I = torch.pow(L,0.01) * R
            # control = control.to(dtype=torch.float32)
            # mean = torch.mean(control)
            # control = adjust_brightness_tensor(control, 0.6/mean)
            # control = control.to(dtype=torch.float16)
            I_h, I_ae_hs = model.encode_first_stage(I * 2 - 1)
            I_h = I_h.mode()


            # _, control, _ = DCE_net(control)
            # L,R,X = enhanced_net(control)
            # I = torch.pow(L,0.14) * R

            h,ae_hs = model.encode_first_stage(control * 2 - 1)
            # h,ae_hs = model.encode_first_stage(control * 2 - 1)
            # h = h.mode()
            # save_img((ae_hs[0][0]+1)/2,"ae_hs")

            # high_img = resize_image(HWC3(high_img), image_resolution)
            # high_img = torch.from_numpy(high_img.copy()).cuda().to(dtype=torch.float16) / 255.0
            # high_img = torch.stack([high_img for _ in range(num_samples)], dim=0)
            # high_img = einops.rearrange(high_img, 'b h w c -> b c h w').clone()
            # high_img_h, high_img_ae_hs = model.encode_first_stage(high_img * 2 - 1)
            # save_img((high_img_ae_hs[0][0]+1)/2,"high_img_ae_hs")


            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if args.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [model.get_unconditional_conditioning(num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control],
                       "c_crossattn": [model.get_unconditional_conditioning(num_samples)]}
            shape = (4, H // 8, W // 8)

            # noise,_=DDIM_sampler.ddim_reverse(I_h, cond,10,unconditional_guidance_scale=scale,unconditional_conditioning=un_cond)
            # noise = 1

            if args.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                        [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = diffusion_sampler.sample(diffusion_steps, num_samples,
                                                              shape, cond, verbose=False, eta=eta,
                                                              unconditional_guidance_scale=scale,
                                                              unconditional_conditioning=un_cond,
                                                              dmp_order=3,h=I_h)

            # save_img((samples[0]+1)/2,"sample_img")


            if args.save_memory:
                model.low_vram_shift(is_diffusing=False)

            if args.use_float16:
                n = len(intermediates)
                tmp_ae_hs = [tensor.repeat(n, 1, 1, 1) for tensor in ae_hs]
                x_samples = model.decode_new_first_stage(samples.to(dtype=torch.float16), ae_hs)
                # tmp = torch.cat(intermediates, dim=0)
                # intermediate = model.decode_new_first_stage(tmp.to(dtype=torch.float16), tmp_ae_hs)
                # for i in range(n):
                #     noise_img = intermediate[i]
                #     save_img(noise_img[0], f"noise_map_{i}")
            else:
                x_samples = model.decode_new_first_stage(samples, ae_hs)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                               255).astype(
                np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results


    # Low-light Enhancement
    # low_img = "dataset/LOLv1/eval15/low"
    # high_img = "dataset/LOLv1/eval15/high"
    low_img = args.input_folder
    high_img = args.gt_folder or args.input_folder
    img_list = glob.glob(f"{low_img}/*.*")
    print(f"Find {len(img_list)} files in {low_img}")
    tot_psnr=0
    tot_ssim=0
    nums=len(img_list)
    runtime_samples = []


    def gamma_transform(image, gamma=1.0):
        # 归一化像素值到[0, 1]
        normalized_image = image / 255.0
        # 应用Gamma变换
        gamma_corrected = np.power(normalized_image, gamma)
        # 还原像素值到[0, 255]并转换为uint8类型
        gamma_corrected = np.uint8(gamma_corrected * 255)
        return gamma_corrected

    for low_img_path in img_list:
        save_name = os.path.split(low_img_path)[1]

        file_suffix = os.path.splitext(save_name)[1]

        # save_name = save_name.rsplit('_', 1)[0]+file_suffix

        high_img_path = os.path.join(high_img, save_name)

        save_name = os.path.splitext(save_name)[0] +file_suffix
        save_path = os.path.join(args.same_folder, save_name)

        if os.path.exists(save_path):
            print(f"Exists {save_path}, skip.")
            continue

        img_name.img_name = save_name

        input_low_image = cv2.imread(low_img_path)
        # gamma = 0.5
        # input_low_image = gamma_transform(input_low_image, gamma=gamma)

        input_high_image = cv2.imread(high_img_path)
        height, width = input_high_image.shape[:2]

        # input_high_image = resize_image(HWC3(input_high_image), 512)
        # if you set num_samples > 1, process will return multiple results
        torch.cuda.synchronize()
        runtime_started = time.perf_counter()
        output = process(input_low_image,input_high_image, num_samples=1)[0]
        torch.cuda.synchronize()
        runtime_samples.append((time.perf_counter() - runtime_started) * 1000)

        output = img = cv2.resize(output, (width, height), interpolation=cv2.INTER_AREA)

        if args.gt_folder:
            psnr = peak_signal_noise_ratio(input_high_image, output, data_range=255)
            ssim = structural_similarity(input_high_image, output, data_range=255, channel_axis=2)

            result_content = f"({save_name})  PSNR: {psnr:.2f} dB  SSIM: {ssim:.4f}\n"
            with open("LOL_result.txt", "a") as f:
                f.write(result_content)
            print(f"PSNR:{psnr};SSIM:{ssim}")

            tot_psnr+=psnr
            tot_ssim+=ssim


        os.makedirs(args.same_folder, exist_ok=True)
        cv2.imwrite(save_path, output)

    if args.gt_folder and runtime_samples:
        avg_psnr = tot_psnr / len(runtime_samples)
        avg_ssim = tot_ssim / len(runtime_samples)
        print(f"avg_psnr:{avg_psnr};avg_ssim:{avg_ssim}")
        result_content = f"""\
           ============================================
           Evaluation Results (averaged over {len(runtime_samples)} pairs)
           ============================================
           PSNR: {avg_psnr:.2f} dB
           SSIM: {avg_ssim:.4f}
           """
        with open("LOL_result.txt", "a") as f:
            f.write(result_content)
    if not runtime_samples:
        print("No new images processed.")
        raise SystemExit(0)
    measured = runtime_samples[1:] if len(runtime_samples) > 1 else runtime_samples
    print("MODEL_RUNTIME_JSON=" + json.dumps({
        "model": "QuadPrior",
        "samples_ms": measured,
        "mean_ms": float(np.mean(measured)),
        "p50_ms": float(np.median(measured)),
        "min_ms": float(np.min(measured)),
        "max_ms": float(np.max(measured)),
        "warmup": 1 if len(runtime_samples) > 1 else 0,
        "repeats": len(measured),
        "input_shape": [1, 3, 400, 600],
        "effective_shape": [1, 3, 512, 768],
        "sampling_steps": 10,
        "precision": "FP16",
        "scope": "decoded_BGR_to_CPU_output; excludes_load_and_file_IO"
    }, sort_keys=True))
