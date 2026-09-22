# CDRDiff

## 环境搭建

需要支持 CUDA 的 NVIDIA 显卡。在仓库根目录执行：

```bash
conda env create -f environment.yaml
conda activate cdrdiff
```

环境配置为 Python 3.8.5、PyTorch 1.12.1、CUDA Toolkit 11.3。测试默认使用 FP16。

## 模型文件位置

将与本代码匹配的四个权重文件放置到以下位置：

```text
CDRDiff/
├── models/
│   └── control_sd15_ini.ckpt
├── checkpoints/
│   ├── My-Checkpoint1.ckpt
│   ├── generatorTrain_epoch_5.pth
│   └── main-epoch=00-step=7000.ckpt
└── empty_embedding.pkl
```

权重文件需单独准备；`empty_embedding.pkl` 已包含在仓库中，保留在根目录即可。

## 如何测试

将待增强图片放入 `test_data` 目录，从仓库根目录运行：

```bash
python my_test.py --input_folder ./test_data --same_folder ./output --checkpoint ./checkpoints/My-Checkpoint1.ckpt
```

增强结果保存在 `output` 目录，已存在的输出文件会跳过。

如需计算 PSNR 和 SSIM，将文件名一一对应的正常光参考图放入 `reference_images` 目录，运行：

```bash
python my_test.py --input_folder ./test_data --gt_folder ./reference_images --same_folder ./output_eval --checkpoint ./checkpoints/My-Checkpoint1.ckpt
```

未指定 `--gt_folder` 时只生成增强图像，不计算 PSNR/SSIM。保持 `--save_memory` 默认关闭。
