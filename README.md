# CDRDiff — inference and evaluation

本仓库整理自本地修改版 QuadPrior，仅发布低光图像增强测试流程及其模型依赖。
入口为 `my_test.py`，不包含训练入口、训练数据加载脚本、数据集、模型权重或实验输出。
共享模型文件保留原有训练方法，以维持模型结构与检查点兼容；这不是完全剥离训练方法的实现。

## 环境

```bash
conda env create -f environment.yaml
conda activate cdrdiff
```

环境文件沿用原项目的 Python 3.8 / PyTorch 1.12.1 / CUDA 11.3 组合并删去训练专用依赖。
当前发布流程只进行了静态检查，未验证全新环境安装或 GPU 端到端推理。
本实现需要 NVIDIA CUDA，使用默认 FP16；原代码的 `--save_memory` 路径标有已知问题，请保持默认关闭。

## 模型文件

按照 [checkpoints/README.md](checkpoints/README.md) 放置四个匹配权重文件。
`empty_embedding.pkl` 是推理所需的小型空文本条件张量，随代码保留。
自定义权重尚未提供下载链接，单独克隆代码不能完成推理。

## 测试

从仓库根目录运行，将待增强图片放入自行创建的 `test_data` 目录：

```bash
python my_test.py --input_folder ./test_data --same_folder ./output --checkpoint ./checkpoints/My-Checkpoint1.ckpt
```

如需计算 PSNR/SSIM，提供文件名一一对应的正常光参考图：

```bash
python my_test.py --input_folder ./test_data --gt_folder ./reference_images --same_folder ./output_eval
```

未传入 `--gt_folder` 时不计算 PSNR/SSIM，避免把低光输入误当作真值。
已存在的输出会跳过；平均指标仅统计本次实际处理的图像。
`paired-metrics.py` 保留原项目的指标辅助函数，不是独立批量评测命令。
原始计时输出中的输入形状字段为固定示例值，不能用作任意数据的复杂度结论。

## 发布范围

详见 [RELEASE_SCOPE.md](RELEASE_SCOPE.md)。原始 `test.py` 是会提前返回的特征提取实验，未作为增强入口发布。

## 来源

本仓库基于 [QuadPrior](https://github.com/daooshee/QuadPrior) 的本地修改版本，
并沿用其 ControlNet / latent diffusion 相关实现。保留源文件中的来源说明。
本仓库不是原作者的官方实现。

原论文：Wenjing Wang, Huan Yang, Jianlong Fu, Jiaying Liu.
*Zero-Reference Low-Light Enhancement via Physical Quadruple Priors*, CVPR 2024.
