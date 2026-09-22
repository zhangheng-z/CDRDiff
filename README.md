# CDRDiff

## Environment Setup

A CUDA-capable NVIDIA GPU is required. Run the following commands from the repository root:

```bash
conda env create -f environment.yaml
conda activate cdrdiff
```

The environment uses Python 3.8.5, PyTorch 1.12.1, and CUDA Toolkit 11.3. Inference uses FP16 by default.

## Model Files

Place the four checkpoint files compatible with this implementation at the following locations:

```text
CDRDiff/
├── models/
│   └── control_sd15_ini.ckpt
├── checkpoints/
│   ├── My-Checkpoint(ours_new).ckpt
│   ├── generatorTrain_epoch_5.pth
│   └── main-epoch=00-step=7000.ckpt
└── empty_embedding.pkl
```

The checkpoints must be obtained separately. The repository already includes `empty_embedding.pkl`; keep it in the repository root.

## Testing

Place the input images in `test_data`, then run the following command from the repository root:

```bash
python my_test.py --input_folder ./test_data --same_folder ./output --checkpoint "./checkpoints/My-Checkpoint(ours_new).ckpt"
```

Enhanced images are saved to `output`. Images with existing output files are skipped.

To compute PSNR and SSIM, place the normal-light reference images in `reference_images`, using the same filenames as the corresponding input images, then run:

```bash
python my_test.py --input_folder ./test_data --gt_folder ./reference_images --same_folder ./output_eval --checkpoint "./checkpoints/My-Checkpoint(ours_new).ckpt"
```

Without `--gt_folder`, the script generates enhanced images without computing PSNR or SSIM. Leave `--save_memory` disabled (the default).
