# Required inference checkpoints

Weights are not included in Git. Place the matching files here:

- `My-Checkpoint(ours_new).ckpt`: customized ControlNet checkpoint; keys use `module.control_model.`.
- `generatorTrain_epoch_10.pth`: decomposition network used by both the prior and inference entry point.
- `main-epoch=00-step=7000.ckpt`: bypass decoder checkpoint.

Also place `control_sd15_ini.ckpt` in `../models/`.
These files must match the architecture in this repository. The original QuadPrior
checkpoint is not a verified replacement for the customized ControlNet checkpoint.
Download the three files listed above from [Baidu Netdisk](https://pan.baidu.com/s/1NlQwiE69ySTulql6a_FRtQ?pwd=4mvd) (extraction code: `4mvd`).

For `control_sd15_ini.ckpt`, use the [official QuadPrior Google Drive folder](https://drive.google.com/drive/folders/1NbqfOJYjv-_zH1NzTaaLmZDKjYA9clbd?usp=drive_link). This is a ControlNet initialization checkpoint based on SD 1.5, not a renamed standard SD 1.5 checkpoint.
