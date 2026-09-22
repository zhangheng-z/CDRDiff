# Required inference checkpoints

Weights are not included in Git. Place the matching files here:

- `My-Checkpoint1.ckpt`: customized ControlNet checkpoint; keys use `module.control_model.`.
- `generatorTrain_epoch_5.pth`: decomposition network used by both the prior and inference entry point.
- `main-epoch=00-step=7000.ckpt`: bypass decoder checkpoint.

Also place `control_sd15_ini.ckpt` in `../models/`.
These files must match the architecture in this repository. The original QuadPrior
checkpoint is not a verified replacement for the customized ControlNet checkpoint.
Obtain the customized weights from the repository owner; no download URL is configured.
