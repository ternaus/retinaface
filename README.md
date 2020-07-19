# Retinaface

This repo is build on top of https://github.com/biubug6/Pytorch_Retinaface

## Differences:

### Train loop moved to [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

IT added a set of functionality:

* Distributed training
* fp16
* Syncronized BatchNorm
* Support for various loggers like [W&B](https://www.wandb.com/) or [Neptune.ml](https://neptune.ai/)

### Hyperparameters are fedined in config file.

Hyperparameters that were scattered  across the code moved to the config at [retinadace/config](retinadace/config)

### Augmentations => [Albumentations](https://albumentations.ai/)

Color that were manually implemented replaced by the Albumentations library.

Todo:
* Horizontal Flip is not implemented in Albumentations
* Spatial transforms like rotations or transpose are not implemented yet.

Color transforms are defined in the config.

### Added mAP calculation for validation.
In order to track thr progress, mAP metric is calculated on validation.

## Training

```
python retinaface/train.py -h                                                                                                                                                                              (anaconda3)  15:14:11
usage: train.py [-h] -c CONFIG_PATH

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        Path to the config.

```
