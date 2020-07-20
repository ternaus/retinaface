# Retinaface

![https://habrastorage.org/webt/uj/ff/vx/ujffvxxpzixwlmae8gyh7jylftq.jpeg](https://habrastorage.org/webt/uj/ff/vx/ujffvxxpzixwlmae8gyh7jylftq.jpeg)

This repo is build on top of [https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

## Differences

### Train loop moved to [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

IT added a set of functionality:

 * Distributed training
 * fp16
 * Syncronized BatchNorm
 * Support for various loggers like [W&B](https://www.wandb.com/) or [Neptune.ml](https://neptune.ai/)

### Hyperparameters are fedined in config file

Hyperparameters that were scattered  across the code moved to the config at [retinadace/config](retinadace/config)

### Augmentations => [Albumentations](https://albumentations.ai/)

Color that were manually implemented replaced by the Albumentations library.

Todo:
* Horizontal Flip is not implemented in Albumentations
* Spatial transforms like rotations or transpose are not implemented yet.

Color transforms are defined in the config.

### Added mAP calculation for validation
In order to track thr progress, mAP metric is calculated on validation.

## Data Preparation

The pipeline expects labels in the format:

```
[
  {
    "file_name": "0--Parade/0_Parade_marchingband_1_849.jpg",
    "annotations": [
      {
        "x_min": 449,
        "y_min": 330,
        "width": 122,
        "height": 149,
        "landmarks": [
          488.906,1
          373.643,
          0.0,
          542.089,
          376.442,
          0.0,
          515.031,
          412.83,
          0.0,
          485.174,
          425.893,
          0.0,
          538.357,
          431.491,
          0.0,
          0.82
        ]
      }
    ]
  },
```



## Training

```
python retinaface/train.py -h                                                                                                                                                                              (anaconda3)  15:14:11
usage: train.py [-h] -c CONFIG_PATH

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        Path to the config.

```

## Inference

```
python retinaface/inference.py -h                                                                                                                                                                                (anaconda3)  14:47:09
usage: inference.py [-h] -i INPUT_PATH -c CONFIG_PATH -o OUTPUT_PATH [-v]
                    [-g NUM_GPUS] [-t TARGET_SIZE] [-m MAX_SIZE]
                    [--origin_size]
                    [--confidence_threshold CONFIDENCE_THRESHOLD]
                    [--nms_threshold NMS_THRESHOLD] [-w WEIGHT_PATH]
                    [--keep_top_k KEEP_TOP_K]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Path with images.
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        Path with images.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save jsons.
  -v, --visualize       Visualize predictions
  -g NUM_GPUS, --num_gpus NUM_GPUS
                        The number of GPUs to use.
  -t TARGET_SIZE, --target_size TARGET_SIZE
                        Target size
  -m MAX_SIZE, --max_size MAX_SIZE
                        Target size
  --origin_size         Whether use origin image size to evaluate
  --confidence_threshold CONFIDENCE_THRESHOLD
                        confidence_threshold
  --nms_threshold NMS_THRESHOLD
                        nms_threshold
  -w WEIGHT_PATH, --weight_path WEIGHT_PATH
                        Path to weights.
  --keep_top_k KEEP_TOP_K
                        keep_top_k
```

[Weights](https://drive.google.com/drive/folders/1DuiwlTd1BbZ0ZzafrV7qMncko1Z5a412?usp=sharing) for the model
with [config](retinaface/configs/2020-07-19.yaml).
