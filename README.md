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
        "bbox": [
          449,
          330,
          571,
          720
        ],
        "landmarks": [
          [
            488.906,
            373.643
          ],
          [
            542.089,
            376.442
          ],
          [
            515.031,
            412.83
          ],
          [
            485.174,
            425.893
          ],
          [
            538.357,
            431.491
          ]
        ]
      }
    ]
  },
```

You can convert the defaule labels of the WiderFaces to the json of the propper format with this [script](https://github.com/ternaus/iglovikov_helper_functions/blob/master/iglovikov_helper_functions/data_processing/wider_face/prepare_data.py).


## Training

```
python retinaface/train.py -h
usage: train.py [-h] -c CONFIG_PATH

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        Path to the config.

```

## Inference

```
python retinaface/inference.py -h
usage: inference.py [-h] -i INPUT_PATH -c CONFIG_PATH -o OUTPUT_PATH [-v]
                    [-g NUM_GPUS] [-m MAX_SIZE] [-b BATCH_SIZE]
                    [-j NUM_WORKERS]
                    [--confidence_threshold CONFIDENCE_THRESHOLD]
                    [--nms_threshold NMS_THRESHOLD] -w WEIGHT_PATH
                    [--keep_top_k KEEP_TOP_K] [--world_size WORLD_SIZE]
                    [--local_rank LOCAL_RANK] [--fp16]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Path with images.
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        Path to config.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save jsons.
  -v, --visualize       Visualize predictions
  -g NUM_GPUS, --num_gpus NUM_GPUS
                        The number of GPUs to use.
  -m MAX_SIZE, --max_size MAX_SIZE
                        Resize the largest side to this number
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch_size
  -j NUM_WORKERS, --num_workers NUM_WORKERS
                        num_workers
  --confidence_threshold CONFIDENCE_THRESHOLD
                        confidence_threshold
  --nms_threshold NMS_THRESHOLD
                        nms_threshold
  -w WEIGHT_PATH, --weight_path WEIGHT_PATH
                        Path to weights.
  --keep_top_k KEEP_TOP_K
                        keep_top_k
  --world_size WORLD_SIZE
                        number of nodes for distributed training
  --local_rank LOCAL_RANK
                        node rank for distributed training
  --fp16                Use fp6
```

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> retinaface/inference.py <parameters>
```

*  [Weights](https://drive.google.com/drive/folders/1DuiwlTd1BbZ0ZzafrV7qMncko1Z5a412?usp=sharing) for the model with [config](retinaface/configs/2020-07-19.yaml).
*  [Weights](https://drive.google.com/file/d/1slNNW1bntYqDKpvi2r1QfcQAwnhsVw9n/view?usp=sharing) for the model with [config](retinaface/configs/2020-07-20.yaml).
