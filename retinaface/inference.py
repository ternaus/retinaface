"""
The script detects faces in images.

Takes a folder with images as input and returns a list of jsons:

<image_id>.json

[
    {
        "file_name": <img_file_name>,
        "annotations": [
                            {
                                "bbox": [int, int, int, int],
                                "confidence": float,
                                "landmarks":  [[x0, y0], [x1, y1] ... [x4, y4]]
                            }
    }
]

"""

import argparse
import json
from pathlib import Path
from typing import List, Any, Dict, Union

import albumentations as albu
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.utils.image_utils import pad_to_size, unpad_from_size, load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms

from retinaface.box_utils import decode, decode_landm
from retinaface.utils import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path with images.", required=True)
    arg("-c", "--config_path", type=Path, help="Path with images.", required=True)
    arg("-o", "--output_path", type=Path, help="Path to save jsons.", required=True)
    arg("-v", "--visualize", action="store_true", help="Visualize predictions")
    arg("-g", "--num_gpus", type=int, help="The number of GPUs to use.")
    arg("-m", "--max_size", type=int, help="Resize the largest side to this number", default=960)
    arg("--confidence_threshold", default=0.7, type=float, help="confidence_threshold")
    arg("--nms_threshold", default=0.4, type=float, help="nms_threshold")
    arg("-w", "--weight_path", type=str, help="Path to weights.")
    arg("--keep_top_k", default=750, type=int, help="keep_top_k")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, file_paths: List[Path], max_size: int, transform: albu.Compose) -> None:
        self.file_paths = file_paths
        self.transform = transform
        self.max_size = max_size
        self.resize = albu.LongestMaxSize(max_size=max_size, p=1)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = load_rgb(image_path, lib="cv2")
        image_height, image_width = image.shape[:2]

        image = self.resize(image=image)["image"]

        paded = pad_to_size(target_size=(self.max_size, self.max_size), image=image)

        image = paded["image"]
        pads = paded["pads"]

        image = self.transform(image=image)["image"]

        return {
            "torched_image": tensor_from_rgb_image(image),
            "image_path": str(image_path),
            "pads": torch.from_numpy(np.array(pads)),
            "image_height": image_height,
            "image_width": image_width,
        }


def unnormalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for c in range(image.shape[-1]):
        image[:, :, c] *= std[c]
        image[:, :, c] += mean[c]
        image[:, :, c] *= 255

    return image


class InferenceModel(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any], weight_path: Union[Path, str]) -> None:
        super().__init__()
        self.hparams = hparams
        self.model = object_from_dict(self.hparams["model"])

        corrections: Dict[str, str] = {"model.": ""}
        checkpoint = load_checkpoint(file_path=weight_path, rename_in_layers=corrections)
        self.model.load_state_dict(checkpoint["state_dict"])

    def setup(self, stage: int = 0) -> None:
        self.output_vis_path = Path(self.hparams["output_path"]) / "viz"

        if self.hparams["visualize"]:
            self.output_vis_path.mkdir(exist_ok=True, parents=True)

        self.output_label_path = Path(self.hparams["output_path"]) / "labels"
        self.output_label_path.mkdir(exist_ok=True, parents=True)

    def forward(self, batch: Dict) -> torch.Tensor:
        return self.model(batch)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            InferenceDataset(
                self.hparams["file_paths"],
                max_size=self.hparams["max_size"],
                transform=from_dict(self.hparams["test_aug"]),
            ),
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        torched_images = batch["torched_image"]  # images that are rescaled and padded
        pads = batch["pads"]
        image_paths = batch["image_path"]
        image_heights = batch["image_height"]
        image_widths = batch["image_width"]

        labels: List[Dict[str, Any]] = []

        loc, conf, land = self.model(torched_images)
        conf = F.softmax(conf, dim=-1)

        batch_size = torched_images.shape[0]

        image_height, image_width = torched_images.shape[2:]

        scale_bboxes = torch.from_numpy(np.tile([image_width, image_height], 5)).to(self.device)
        scale_landmarks = torch.from_numpy(np.tile([image_width, image_height], 2)).to(self.device)

        priors = object_from_dict(hparams["prior_box"], image_size=(image_height, image_width)).to(loc.device)

        for batch_id in range(batch_size):
            image_path = image_paths[batch_id]
            file_id = Path(str(image_path)).stem

            # resize = max(raw_image.shape[:2]) / self.hparams["max_size"]
            boxes = decode(loc.data[batch_id], priors, hparams["test_parameters"]["variance"])
            boxes *= scale_landmarks
            scores = conf[batch_id][:, 1]
            # boxes *= scale_landmarks / resize
            landmarks = decode_landm(land.data[batch_id], priors, hparams["test_parameters"]["variance"])
            # landmarks *= scale_bboxes / resize
            landmarks *= scale_bboxes

            # ignore low scores
            valid_index = torch.where(scores > self.hparams["confidence_threshold"])[0]
            boxes = boxes[valid_index]
            landmarks = landmarks[valid_index]
            scores = scores[valid_index]

            order = scores.argsort(descending=True)

            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

            # do NMS
            keep = nms(boxes, scores, self.hparams["nms_threshold"])
            boxes = boxes[keep, :].int()

            if boxes.shape[0] == 0:
                continue

            landmarks = landmarks[keep].int()
            scores = scores[keep].cpu().numpy().astype(np.float64)

            boxes = boxes[: self.hparams["keep_top_k"]].cpu().numpy()
            landmarks = landmarks[: self.hparams["keep_top_k"]].cpu().numpy().reshape([-1, 2])
            scores = scores[: self.hparams["keep_top_k"]]

            # Now we need to unpad
            normalized_image = np.transpose(torched_images[batch_id].cpu().numpy(), (1, 2, 0))
            image = unnormalize(normalized_image)

            unpadded = unpad_from_size(pads[batch_id].cpu().numpy(), image=image, bboxes=boxes, keypoints=landmarks)

            original_image_height = image_heights[batch_id].item()
            original_image_width = image_widths[batch_id].item()

            image = unpadded["image"]
            image = cv2.resize(image.astype(np.uint8), (original_image_width, original_image_height))

            resize_coeff = max(original_image_height, original_image_width) / torched_images.shape[-1]

            boxes = (unpadded["bboxes"] * resize_coeff).astype(int)
            landmarks = (unpadded["keypoints"].reshape(-1, 10) * resize_coeff).astype(int)

            if self.hparams["visualize"]:
                vis_image = image.copy()

                for crop_id, bbox in enumerate(boxes):
                    landms = landmarks[crop_id].reshape(-1, 2)

                    colors = [(255, 0, 0), (128, 255, 0), (255, 178, 102), (102, 128, 255), (0, 255, 255)]

                    for i, (x, y) in enumerate(landms):
                        vis_image = cv2.circle(vis_image, (x, y), radius=3, color=colors[i], thickness=3)

                    x_min, y_min, x_max, y_max = bbox

                    x_min = np.clip(x_min, 0, x_max - 1)
                    y_min = np.clip(y_min, 0, y_max - 1)

                    vis_image = cv2.rectangle(
                        vis_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2
                    )

                    cv2.imwrite(
                        str(self.output_vis_path / f"{file_id}.jpg"), cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    )

            for crop_id, bbox in enumerate(boxes):
                labels += [
                    {
                        "crop_id": crop_id,
                        "bbox": bbox.tolist(),
                        "score": scores[crop_id],
                        "landmarks": landmarks[crop_id].tolist(),
                    }
                ]

            result = {"file_path": image_path, "file_id": file_id, "bboxes": labels}

            with open(self.output_label_path / f"{file_id}.json", "w") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    hparams.update(
        {
            "file_paths": sorted([x for x in args.input_path.rglob("*") if x.is_file()]),
            "json_path": args.output_path,
            "output_path": args.output_path,
            "visualize": args.visualize,
            "max_size": args.max_size,
            "confidence_threshold": args.confidence_threshold,
            "nms_threshold": args.nms_threshold,
            "keep_top_k": args.keep_top_k,
            "num_workers": 12,
        }
    )
    hparams["trainer"]["gpus"] = 1  # Right now we work only with one GPU

    model = InferenceModel(hparams, weight_path=args.weight_path)
    trainer = object_from_dict(
        hparams["trainer"], checkpoint_callback=object_from_dict(hparams["checkpoint_callback"]),
    )

    trainer.test(model)
