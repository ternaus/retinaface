import json
from pathlib import Path
from typing import Dict, Any, List

import albumentations as albu
import numpy as np
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils import data

from retinaface.data_augment import Preproc


class FaceDetectionDataset(data.Dataset):
    def __init__(self, label_path: str, image_path: str, transform: albu.Compose, preproc: Preproc) -> None:
        self.preproc = preproc

        self.image_path = Path(image_path)
        self.transform = transform

        with open(label_path) as f:
            self.labels = json.load(f)

        self.valid_annotation_indices = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        labels = self.labels[index]
        file_name = labels["file_name"]
        image = load_rgb(self.image_path / file_name)

        # annotations will have the format
        # 4: box, 10 landmarks, 1: landmarks / no landmarks
        num_annotations = 4 + 10 + 1
        annotations = np.zeros((0, num_annotations))

        image_height, image_width = image.shape[:2]

        for label in labels["annotations"]:
            annotation = np.zeros((1, num_annotations))
            # bbox

            annotation[0, 0] = np.clip(label["x_min"], 0, image_width - 1)
            annotation[0, 1] = np.clip(label["y_min"], 0, image_height - 1)
            annotation[0, 2] = np.clip(label["x_min"] + label["width"], 1, image_width - 1)
            annotation[0, 3] = np.clip(label["y_min"] + label["height"], 1, image_height - 1)

            if not 0 <= annotation[0, 0] < annotation[0, 2] < image_width:
                continue
            if not 0 <= annotation[0, 1] < annotation[0, 3] < image_height:
                continue

            if "landmarks" in label and label["landmarks"]:
                landmarks = np.array(label["landmarks"])
                # landmarks
                annotation[0, 4:14] = landmarks[self.valid_annotation_indices]
                if annotation[0, 4] < 0:
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)

        image, target = self.preproc(image, annotations)

        image = self.transform(image=image)["image"]

        return {
            "image": tensor_from_rgb_image(image),
            "annotation": target.astype(np.float32),
            "file_name": file_name,
        }


def detection_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    annotation = []
    images = []
    file_names = []

    for sample in batch:
        images.append(sample["image"])
        annotation.append(torch.from_numpy(sample["annotation"]).float())
        file_names.append(sample["file_name"])

    return {"image": torch.stack(images), "annotation": annotation, "file_name": file_names}
