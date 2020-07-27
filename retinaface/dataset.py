import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import albumentations as albu
import numpy as np
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils import data

from retinaface.data_augment import Preproc


class FaceDetectionDataset(data.Dataset):
    def __init__(
        self,
        label_path: str,
        image_path: Optional[str],
        transform: albu.Compose,
        preproc: Preproc,
        rotate90: bool = False,
    ) -> None:
        self.preproc = preproc

        if image_path is None:
            self.image_path = image_path
        else:
            self.image_path = Path(image_path)

        self.transform = transform
        self.rotate90 = rotate90

        with open(label_path) as f:
            self.labels = json.load(f)

        self.labels = [x for x in self.labels if Path(x["file_path"]).exists()]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        labels = self.labels[index]

        file_name = labels["file_name"]

        if self.image_path is None:
            image = load_rgb(labels["file_path"])
        else:
            image = load_rgb(self.image_path / file_name)

        # annotations will have the format
        # 4: box, 10 landmarks, 1: landmarks / no landmarks
        num_annotations = 4 + 10 + 1
        annotations = np.zeros((0, num_annotations))

        image_height, image_width = image.shape[:2]

        for label in labels["annotations"]:
            annotation = np.zeros((1, num_annotations))
            x_min, y_min, x_max, y_max = label["bbox"]

            annotation[0, 0] = np.clip(x_min, 0, image_width - 1)
            annotation[0, 1] = np.clip(y_min, 0, image_height - 1)
            annotation[0, 2] = np.clip(x_max, x_min + 1, image_width - 1)
            annotation[0, 3] = np.clip(y_max, y_min + 1, image_height - 1)

            if "landmarks" in label and label["landmarks"]:
                landmarks = np.array(label["landmarks"])
                # landmarks
                annotation[0, 4:14] = landmarks.reshape(-1, 10)
                if annotation[0, 4] < 0:
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)

        if self.rotate90:
            image, annotations = random_rotate_90(image, annotations.astype(int))

        image, annotations = self.preproc(image, annotations)

        image = self.transform(image=image)["image"]

        return {
            "image": tensor_from_rgb_image(image),
            "annotation": annotations.astype(np.float32),
            "file_name": file_name,
        }


def random_rotate_90(image: np.ndarray, annotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image_height, image_width = image.shape[:2]

    boxes = annotations[:, :4]
    keypoints = annotations[:, 4:-1].reshape(-1, 2)
    labels = annotations[:, -1:]

    invalid_index = keypoints.sum(axis=1) == -2

    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image_width - 1)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image_height - 1)

    keypoints[invalid_index] = 0

    category_ids = list(range(boxes.shape[0]))

    transform = albu.Compose(
        [albu.RandomRotate90(p=1)],
        keypoint_params=albu.KeypointParams(format="xy"),
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )
    transformed = transform(
        image=image, keypoints=keypoints.tolist(), bboxes=boxes.tolist(), category_ids=category_ids
    )

    keypoints = np.array(transformed["keypoints"])
    keypoints[invalid_index] = -1

    keypoints = keypoints.reshape(-1, 10)
    boxes = np.array(transformed["bboxes"])
    image = transformed["image"]

    annotations = np.hstack([boxes, keypoints, labels])

    return image, annotations


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
