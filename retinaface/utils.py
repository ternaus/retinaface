from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image


def vis_annotations(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    vis_image = image.copy()

    for annotation in annotations:
        landmarks = annotation["landmarks"]

        colors = [(255, 0, 0), (128, 255, 0), (255, 178, 102), (102, 128, 255), (0, 255, 255)]

        for landmark_id, (x, y) in enumerate(landmarks):
            vis_image = cv2.circle(vis_image, (x, y), radius=3, color=colors[landmark_id], thickness=3)

        x_min, y_min, x_max, y_max = annotation["bbox"]

        x_min = np.clip(x_min, 0, x_max - 1)
        y_min = np.clip(y_min, 0, y_max - 1)

        vis_image = cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    return vis_image


def filer_labels(labels: List[Dict], image_path: Path, min_size: int) -> List[Dict]:
    result: List[Dict[str, Any]] = []

    print("Before = ", len(labels))

    for label in labels:
        if not (image_path / label["file_name"]).exists():
            continue

        temp: List[Dict[str, Any]] = []

        width, height = Image.open(image_path / label["file_name"]).size

        for annotation in label["annotations"]:
            x_min, y_min, x_max, y_max = annotation["bbox"]

            x_min = np.clip(x_min, 0, width - 1)
            y_min = np.clip(y_min, 0, height - 1)
            x_max = np.clip(x_max, x_min + 1, width - 1)
            y_max = np.clip(y_max, y_min + 1, height - 1)

            annotation["bbox"] = x_min, y_min, x_max, y_max

            if x_max - x_min >= min_size and y_max - y_min >= min_size:
                temp += [annotation]

        if len(temp) > 0:
            label["annotation"] = temp
            result += [label]

    print("After = ", len(result))

    return result
