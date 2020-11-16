from typing import Any, Dict, List

import cv2
import numpy as np


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
