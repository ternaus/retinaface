from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_rgb(file_path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(file_path))

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


images = {
    "with_faces": {
        "image": load_rgb("tests/data/13.jpg"),
        "faces": [
            {
                "bbox": [256.9, 93.64, 336.79, 201.76],
                "score": 1.0,
                "landmarks": [
                    [286.17, 134.94],
                    [323.32, 135.28],
                    [309.15, 161.34],
                    [283.74, 168.48],
                    [320.72, 168.48],
                ],
            },
            {
                "bbox": [436.62, 118.5, 510.04, 211.13],
                "score": 1.0,
                "landmarks": [[460.96, 155.7], [494.47, 154.35], [480.52, 175.92], [464.73, 188.05], [491.9, 187.53]],
            },
            {
                "bbox": [657.3, 156.87, 729.81, 245.78],
                "score": 1.0,
                "landmarks": [[665.64, 187.11], [696.5, 196.97], [670.65, 214.76], [666.92, 220.2], [689.45, 228.91]],
            },
        ],
    },
    "with_no_faces": {
        "image": load_rgb("tests/data/no_face.jpg"),
        "faces": [{"bbox": [], "score": -1, "landmarks": []}],
    },
}
