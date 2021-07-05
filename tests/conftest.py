from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch


def load_rgb(file_path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(file_path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


images = {
    "with_faces": {
        "image": load_rgb("tests/data/13.jpg"),
        "faces": [{'bbox': [254.99, 91.76, 337.66, 201.98],
                   'score': 1.0,
                   'landmarks': [[288.83, 134.29],
                                 [325.65, 136.9],
                                 [312.03, 160.74],
                                 [281.34, 168.41],
                                 [319.33, 170.77]]},
                  {'bbox': [438.06, 123.93, 508.97, 210.71],
                   'score': 1.0,
                   'landmarks': [[463.38, 155.05],
                                 [495.77, 155.46],
                                 [482.85, 175.24], [466.35, 188.72], [491.6, 189.14]]}, {'bbox': [657.0, 158.76, 729.22, 246.83], 'score': 1.0, 'landmarks': [[665.87, 188.04], [696.9, 197.93], [670.69, 214.92], [666.21, 221.44], [686.91, 229.49]]}]
    },
    "with_no_faces": {
        "image": load_rgb("tests/data/no_face.jpg"),
        "faces": [{'bbox': [], 'score': -1, 'landmarks': []}]
    }
}