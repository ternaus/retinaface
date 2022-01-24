"""There is a lot of post processing of the predictions."""
from collections import OrderedDict
from typing import Dict, List, Union

import albumentations as A
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

from retinaface.box_utils import decode, decode_landm
from retinaface.network import RetinaFace
from retinaface.prior_box import priorbox
from retinaface.utils import tensor_from_rgb_image

ROUNDING_DIGITS = 2


class Model:
    def __init__(self, max_size: int = 960, device: str = "cpu") -> None:
        self.model = RetinaFace(
            name="Resnet50",
            pretrained=False,
            return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
            in_channels=256,
            out_channels=256,
        ).to(device)
        self.device = device
        self.transform = A.Compose([A.LongestMaxSize(max_size=max_size, p=1), A.Normalize(p=1)])
        self.max_size = max_size
        self.variance = [0.1, 0.2]

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self.model.load_state_dict(state_dict)

    def eval(self) -> None:  # noqa: A003
        self.model.eval()

    def predict_jsons(
        self, image: np.ndarray, confidence_threshold: float = 0.7, nms_threshold: float = 0.4
    ) -> List[Dict[str, Union[List, float]]]:
        with torch.no_grad():
            original_height, original_width = image.shape[:2]

            transformed_image = self.transform(image=image)["image"]

            transformed_height, transformed_width = transformed_image.shape[:2]
            transformed_size = (transformed_width, transformed_height)

            scale_landmarks = torch.from_numpy(np.tile(transformed_size, 5)).to(self.device)
            scale_bboxes = torch.from_numpy(np.tile(transformed_size, 2)).to(self.device)

            prior_box = priorbox(
                min_sizes=[[16, 32], [64, 128], [256, 512]],
                steps=[8, 16, 32],
                clip=False,
                image_size=transformed_image.shape[:2],
            ).to(self.device)

            torched_image = tensor_from_rgb_image(transformed_image).to(self.device)

            loc, conf, land = self.model(torched_image.unsqueeze(0))  # pylint: disable=E1102

            conf = F.softmax(conf, dim=-1)

            annotations: List[Dict[str, Union[List, float]]] = []

            boxes = decode(loc.data[0], prior_box, self.variance)

            boxes *= scale_bboxes
            scores = conf[0][:, 1]

            landmarks = decode_landm(land.data[0], prior_box, self.variance)
            landmarks *= scale_landmarks

            # ignore low scores
            valid_index = torch.where(scores > confidence_threshold)[0]
            boxes = boxes[valid_index]
            landmarks = landmarks[valid_index]
            scores = scores[valid_index]

            # do NMS
            keep = nms(boxes, scores, nms_threshold)
            boxes = boxes[keep, :]

            if boxes.shape[0] == 0:
                return [{"bbox": [], "score": -1, "landmarks": []}]

            landmarks = landmarks[keep]

            scores = scores[keep].cpu().numpy().astype(float)

            boxes_np = boxes.cpu().numpy()
            landmarks_np = landmarks.cpu().numpy()
            resize_coeff = original_height / transformed_height

            boxes_np *= resize_coeff
            landmarks_np = landmarks_np.reshape(-1, 10) * resize_coeff

            for box_id, bbox in enumerate(boxes_np):
                x_min, y_min, x_max, y_max = bbox

                x_min = np.clip(x_min, 0, original_width - 1)
                x_max = np.clip(x_max, x_min + 1, original_width - 1)

                if x_min >= x_max:
                    continue

                y_min = np.clip(y_min, 0, original_height - 1)
                y_max = np.clip(y_max, y_min + 1, original_height - 1)

                if y_min >= y_max:
                    continue

                annotations += [
                    {
                        "bbox": np.round(bbox.astype(float), ROUNDING_DIGITS).tolist(),
                        "score": np.round(scores, ROUNDING_DIGITS)[box_id],
                        "landmarks": np.round(landmarks_np[box_id].astype(float), ROUNDING_DIGITS)
                        .reshape(-1, 2)
                        .tolist(),
                    }
                ]

            return annotations
