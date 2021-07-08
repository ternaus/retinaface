import argparse
from typing import Dict, List, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torchvision.ops import nms

from retinaface.box_utils import decode, decode_landm
from retinaface.network import RetinaFace
from retinaface.prior_box import priorbox
from retinaface.utils import tensor_from_rgb_image, vis_annotations

state_dict = model_zoo.load_url(
    "https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip",
    progress=True,
    map_location="cpu",
)


class M(nn.Module):
    def __init__(self, max_size: int = 1280):
        super().__init__()
        self.model = RetinaFace(
            name="Resnet50",
            pretrained=False,
            return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
            in_channels=256,
            out_channels=256,
        )
        self.model.load_state_dict(state_dict)

        self.max_size = max_size

        self.scale_landmarks = torch.from_numpy(np.tile([self.max_size, self.max_size], 5))
        self.scale_bboxes = torch.from_numpy(np.tile([self.max_size, self.max_size], 2))

        self.prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(self.max_size, self.max_size),
        )
        self.nms_threshold: float = 0.4
        self.variance = [0.1, 0.2]
        self.confidence_threshold: float = 0.7

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loc, conf, land = self.model(x)

        conf = F.softmax(conf, dim=-1)

        boxes = decode(loc.data[0], self.prior_box, self.variance)

        boxes *= self.scale_bboxes
        scores = conf[0][:, 1]

        landmarks = decode_landm(land.data[0], self.prior_box, self.variance)
        landmarks *= self.scale_landmarks

        # ignore low scores
        valid_index = torch.where(scores > self.confidence_threshold)[0]
        boxes = boxes[valid_index]
        landmarks = landmarks[valid_index]
        scores = scores[valid_index]

        # do NMS
        keep = nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep, :]

        landmarks = landmarks[keep]
        scores = scores[keep]
        return boxes, scores, landmarks


def prepare_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    image = albu.Compose([albu.LongestMaxSize(max_size=max_size), albu.Normalize(p=1)])(image=image)["image"]

    height, width = image.shape[:2]

    return cv2.copyMakeBorder(image, 0, max_size - height, 0, max_size - width, borderType=cv2.BORDER_CONSTANT)


def main() -> None:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-m",
        "--max_size",
        type=int,
        help="Size of the input image. The onnx model will predict on (max_size, max_size)",
        required=True,
    )

    arg("-o", "--output_file", type=str, help="Path to save onnx model.", required=True)
    args = parser.parse_args()

    raw_image = cv2.imread("tests/data/13.jpg")

    image = prepare_image(raw_image, args.max_size)

    x = tensor_from_rgb_image(image).unsqueeze(0).float()

    model = M(max_size=args.max_size)
    model.eval()
    with torch.no_grad():
        out_torch = model(x)

    torch.onnx.export(
        model,
        x,
        args.output_file,
        verbose=True,
        opset_version=12,
        input_names=["input"],
        export_params=True,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(args.output_file)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(args.output_file)

    outputs = ort_session.run(None, {"input": np.expand_dims(np.transpose(image, (2, 0, 1)), 0)})

    for i in range(3):
        if not np.allclose(out_torch[i].numpy(), outputs[i]):
            raise ValueError("torch and onnx models do not match!")

    annotations: List[Dict[str, List[Union[float, List[float]]]]] = []

    for box_id, box in enumerate(outputs[0]):
        annotations += [
            {
                "bbox": box.tolist(),
                "score": outputs[1][box_id],
                "landmarks": outputs[2][box_id].reshape(-1, 2).tolist(),
            }
        ]

    im = albu.Compose([albu.LongestMaxSize(max_size=1280)])(image=raw_image)["image"]
    cv2.imwrite("example.jpg", vis_annotations(im, annotations))


if __name__ == "__main__":
    main()
