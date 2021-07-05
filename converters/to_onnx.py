# import onnx
from typing import List, Union, Dict

import torch
from torch import nn
# from retinaface.pre_trained_models import get_model
from retinaface.network import RetinaFace
from torchvision.ops import nms
from torch.nn import functional as F

from retinaface.network import RetinaFace
from retinaface.prior_box import priorbox

from torch.utils import model_zoo
import numpy as np
from retinaface.box_utils import decode, decode_landm
import onnx
import onnxruntime as ort

state_dict = model_zoo.load_url("https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip", progress=True, map_location="cpu")


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.model = RetinaFace(
                name="Resnet50",
                pretrained=False,
                return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
                in_channels=256,
                out_channels=256,
            )
        self.model.load_state_dict(state_dict)

        self.max_size = 1280

        self.prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(self.max_size, self.max_size)
        )

        self.scale_landmarks = torch.from_numpy(np.tile([self.max_size, self.max_size], 5))
        self.scale_bboxes = torch.from_numpy(np.tile([self.max_size, self.max_size], 2))
        self.nms_threshold: float = 0.4
        self.variance = [0.1, 0.2]
        self.confidence_threshold: float = 0.7

    def forward(self, x: torch.Tensor):
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

        # # Sort from high to low
        # order = scores.argsort(descending=True)
        # boxes = boxes[order]
        # landmarks = landmarks[order]
        # scores = scores[order]
        #
        # # do NMS
        keep = nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep, :]
        #
        # if boxes.shape[0] == 0:
        #     return [{"bbox": [], "score": -1, "landmarks": []}]

        landmarks = landmarks[keep]
        scores = scores[keep]
        boxes = boxes[keep]

        return boxes, scores, landmarks
        # scores = scores[keep].cpu().numpy().astype(np.float64)
        # boxes = boxes.cpu().numpy()
        # landmarks = landmarks.cpu().numpy()
        # landmarks = landmarks.reshape([-1, 2])

        # return landmarks
        #
        # unpadded = unpad_from_size(pads, bboxes=boxes, keypoints=landmarks)
        #
        # resize_coeff = max(original_height, original_width) / self.max_size
        #
        # boxes = (unpadded["bboxes"] * resize_coeff).astype(int)
        # landmarks = (unpadded["keypoints"].reshape(-1, 10) * resize_coeff).astype(int)
        #
        # for box_id, bbox in enumerate(boxes):
        #     x_min, y_min, x_max, y_max = bbox
        #
        #     x_min = np.clip(x_min, 0, original_width - 1)
        #     x_max = np.clip(x_max, x_min + 1, original_width - 1)
        #
        #     if x_min >= x_max:
        #         continue
        #
        #     y_min = np.clip(y_min, 0, original_height - 1)
        #     y_max = np.clip(y_max, y_min + 1, original_height - 1)
        #
        #     if y_min >= y_max:
        #         continue
        #
        #     annotations += [
        #         {
        #             "bbox": bbox.tolist(),
        #             "score": scores[box_id],
        #             "landmarks": landmarks[box_id].reshape(-1, 2).tolist(),
        #         }
        #     ]
        #
        # return annotations





# size = 1280
#
# model = get_model("resnet50_2020-07-20", max_size=1280, device="cpu")
#
x = torch.randn(1, 3, 1280, 1280)

model = M()
out = model(x)
#

# traced = torch.jit.script(model)

#
# #
# #
torch.onnx.export(model, x, "retinaface.onnx", verbose=True, opset_version=12,
                  input_names = ['input'],
                  export_params=True,
                  do_constant_folding=True
                  )
#
# onnx_model = torch.onnx.lo

onnx_model = onnx.load("retinaface.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

ort_session = ort.InferenceSession('retinaface.onnx')

outputs = ort_session.run(None, {'input': np.random.randn(1, 3, 1280, 1280).astype(np.float32)})

print(outputs[0])