import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.utils import state_dict_from_disk
from iglovikov_helper_functions.utils.image_utils import pad_to_size, unpad_from_size
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.ops import nms
from tqdm import tqdm

from retinaface.box_utils import decode, decode_landm
from retinaface.utils import tensor_from_rgb_image, vis_annotations


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path with images.", required=True)
    arg("-c", "--config_path", type=Path, help="Path to config.", required=True)
    arg("-o", "--output_path", type=Path, help="Path to save jsons.", required=True)
    arg("-v", "--visualize", action="store_true", help="Visualize predictions")
    arg("-m", "--max_size", type=int, help="Resize the largest side to this number", default=960)
    arg("-b", "--batch_size", type=int, help="batch_size", default=1)
    arg("-j", "--num_workers", type=int, help="num_workers", default=12)
    arg("--confidence_threshold", default=0.7, type=float, help="confidence_threshold")
    arg("--nms_threshold", default=0.4, type=float, help="nms_threshold")
    arg("-w", "--weight_path", type=str, help="Path to weights.", required=True)
    arg("--keep_top_k", default=750, type=int, help="keep_top_k")
    arg("--world_size", default=-1, type=int, help="number of nodes for distributed training")
    arg("--local_rank", default=-1, type=int, help="node rank for distributed training")
    arg("--fp16", action="store_true", help="Use fp6")
    arg("--folder_in_name", action="store_true", help="Add folder to the saved labels.")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(
        self, file_paths: List[Path], max_size: int, transform: albu.Compose
    ) -> None:  # pylint: disable=W0231
        self.file_paths = file_paths
        self.transform = transform
        self.max_size = max_size
        self.resize = albu.LongestMaxSize(max_size=max_size, p=1)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        image_path = self.file_paths[idx]

        image = np.array(Image.open(image_path))

        image_height, image_width = image.shape[:2]

        image = self.resize(image=image)["image"]

        paded = pad_to_size(target_size=(self.max_size, self.max_size), image=image)

        image = paded["image"]
        pads = paded["pads"]

        image = self.transform(image=image)["image"]

        return {
            "torched_image": tensor_from_rgb_image(image),
            "image_path": str(image_path),
            "pads": np.array(pads),
            "image_height": image_height,
            "image_width": image_width,
        }


def unnormalize(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for c in range(image.shape[-1]):
        image[:, :, c] *= std[c]  # type: ignore
        image[:, :, c] += mean[c]  # type: ignore
        image[:, :, c] *= 255  # type: ignore

    return image


def process_predictions(
    prediction: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    original_shapes: List[Tuple[int, int]],
    input_shape: Tuple[int, int, int, int],
    pads: Tuple[int, int, int, int],
    confidence_threshold: float,
    nms_threshold: float,
    prior_box: torch.Tensor,
    variance: Tuple[float, float],
    keep_top_k: bool,
) -> List[List[Dict[str, Union[float, List[float]]]]]:
    loc, conf, land = prediction

    conf = F.softmax(conf, dim=-1)

    result: List[List[Dict[str, Union[List[float], float]]]] = []

    batch_size, _, image_height, image_width = input_shape

    scale1 = torch.from_numpy(np.tile([image_width, image_height], 5)).to(loc.device)
    scale = torch.from_numpy(np.tile([image_width, image_height], 2)).to(loc.device)

    for batch_id in range(batch_size):
        annotations: List[Dict[str, Union[List, float]]] = []

        boxes = decode(loc.data[batch_id], prior_box.to(loc.device), variance)

        boxes *= scale
        scores = conf[batch_id][:, 1]

        landmarks = decode_landm(land.data[batch_id], prior_box.to(land.device), variance)
        landmarks *= scale1

        # ignore low scores
        valid_index = torch.where(scores > confidence_threshold)[0]
        boxes = boxes[valid_index]
        landmarks = landmarks[valid_index]
        scores = scores[valid_index]

        order = scores.argsort(descending=True)

        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # do NMS
        keep = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep, :].int()

        if boxes.shape[0] == 0:
            result += [[{"bbox": [], "score": -1, "landmarks": []}]]
            continue

        landmarks = landmarks[keep]

        scores = scores[keep].cpu().numpy().astype(np.float64)[:keep_top_k]
        boxes = boxes.cpu().numpy()[:keep_top_k, :]
        landmarks = landmarks.cpu().numpy()[:keep_top_k, :]
        landmarks = landmarks.reshape([-1, 2])

        if pads is None:
            pads_numpy = np.array([0, 0, 0, 0])
        else:
            pads_numpy = pads[batch_id]

        unpadded = unpad_from_size(pads_numpy, bboxes=boxes, keypoints=landmarks)

        resize_coeff = max(original_shapes[batch_id]) / max(image_height, image_width)

        boxes = (unpadded["bboxes"] * resize_coeff).astype(int)
        landmarks = (unpadded["keypoints"].reshape(-1, 10) * resize_coeff).astype(int)

        for crop_id, bbox in enumerate(boxes):
            annotations += [
                {
                    "bbox": bbox.tolist(),
                    "score": float(scores[crop_id]),
                    "landmarks": landmarks[crop_id].reshape(-1, 2).tolist(),
                }
            ]

        result += [annotations]

    return result


def main() -> None:
    args = get_args()
    torch.distributed.init_process_group(backend="nccl")

    with args.config_path.open() as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    hparams.update(
        {
            "json_path": args.output_path,
            "visualize": args.visualize,
            "confidence_threshold": args.confidence_threshold,
            "nms_threshold": args.nms_threshold,
            "keep_top_k": args.keep_top_k,
            "local_rank": args.local_rank,
            "prior_box": object_from_dict(hparams["prior_box"], image_size=[args.max_size, args.max_size]),
            "fp16": args.fp16,
            "folder_in_name": args.folder_in_name,
        }
    )

    if args.visualize:
        output_vis_path = args.output_path / "viz"
        output_vis_path.mkdir(parents=True, exist_ok=True)
        hparams["output_vis_path"] = output_vis_path

    output_label_path = args.output_path / "labels"
    output_label_path.mkdir(parents=True, exist_ok=True)
    hparams["output_label_path"] = output_label_path

    device = torch.device("cuda", args.local_rank)

    model = object_from_dict(hparams["model"])
    model = model.to(device)

    if args.fp16:
        model = model.half()

    corrections: Dict[str, str] = {"model.": ""}
    state_dict = state_dict_from_disk(file_path=args.weight_path, rename_in_layers=corrections)
    model.load_state_dict(state_dict)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    file_paths = list(args.input_path.rglob("*.jpg"))

    dataset = InferenceDataset(file_paths, max_size=args.max_size, transform=from_dict(hparams["test_aug"]))

    sampler: DistributedSampler = DistributedSampler(dataset, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        sampler=sampler,
    )

    predict(dataloader, model, hparams, device)


def predict(dataloader: torch.utils.data.DataLoader, model: nn.Module, hparams: dict, device: torch.device) -> None:
    model.eval()

    if hparams["local_rank"] == 0:
        loader = tqdm(dataloader)
    else:
        loader = dataloader

    with torch.no_grad():
        for batch in loader:
            torched_images = batch["torched_image"]  # images that are rescaled and padded

            if hparams["fp16"]:
                torched_images = torched_images.half()

            pads = batch["pads"]
            image_paths = batch["image_path"]
            image_heights = batch["image_height"]
            image_widths = batch["image_width"]

            batch_size = torched_images.shape[0]

            image_heights = image_heights.cpu().numpy()
            image_widths = image_widths.cpu().numpy()

            original_shapes = list(zip(image_heights, image_widths))

            prediction = model(torched_images.to(device))

            output_annotations = process_predictions(
                prediction=prediction,
                original_shapes=original_shapes,
                input_shape=torched_images.shape,
                pads=pads.cpu().numpy(),
                confidence_threshold=hparams["confidence_threshold"],
                nms_threshold=hparams["nms_threshold"],
                prior_box=hparams["prior_box"],
                variance=hparams["test_parameters"]["variance"],
                keep_top_k=hparams["keep_top_k"],
            )

            for batch_id in range(batch_size):
                annotations = output_annotations[batch_id]
                if not annotations[0]["bbox"]:
                    continue

                folder_name = Path(image_paths[batch_id]).parent.name
                file_name = Path(image_paths[batch_id]).name
                file_id = Path(image_paths[batch_id]).stem

                predictions = {
                    "file_name": file_name,
                    "annotations": annotations,
                    "file_path": str(Path(folder_name) / file_name),
                }

                (hparams["output_label_path"] / folder_name).mkdir(exist_ok=True, parents=True)
                result_path = hparams["output_label_path"] / folder_name / f"{file_id}.json"

                with result_path.open("w") as f:
                    json.dump(predictions, f, indent=2)

                if hparams["visualize"]:
                    normalized_image = np.transpose(torched_images[batch_id].cpu().numpy(), (1, 2, 0))
                    image = unnormalize(normalized_image)
                    unpadded = unpad_from_size(pads[batch_id].cpu().numpy(), image)

                    original_image_height = image_heights[batch_id].item()
                    original_image_width = image_widths[batch_id].item()

                    image = cv2.resize(
                        unpadded["image"].astype(np.uint8), (original_image_width, original_image_height)
                    )

                    image = vis_annotations(image, annotations=annotations)  # type: ignore

                    (hparams["output_vis_path"] / folder_name).mkdir(exist_ok=True, parents=True)
                    result_path = hparams["output_vis_path"] / folder_name / f"{file_id}.jpg"

                    cv2.imwrite(str(result_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
