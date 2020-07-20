import argparse
import json
from pathlib import Path
from typing import Dict, List

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.utils.image_utils import load_rgb
from jpeg4py import JPEGRuntimeError
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
from tqdm import tqdm

from retinaface.box_utils import decode, decode_landm
from retinaface.train import RetinaFace
from retinaface.utils import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description="Retinaface")
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-w", "--weights", help="Saved weights")
    arg("-i", "--input_path", type=Path, help="Path where images are stored", required=True)
    arg(
        "-o",
        "--output_path",
        type=Path,
        help="Path where results will be saved: " "images folder for images and" "labels folder for bounding boxes",
        required=True,
    )
    arg("--cpu", action="store_true", default=False, help="Use cpu inference")
    arg("--confidence_threshold", default=0.7, type=float, help="confidence_threshold")
    arg("--nms_threshold", default=0.4, type=float, help="nms_threshold")

    arg("-j", "--num_workers", type=int, help="Number of CPU threads", default=1)
    arg("-s", "--save_crops", action="store_true", default=False, help="If we want to store crops.")
    arg("-b", "--save_boxes", action="store_true", default=False, help="If we want to store bounding boxes.")
    arg("--origin_size", action="store_true", help="Whether use origin image size to evaluate")
    arg("--fp16", action="store_true", help="Whether use fp16")
    arg("-v", "--visualize", action="store_true", help="Visualize predictions")
    arg("-t", "--target_size", type=int, help="Target size", default=1600)
    arg("-m", "--max_size", type=int, help="Target size", default=2150)
    arg("--keep_top_k", default=750, type=int, help="keep_top_k")
    arg(
        "--batch_size",
        type=int,
        help="Size of the batch size. Use non 1 value only if you are sure that" "all images are of the same size.",
        default=1,
    )
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(
        self, file_paths: List[Path], origin_size: int, target_size: int, max_size: int, transform: albu.Compose
    ) -> None:
        self.file_paths = file_paths
        self.transform = transform
        self.origin_size = origin_size
        self.target_size = target_size
        self.max_size = max_size

    def __len__(self) -> int:

        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]

        try:
            raw_image = load_rgb(image_path, lib="jpeg4py")
        except JPEGRuntimeError:
            raw_image = load_rgb(image_path, lib="cv2")

        image = raw_image.astype(np.float32)

        if self.origin_size:
            resize = 1
        else:
            # testing scale
            im_shape = image.shape
            image_size_min = np.min(im_shape[:2])
            image_size_max = np.max(im_shape[:2])
            resize = float(self.target_size) / float(image_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * image_size_max) > self.max_size:
                resize = float(self.max_size) / float(image_size_max)

            image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        image = self.transform(image=image)["image"]

        return {
            "torched_image": tensor_from_rgb_image(image),
            "resize": resize,
            "raw_image": raw_image,
            "image_path": str(image_path),
        }


def get_model(hparams, weights_path, is_fp16, device):
    net = RetinaFace(hparams)

    corrections: Dict[str, str] = {"module": "model"}

    checkpoint = load_checkpoint(file_path=weights_path, rename_in_layers=corrections)

    net.load_state_dict(checkpoint["state_dict"])

    net.eval()
    if is_fp16:
        net = net.half()

    print("Finished loading model!")
    cudnn.benchmark = True
    return net.to(device)


def prepare_output_folders(
    output_vis_path: Path,
    output_label_path: Path,
    output_image_path: Path,
    is_labels: bool,
    is_crops: bool,
    is_visualize: bool,
) -> None:
    if is_visualize:
        output_vis_path.mkdir(exist_ok=True, parents=True)

    if is_labels:
        output_label_path.mkdir(exist_ok=True, parents=True)

    if is_crops:
        output_image_path.mkdir(exist_ok=True, parents=True)


def main():
    args = get_args()
    torch.set_grad_enabled(False)

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device("cpu" if args.cpu else "cuda")

    net = get_model(hparams, args.weights, args.fp16, device)

    file_paths = sorted(args.input_path.rglob("*"))

    output_path = args.output_path
    output_vis_path = output_path / "viz"
    output_label_path = output_path / "labels"
    output_image_path = output_path / "images"

    prepare_output_folders(
        output_vis_path, output_label_path, output_image_path, args.save_boxes, args.save_crops, args.visualize
    )

    transform = from_dict(hparams["test_aug"])

    test_loader = DataLoader(
        InferenceDataset(
            file_paths, args.origin_size, transform=transform, target_size=args.target_size, max_size=args.max_size
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        for raw_input in tqdm(test_loader):
            torched_images = raw_input["torched_image"].type(net.dtype)

            resizes = raw_input["resize"]
            image_paths = raw_input["image_path"]
            raw_images = raw_input["raw_image"]

            labels = []

            if (
                args.batch_size == 1
                and args.save_boxes
                and (output_label_path / f"{Path(image_paths[0]).stem}.json").exists()
            ):
                continue

            loc, conf, land = net(torched_images.to(device))  # forward pass

            conf = F.softmax(conf, dim=-1)

            batch_size = torched_images.shape[0]

            image_height, image_width = torched_images.shape[2:]

            scale1 = torch.from_numpy(np.tile([image_width, image_height], 5)).to(device)
            scale = torch.from_numpy(np.tile([image_width, image_height], 2)).to(device)

            priors = object_from_dict(hparams["prior_box"], image_size=(image_height, image_width)).to(loc.device)

            for batch_id in range(batch_size):
                image_path = image_paths[batch_id]
                file_id = Path(image_path).stem
                raw_image = raw_images[batch_id]

                resize = resizes[batch_id].float()

                boxes = decode(loc.data[batch_id], priors, hparams["test_parameters"]["variance"])

                boxes *= scale / resize
                scores = conf[batch_id][:, 1]

                landmarks = decode_landm(land.data[batch_id], priors, hparams["test_parameters"]["variance"])
                landmarks *= scale1 / resize

                # ignore low scores
                valid_index = torch.where(scores > args.confidence_threshold)[0]
                boxes = boxes[valid_index]
                landmarks = landmarks[valid_index]
                scores = scores[valid_index]

                order = scores.argsort(descending=True)

                boxes = boxes[order]
                landmarks = landmarks[order]
                scores = scores[order]

                # do NMS
                keep = nms(boxes, scores, args.nms_threshold)
                boxes = boxes[keep, :].int()

                if boxes.shape[0] == 0:
                    continue

                landmarks = landmarks[keep].int()
                scores = scores[keep].cpu().numpy().astype(np.float64)

                boxes = boxes[: args.keep_top_k]
                landmarks = landmarks[: args.keep_top_k]
                scores = scores[: args.keep_top_k]

                if args.visualize:
                    vis_image = raw_image.cpu().numpy().copy()

                    for crop_id, bbox in enumerate(boxes):
                        landms = landmarks[crop_id].cpu().numpy().reshape([5, 2])

                        colors = [(255, 0, 0), (128, 255, 0), (255, 178, 102), (102, 128, 255), (0, 255, 255)]
                        for i, (x, y) in enumerate(landms):
                            vis_image = cv2.circle(vis_image, (x, y), radius=5, color=colors[i], thickness=5)

                for crop_id, bbox in enumerate(boxes):
                    bbox = bbox.cpu().numpy()

                    labels += [
                        {
                            "crop_id": crop_id,
                            "bbox": bbox.tolist(),
                            "score": scores[crop_id],
                            "landmarks": landmarks[crop_id].tolist(),
                        }
                    ]

                    if args.save_crops:
                        x_min, y_min, x_max, y_max = bbox

                        x_min = np.clip(x_min, 0, x_max - 1)
                        y_min = np.clip(y_min, 0, y_max - 1)

                        crop = raw_image[y_min:y_max, x_min:x_max].cpu().numpy()

                        if args.visualize:
                            vis_image = cv2.rectangle(
                                vis_image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3
                            )

                        target_folder = output_image_path / f"{file_id}"
                        target_folder.mkdir(exist_ok=True, parents=True)

                        crop_file_path = target_folder / f"{file_id}_{crop_id}.jpg"

                        if crop_file_path.exists():
                            continue

                        # Some faces are really small. Resize them for better visual inspection
                        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)

                        cv2.imwrite(str(crop_file_path), cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                if args.visualize:
                    cv2.imwrite(str(output_vis_path / f"{file_id}.jpg"), cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))

                if args.save_boxes:
                    result = {
                        "file_path": image_path,
                        "file_id": file_id,
                        "bboxes": labels,
                    }

                    with open(output_label_path / f"{file_id}.json", "w") as f:
                        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
