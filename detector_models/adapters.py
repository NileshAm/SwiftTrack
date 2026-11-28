"""Thin adapters around official pretrained implementations."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

from .registry import get


class AvailabilityError(RuntimeError):
    pass


class YOLO26M:
    def __init__(self, args):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise AvailabilityError("Install/upgrade the official ultralytics package for YOLO26.") from error
        self.model = YOLO(args.weights or "yolo26m.pt")
        self.device, self.size, self.floor, self.nms = args.device, args.image_size, args.cache_floor, args.nms_iou
        self.recipe = {"format": "mot_person_v1", "backend": "yolo26_m",
                       "model": "Ultralytics YOLO26-M", "weights": str(args.weights or "yolo26m.pt"),
                       "training_data": "COCO 2017", "image_size": self.size,
                       "min_score": self.floor, "nms_iou": self.nms, "person_class_id": 0,
                       "box_origin": 1, "implementation": get("yolo26_m").official_repo,
                       "postprocessing": "official one-to-many head plus Ultralytics NMS"}

    @torch.inference_mode()
    def predict(self, image):
        result = self.model.predict(image, imgsz=self.size, conf=self.floor, iou=self.nms,
                                    classes=[0], device=self.device, verbose=False, nms=True)[0]
        if result.boxes is None:
            return np.empty((0, 4), np.float32), np.empty(0, np.float32)
        return (result.boxes.xyxy.detach().cpu().numpy().astype(np.float32),
                result.boxes.conf.detach().cpu().numpy().astype(np.float32))


class DEIMv2S:
    def __init__(self, args):
        repo = Path(args.official_repo or "detector_models/official/DEIMv2").resolve()
        checkpoint = Path(args.weights).resolve() if args.weights else None
        config = repo / "configs/deimv2/deimv2_dinov3_s_coco.yml"
        if not (repo / "engine").is_dir():
            raise AvailabilityError(f"Missing official DEIMv2 checkout: {repo}")
        if checkpoint is None or not checkpoint.is_file():
            raise AvailabilityError("Pass --weights to the official DEIMv2-S .pth checkpoint.")
        sys.path.insert(0, str(repo))
        try:
            from engine.core import YAMLConfig
        except Exception as error:
            raise AvailabilityError(f"Could not import official DEIMv2 runtime from {repo}: {error}") from error
        cfg = YAMLConfig(str(config), resume=str(checkpoint))
        cfg.yaml_cfg["DINOv3STAs"]["pretrained"] = False
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved["ema"]["module"] if "ema" in saved else saved["model"]
        cfg.model.load_state_dict(state)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.post = cfg.postprocessor.deploy()
            def forward(self, images, sizes):
                return self.post(self.model(images), sizes)

        self.model = Model().to(args.device).eval()
        self.device, self.size, self.floor = args.device, args.image_size, args.cache_floor
        self.recipe = {"format": "mot_person_v1", "backend": "deimv2_s",
                       "model": "DEIMv2-DINOv3-S", "weights": str(checkpoint),
                       "training_data": "COCO 2017", "image_size": self.size,
                       "min_score": self.floor, "person_class_id": 0, "box_origin": 1,
                       "implementation": get("deimv2_s").official_repo,
                       "postprocessing": "official DEIMv2 deploy postprocessor"}

    @torch.inference_mode()
    def predict(self, image):
        from PIL import Image
        from torchvision.transforms import functional as F
        height, width = image.shape[:2]
        pil = Image.fromarray(image).resize((self.size, self.size))
        tensor = F.to_tensor(pil)
        tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        sizes = torch.tensor([[width, height]], device=self.device)
        labels, boxes, scores = self.model(tensor.unsqueeze(0).to(self.device), sizes)
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        keep = (labels == 0) & (scores >= self.floor)
        return (boxes[keep].detach().cpu().numpy().astype(np.float32),
                scores[keep].detach().cpu().numpy().astype(np.float32))



class RFDETRDetector:
    def __init__(self, args, size="medium"):
        try:
            from rfdetr import RFDETRMedium, RFDETRLarge
        except ImportError as error:
            raise AvailabilityError(
                "Install the official rfdetr package."
            ) from error

        cls = RFDETRMedium if size == "medium" else RFDETRLarge
        self.model = cls()
        self.floor = args.cache_floor
        self.size = size
        self.recipe = {
            "format": "mot_person_v1",
            "backend": f"rfdetr_{size}",
            "model": f"RF-DETR {size}",
            "training_data": "COCO",
            "min_score": self.floor,
            "person_class_id": 0,
            "box_origin": 1,
            "implementation": "https://github.com/roboflow/rf-detr",
            "postprocessing": "official RF-DETR inference",
        }

    @torch.inference_mode()
    def predict(self, image):
        detections = self.model.predict(image, threshold=self.floor)
        keep = detections.class_id == 0

        return (
            np.asarray(detections.xyxy[keep], dtype=np.float32),
            np.asarray(detections.confidence[keep], dtype=np.float32),
        )


class YOLO26L(YOLO26M):
    def __init__(self, args):
        args.weights = args.weights or "yolo26l.pt"
        super().__init__(args)
        self.recipe["backend"] = "yolo26_l"
        self.recipe["model"] = "Ultralytics YOLO26-L"
        self.recipe["weights"] = str(args.weights)


def load_detector(name, args):
    spec = get(name)
    if spec.runtime in ("legacy_crowddet", "legacy_detectron2"):
        raise AvailabilityError(spec.availability_note)
    if name == "rfdetr_medium":
        return RFDETRDetector(args, "medium")
    if name == "rfdetr_large":
        return RFDETRDetector(args, "large")
    if name == "yolo26_l":
        return YOLO26L(args)

    return {
        "deimv2": DEIMv2S,
        "ultralytics": YOLO26M,
    }[spec.runtime](args)

