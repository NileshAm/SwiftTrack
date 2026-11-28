"""Verified model registry. Never replace an unavailable model implicitly."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    title: str
    training_data: str
    official_repo: str
    checkpoint: str | None
    runtime: str
    input_size: int
    person_class: int
    availability_note: str = ""


SPECS = {
    "crowddet": DetectorSpec(
        "crowddet", "CrowdDet", "CrowdHuman",
        "https://github.com/megvii-model/CrowdDetection",
        None, "legacy_crowddet", 800, 1,
        "The official README publishes only an ImageNet ResNet-50 initialization, not a trained "
        "CrowdHuman detector checkpoint. It also pins Python 3.6/PyTorch 1.5/CUDA 10.1. "
        "This arm must be recorded unavailable unless an official detector checkpoint is supplied."),
    "iter_e2edet": DetectorSpec(
        "iter_e2edet", "Iter-E2EDET (Iter Sparse R-CNN)", "CrowdHuman",
        "https://github.com/megvii-research/Iter-E2EDET",
        "Official Google Drive checkpoint linked by the repository README",
        "legacy_detectron2", 800, 0,
        "The official implementation supports Linux/macOS and compiles its bundled Detectron2. "
        "It is not compatible with this native Windows/PyTorch 2.2 environment. Use WSL/Linux and "
        "pass its exported MOT cache back to this suite; do not substitute a COCO Detectron2 model."),
    "deimv2_s": DetectorSpec(
        "deimv2_s", "DEIMv2-S", "COCO 2017",
        "https://github.com/Intellindust-AI-Lab/DEIMv2",
        "Intellindust/DEIMv2_DINOv3_S_COCO", "deimv2", 640, 0),
    "yolo26_m": DetectorSpec(
        "yolo26_m", "YOLO26-M", "COCO 2017",
        "https://github.com/ultralytics/ultralytics",
        "yolo26m.pt", "ultralytics", 640, 0),
    
    "rfdetr_medium": DetectorSpec(
        "rfdetr_medium", "RF-DETR Medium", "COCO 2017",
        "https://github.com/roboflow/rf-detr",
        "Official RFDETRMedium weights",
        "rfdetr", 576, 0,
    ),
    "rfdetr_large": DetectorSpec(
        "rfdetr_large", "RF-DETR Large", "COCO 2017",
        "https://github.com/roboflow/rf-detr",
        "Official RFDETRLarge weights",
        "rfdetr", 704, 0,
    ),
    "yolo26_l": DetectorSpec(
        "yolo26_l", "YOLO26-L", "COCO 2017",
        "https://github.com/ultralytics/ultralytics",
        "yolo26l.pt",
        "ultralytics", 640, 0,
    ),

}

ORDER = tuple(SPECS)


def get(name):
    try:
        return SPECS[name]
    except KeyError as error:
        raise ValueError(f"Unknown detector {name!r}; choose {', '.join(ORDER)}") from error

