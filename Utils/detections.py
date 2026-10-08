"""Resolve supplied/cached boxes and reject incomplete or incompatible caches."""

import hashlib
import json
from pathlib import Path

from Utils import config
from Utils.data import _sequence_info


def detector_settings():
    """Recipe for cache compatibility; device is not part of the recipe."""
    if not 0 <= config.DETECTOR_MIN_SCORE <= 1:
        raise ValueError("DETECTOR_MIN_SCORE must be between 0 and 1")
    mode = config.DETECTOR_INPUT_MODE
    if mode not in ("square", "original"):
        raise ValueError("DETECTOR_INPUT_MODE must be square or original")
    if (not isinstance(config.DETECTOR_IMAGE_SIZE, int) or config.DETECTOR_IMAGE_SIZE < 32
            or config.DETECTOR_IMAGE_SIZE % 32):
        raise ValueError("DETECTOR_IMAGE_SIZE must be a positive multiple of 32")
    if config.BOX_ORIGIN != 1:
        raise ValueError("Cached RT-DETRv2 detections use MOT's BOX_ORIGIN=1")
    recipe = {"format": "mot_person_v1", "model": config.DETECTOR_MODEL_NAME,
            "revision": config.DETECTOR_REVISION, "image_size": config.DETECTOR_IMAGE_SIZE,
            "min_score": config.DETECTOR_MIN_SCORE, "class": "person", "box_origin": 1,
            "precision": "float32_tf32_disabled", "postprocessing": "official_topk_no_nms_clip"}
    # Keep the existing square recipe identical so old caches remain usable.
    if mode == "original":
        recipe.update(image_size=None, input_mode="original", padding="bottom_right_zero_multiple_32",
                      anchors="dynamic", position_embeddings="dynamic")
    elif config.DETECTOR_IMAGE_SIZE != 640:
        recipe.update(anchors="dynamic", position_embeddings="dynamic")
    return recipe


def detection_settings():
    """Record box source in the split/checkpoint metadata, preventing silent switches."""
    if config.DETECTION_BACKEND == "supplied":
        source = {"file": str(config.DETECTION_FILE)}
    elif config.DETECTION_BACKEND == "tracktor":
        source = {"archive_sha256": config.TRACKTOR_ARCHIVE_SHA256,
                  "recipe": "tracktor_prepr_det", "upstream_ids": "discarded",
                  "missing_confidence": "minus_one_to_one", "box_origin": 1}
    elif config.DETECTION_BACKEND == "rtdetrv2":
        source = detector_settings()
    else:
        raise ValueError("DETECTION_BACKEND must be supplied, tracktor, or rtdetrv2")
    return {"backend": config.DETECTION_BACKEND, "source": source,
            "box_origin": config.BOX_ORIGIN, "read_min_score": config.DETECTION_MIN_SCORE}


def cache_path(sequence_path):
    """Keep generated boxes outside the dataset: cache/<dataset-split>/<sequence>."""
    sequence_path = Path(sequence_path)
    return Path(config.DETECTION_CACHE_DIR) / sequence_path.parent.name / sequence_path.name / "det.txt"


def cache_identity(sequence_path):
    """Expected full-video coverage and recipe, without reading labels."""
    sequence_path = Path(sequence_path)
    info = _sequence_info(sequence_path)
    return {"sequence_path": str(sequence_path.resolve()), "frames": info.getint("seqLength"),
            "width": info.getint("imWidth"), "height": info.getint("imHeight"),
            "detector": detector_settings()}


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_cache(sequence_path):
    """Return a complete cache path; empty files are valid if all frames were processed."""
    path = cache_path(sequence_path)
    summary = path.with_name("summary.json")
    if not path.is_file() or not summary.is_file():
        raise FileNotFoundError(f"Missing completed detection cache: {path}. Run python detect.py first.")
    metadata = json.loads(summary.read_text())
    if (metadata.get("complete") is not True or metadata.get("identity") != cache_identity(sequence_path)
            or metadata.get("sha256") != file_digest(path)):
        raise ValueError(f"Incomplete, changed, or incompatible detection cache: {path}. "
                         "Choose a new DETECTION_CACHE_DIR and rerun detect.py.")
    return path.resolve()


def detection_path(sequence_path):
    """Shared entry point for training and tracking; never fall back silently."""
    detection_settings()  # Validate the selected backend even before loading a model.
    if config.DETECTION_BACKEND == "tracktor":
        return validate_tracktor(sequence_path)
    if config.DETECTION_BACKEND == "rtdetrv2":
        return validate_cache(sequence_path)
    path = Path(config.DETECTION_FILE)
    path = path if path.is_absolute() else Path(sequence_path) / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing supplied detections: {path}")
    return path.resolve()


def tracktor_path(sequence_path):
    sequence_path = Path(sequence_path)
    return Path(config.TRACKTOR_CACHE_DIR) / sequence_path.parent.name / sequence_path.name / "det.txt"


def tracktor_identity(sequence_path):
    if config.BOX_ORIGIN != 1:
        raise ValueError("Imported Tracktor boxes use MOT's BOX_ORIGIN=1")
    sequence_path = Path(sequence_path)
    info = _sequence_info(sequence_path)
    return {"sequence_path": str(sequence_path.resolve()), "frames": info.getint("seqLength"),
            "source": {"archive_sha256": config.TRACKTOR_ARCHIVE_SHA256,
                       "recipe": "tracktor_prepr_det", "missing_confidence": "minus_one_to_one",
                       "upstream_ids": "discarded", "box_origin": 1}}


def validate_tracktor(sequence_path):
    path = tracktor_path(sequence_path)
    summary = path.with_name("summary.json")
    if not path.is_file() or not summary.is_file():
        raise FileNotFoundError(f"Missing Tracktor cache {path}. Run python import_tracktor.py.")
    metadata = json.loads(summary.read_text())
    if (metadata.get("complete") is not True or metadata.get("identity") != tracktor_identity(sequence_path)
            or metadata.get("sha256") != file_digest(path)):
        raise ValueError(f"Incomplete, changed, or incompatible Tracktor cache: {path}")
    return path.resolve()
