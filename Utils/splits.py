"""Whole-video train/validation selection and checkpoint split compatibility."""

from pathlib import Path
from Utils import config
from Utils.data import _sequence_info
from Utils.detections import detection_settings

SPLIT_PROTOCOL = "whole_video_v1"


def video_key(sequence):
    """Ignore MOT detector suffixes when checking whether videos overlap."""
    if not isinstance(sequence, str) or not sequence or Path(sequence).name != sequence or sequence in (".", ".."):
        raise ValueError("Sequences must be folder names, not paths")
    base, _, suffix = sequence.rpartition("-")
    return base if suffix.upper() in ("FRCNN", "DPM", "SDP") else sequence


def validate_video_splits():
    """Reject empty lists, repeated names, or any shared base video across sets."""
    groups = {}
    for name in ("TRAIN_SEQUENCES", "VAL_SEQUENCES"):
        sequences = getattr(config, name)
        if not isinstance(sequences, (list, tuple)) or not sequences:
            raise ValueError(f"{name} must be a nonempty list of sequence folder names")
        groups[name] = {video_key(sequence) for sequence in sequences}
        if len(set(sequences)) != len(sequences):
            raise ValueError(f"{name} contains duplicate sequence names")
    overlap = groups["TRAIN_SEQUENCES"] & groups["VAL_SEQUENCES"]
    if overlap:
        raise ValueError("Training/validation share videos: " + ", ".join(sorted(overlap))
                         + ". Keep all detector variants of each video in one set.")


def full_frame_range(sequence_path):
    """Return the entire video as inclusive original frame IDs."""
    return 1, _sequence_info(sequence_path).getint("seqLength")


def split_manifest():
    """Record dataset location, membership and full ranges; reject old checkpoints.

    Sorting makes list order irrelevant. No images or GT are read. A changed
    video assignment or dataset location requires a new training run.
    """
    validate_video_splits()
    return {
        "protocol": SPLIT_PROTOCOL,
        "dataset_root": str(Path(config.DATASET_ROOT).resolve()),
        "dataset_split": config.DATASET_SPLIT,
        "detections": detection_settings(),
        "sequences": {
            split: {sequence: list(full_frame_range(Path(config.DATASET_ROOT) / config.DATASET_SPLIT / sequence))
                    for sequence in sorted(sequences)}
            for split, sequences in (("train", config.TRAIN_SEQUENCES), ("val", config.VAL_SEQUENCES))
        },
    }


def check_checkpoint_split(checkpoint):
    """Prevent temporal-split or differently trained weights being reported as clean validation."""
    if checkpoint.get("data_split") != split_manifest():
        raise ValueError("Checkpoint uses an old/different video split or detection source. Train a fresh RUN_NAME "
                         "with the current video lists and detection settings.")
