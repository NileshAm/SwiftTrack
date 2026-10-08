"""Shared per-sequence tracking paths for validation, evaluation and test videos."""

from pathlib import Path
from Utils import config
from Utils.splits import full_frame_range, validate_video_splits


def tracking_folder(sequence, split=None):
    """Return the configured validation/test variant folder for one sequence."""
    split = config.TRACK_SPLIT if split is None else split
    if not sequence or Path(sequence).name != sequence or sequence in (".", ".."):
        raise ValueError("Sequence must be a folder name, not a path")
    if split == "test":
        root = config.TEST_OUTPUT_DIR
    elif split == "val":
        root = (config.TRACK_OUTPUT_DIR if split == config.TRACK_SPLIT
                else Path(config.OUTPUT_DIR) / "tracking" / split)
    else:
        raise ValueError("Tracking split must be val or test")
    return Path(root) / sequence


def tracking_context(sequence, split=None):
    """Resolve complete validation/test videos without reading GT."""
    split = config.TRACK_SPLIT if split is None else split
    folder = tracking_folder(sequence, split)
    test = split == "test"
    root = Path(config.TEST_DATASET_ROOT if test else config.DATASET_ROOT)
    dataset_split = config.TEST_DATASET_SPLIT if test else config.DATASET_SPLIT
    path = root / dataset_split / sequence
    validate_video_splits()
    choices = config.TEST_SEQUENCES if test else config.VAL_SEQUENCES
    if sequence not in choices:
        raise ValueError(f"{sequence} is not in the configured {split} video list")
    frame_range = full_frame_range(path)
    return {"split": split, "sequence_path": path, "frame_range": frame_range,
            "output_dir": folder, "dataset_root": root,
            "dataset_split": dataset_split,
            "save_video": config.SAVE_TEST_VIDEO if test else config.SAVE_VIDEO}
