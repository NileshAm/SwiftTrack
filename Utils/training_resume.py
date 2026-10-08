"""Epoch-boundary training recovery; no model initialization or dataset loading.

See Documentation/Implementation/23-resume-training.md. Checkpoints contain
history and random states so an interrupted CSV write cannot lose completed work.
"""
import csv
from datetime import datetime
import math
import random
import shutil

import numpy as np
import torch

from Utils import config
from Utils.splits import check_checkpoint_split
from Utils.track_state import state_settings
from model import MODEL_VERSION


# Output, device, epoch budget and learning rate may change on continuation.
RECIPE_SETTINGS = (
    "ENCODER_NAME", "USE_PRETRAINED_WEIGHTS", "FREEZE_ENCODER", "FEATURE_DIM",
    "GRAPH_HIDDEN_DIMS", "BOX_FORMAT", "BOX_ORIGIN", "DETECTION_FILE",
    "DETECTION_MIN_SCORE", "GT_FILE", "GT_MATCH_IOU", "CROP_SIZE", "CROP_BATCH_SIZE",
    "DETR_MODEL_NAME", "DETR_REVISION", "DETR_SHORT_SIDE", "DETR_MAX_SIZE",
    "DETR_GRADIENT_CHECKPOINTING", "TRAIN_BATCH_SIZE", "VAL_BATCH_SIZE",
    "TRAIN_SEQUENCES", "VAL_SEQUENCES", "LOSS_ALPHA", "WBCE_POSITIVE_WEIGHT",
    "WBCE_NEGATIVE_WEIGHT", "CONTRASTIVE_MARGIN", "NEGATIVE_RATIO",
    "HARD_NEGATIVE_FRACTION", "GRAD_CLIP_NORM", "SEED", "SINKHORN_ITERATIONS",
    "SINKHORN_SCORE_SCALE", "SINKHORN_UNMATCHED_LOGIT",
)


def load_resume(last_path, best_path, history_path, frame_limit):
    """Validate same-run checkpoints before any output writes or heavy inference.

    Saving best precedes saving last. If interrupted between those atomic writes,
    a newer best checkpoint is itself a complete epoch and can recover that epoch.
    Legacy checkpoints use CSV history; new checkpoints carry their own history.
    """
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {best_path}. Restore the run before resuming.")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    latest = torch.load(last_path, map_location="cpu", weights_only=True) if last_path.is_file() else best
    source = last_path if last_path.is_file() else best_path
    if best.get("epoch", 0) > latest.get("epoch", 0):
        latest, source = best, best_path
    for saved in (latest, best):
        if saved.get("model_version") != MODEL_VERSION:
            raise ValueError("Cannot resume a different model architecture; use a fresh RUN_NAME")
        check_checkpoint_split(saved)
        temporal = saved.get("temporal_sampling", {})
        if (temporal.get("protocol") != "chronological_track_state_v1"
                or temporal.get("max_frame_gap") != config.MAX_MISSED_FRAMES + 1
                or temporal.get("state_settings") != state_settings()):
            raise ValueError("Cannot resume with different motion/history/gate/look-back settings")
        if saved.get("training_frame_limit") != frame_limit:
            raise ValueError("Cannot resume with a different training frame limit")
        for name in RECIPE_SETTINGS:
            if saved.get("settings", {}).get(name) != getattr(config, name):
                raise ValueError(f"Cannot resume: config.{name} differs from the saved training recipe")
    epoch = latest.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Checkpoint has an invalid completed epoch")
    if "optimizer_state_dict" not in latest:
        raise ValueError("Checkpoint has no optimizer state; cannot continue training")
    if (best.get("epoch") != latest.get("best_epoch")
            or best.get("best_val_loss") != latest.get("best_val_loss")):
        raise ValueError("Best/latest checkpoints disagree; restore files from the same run")
    if not math.isfinite(float(latest["best_val_loss"])):
        raise ValueError("Checkpoint best loss is not finite")
    if "history" in latest:
        history = latest["history"]
    else:
        if not history_path.is_file():
            raise FileNotFoundError("Legacy checkpoint needs its original history.csv to resume")
        with history_path.open(newline="") as stream:
            history = list(csv.DictReader(stream))
        # A legacy last.pt may precede the CSV when SAVE_EVERY_EPOCH was false.
        # The caller backs up the complete CSV before restoring this prefix.
        history = [row for row in history if int(row["epoch"]) <= epoch]
    if [int(row["epoch"]) for row in history] != list(range(1, epoch + 1)):
        raise ValueError("Resume history must cover each completed epoch exactly once")
    losses = [float(row["val_total"]) for row in history]
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("Resume history contains non-finite validation losses")
    if (not math.isclose(min(losses), float(latest["best_val_loss"]), rel_tol=1e-9, abs_tol=1e-9)
            or losses.index(min(losses)) + 1 != latest["best_epoch"]):
        raise ValueError("History and best checkpoint disagree")
    return latest, history, source


def random_state():
    """Use only tensors and Python primitives supported by weights_only loading."""
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state):
    """Call after constructing/restoring the model, immediately before epochs."""
    random.setstate(state["python"])
    name, values, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA random states require the same GPU count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(checkpoint, path):
    """Publish a complete file atomically; interrupted writes leave the old file."""
    pending = path.with_name(path.name + ".pending")
    torch.save(checkpoint, pending)
    pending.replace(path)


def save_history(history, path):
    """Rewrite the recovered prefix plus new rows without duplicating epochs."""
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[-1]))
        writer.writeheader()
        writer.writerows(history)
    pending.replace(path)


def backup_run(output_dir, paths):
    """Preserve weights/history before a continuation replaces any run artifacts."""
    folder = output_dir / "resume_backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if path.is_file():
            shutil.copy2(path, folder / path.name)
    return folder
