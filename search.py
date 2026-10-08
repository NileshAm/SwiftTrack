"""Edit the settings below, then run: python search.py

Each combination gets separate outputs and a ranked validation result.
Overrides exist only in this process; Utils/config.py is never written.
See Documentation/Implementation/20-parameter-search.md for examples.
"""

from contextlib import contextmanager
from datetime import datetime
import csv
import fcntl
import gc
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm

from Utils import config
from model import MODEL_VERSION


# ---------------- EDIT YOUR SEARCH HERE ----------------
MODE = "tracking"  # "train": fresh training per trial; "tracking": reuse SOURCE_CHECKPOINT.

# Keys are names from config.py. Every combination of the listed values is tried.
PARAM_GRID = {
    "MATCH_THRESHOLD": [0.05, 0.3, 0.5, 0.7],
}

# Independent of config.EPOCHS. Each count starts a NEW training run.
# [3, 5] tries 3 and 5 epochs; range(2, 6) tries 2, 3, 4, 5 epochs.
EPOCH_COUNTS = [3]
TRAIN_MAX_FRAMES_PER_SEQUENCE = None  # None = full videos; e.g. 200 = first 200 frames of EACH training video.
EVALUATE_CHECKPOINT = "best"  # "last": exact epoch count; "best": lowest validation loss.

# Used ONLY in MODE="tracking". Remove training-only keys (e.g. LEARNING_RATE)
# from PARAM_GRID in that mode. EPOCH_COUNTS is ignored; no training is performed.
SOURCE_CHECKPOINT = config.CHECKPOINT_PATH

SEARCH_ROOT = config.PROJECT_ROOT / "outputs" / "searches"
SEARCH_NAME = "tracking_match_thresholds2"  # A timestamp is added automatically.
RESUME = True  # Reuse the newest matching search; False always starts a new folder.
RESUME_DIR = None  # Optional Path("outputs/searches/<existing-search>"); requires matching settings.
RANK_BENCHMARK = "MOT17"  # Select one benchmark; MOT17/MOT20 scores are not averaged.
RANK_METRIC = "HOTA"  # "HOTA", "IDF1", or "MOTA"; higher wins.
SHOW_PROGRESS = True
SAVE_SEARCH_PLOTS = False  # Save plots per trial if desired; videos are skipped.
# -------------------------------------------------------


# Keep meaningless changes (paths, data membership, unused training settings)
# out of the grid. Extend TRAIN_PARAMETERS when adding new training controls.
TRACKING_PARAMETERS = {
    "MATCH_THRESHOLD", "NEW_TRACK_MIN_SCORE", "TRACK_CONFIRM_MIN_HITS",
    "TRACK_CONFIRM_WINDOW", "ASSOCIATION_DISTANCE_PX", "USE_MOTION_PREDICTION",
    "USE_APPEARANCE_HISTORY", "APPEARANCE_HISTORY_SIZE", "APPEARANCE_HISTORY_MIN_SCORE",
}
TRAIN_PARAMETERS = TRACKING_PARAMETERS | {
    "LEARNING_RATE", "TRAIN_BATCH_SIZE", "VAL_BATCH_SIZE", "LOSS_ALPHA",
    "WBCE_POSITIVE_WEIGHT", "WBCE_NEGATIVE_WEIGHT", "CONTRASTIVE_MARGIN",
    "NEGATIVE_RATIO", "HARD_NEGATIVE_FRACTION", "LOOKBACK_FRAMES", "GRAD_CLIP_NORM",
    "GRAPH_HIDDEN_DIMS", "SINKHORN_ITERATIONS", "SINKHORN_SCORE_SCALE",
    "SINKHORN_UNMATCHED_LOGIT", "FREEZE_ENCODER", "DETR_GRADIENT_CHECKPOINTING",
    "GT_MATCH_IOU", "SEED",
}


@contextmanager
def temporary_settings(overrides):
    """Apply known config attributes in memory; restore them even on failure."""
    original = {name: getattr(config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)


def build_trials():
    """Validate the editable grid and expand parameters × training epoch counts."""
    if MODE not in {"train", "tracking"}:
        raise ValueError('MODE must be "train" or "tracking"')
    if MODE == "train" and EVALUATE_CHECKPOINT not in {"last", "best"}:
        raise ValueError('EVALUATE_CHECKPOINT must be "last" or "best"')
    if RANK_METRIC not in {"HOTA", "IDF1", "MOTA"}:
        raise ValueError("RANK_METRIC must be HOTA, IDF1, or MOTA")
    if RANK_BENCHMARK not in {"MOT17", "MOT20"}:
        raise ValueError("RANK_BENCHMARK must be MOT17 or MOT20")
    if not SEARCH_NAME or Path(SEARCH_NAME).name != SEARCH_NAME or SEARCH_NAME in {".", ".."}:
        raise ValueError("SEARCH_NAME must be a folder name, not a path")
    allowed = TRAIN_PARAMETERS if MODE == "train" else TRACKING_PARAMETERS
    if not isinstance(PARAM_GRID, dict):
        raise ValueError("PARAM_GRID must map parameter names to lists of values")
    for name, values in PARAM_GRID.items():
        if name not in allowed:
            raise ValueError(f"{name} cannot be searched in {MODE} mode. "
                             "Use EPOCH_COUNTS for epochs and LOOKBACK_FRAMES for the training window. "
                             "Tracking mode accepts only TRACKING_PARAMETERS listed in search.py.")
        if not isinstance(values, (list, tuple, range)) or not values:
            raise ValueError(f"{name}: supply a nonempty list, tuple, or range of values")
        # Fail before starting expensive training for values that cannot be recorded.
        json.dumps(values if not isinstance(values, range) else list(values), allow_nan=False)
    if MODE == "train":
        limit = TRAIN_MAX_FRAMES_PER_SEQUENCE
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 2):
            raise ValueError("TRAIN_MAX_FRAMES_PER_SEQUENCE must be None or an integer of at least 2")
        if (not isinstance(EPOCH_COUNTS, (list, tuple, range)) or not EPOCH_COUNTS
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in EPOCH_COUNTS)):
            raise ValueError("EPOCH_COUNTS must be a nonempty list/range of positive integers")
    epochs = list(EPOCH_COUNTS) if MODE == "train" else [None]
    trials = []
    seen = set()
    for values in product(*PARAM_GRID.values()):
        parameters = dict(zip(PARAM_GRID, values))
        for count in epochs:
            key = json.dumps([parameters, count], sort_keys=True)
            if key not in seen:
                trials.append({"parameters": parameters.copy(), "epochs": count})
                seen.add(key)
    return trials


def trial_settings(trial, folder):
    """Resolve dependent paths/window settings and restrict work to validation."""
    settings = dict(trial["parameters"])
    if "LOOKBACK_FRAMES" in settings:
        gap = settings["LOOKBACK_FRAMES"]
        if isinstance(gap, bool) or not isinstance(gap, int) or gap < 1:
            raise ValueError("LOOKBACK_FRAMES must be a positive integer")
        settings["MAX_MISSED_FRAMES"] = gap - 1
    settings.update(
        RUN_NAME=folder.name, OUTPUT_DIR=folder,
        CHECKPOINT_PATH=folder / "model.pt" if MODE == "train" else Path(SOURCE_CHECKPOINT).resolve(),
        TRACKING_VARIANT="search", TRACK_SPLIT="val", RUN_SPLIT_TRACKING=True,
        TRACK_OUTPUT_DIR=folder / "tracking" / "search" / "val",
        EVALUATION_OUTPUT_DIR=folder / "evaluation" / "search" / "val",
        TEST_OUTPUT_DIR=folder / "tracking" / "search" / "test",
        TEST_SEQUENCES=[], SAVE_VIDEO=False, SAVE_TEST_VIDEO=False,
        SHOW_PROGRESS=SHOW_PROGRESS, SAVE_PLOTS=SAVE_SEARCH_PLOTS,
    )
    if MODE == "train":
        settings.update(EPOCHS=trial["epochs"], SAVE_EVERY_EPOCH=True, RESUME_TRAINING=False)
    return settings


def write_json(path, value):
    """Replace JSON atomically so interruption cannot leave a half-written result."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def search_signature(manifest):
    """Compare effective experiment settings, ignoring overridden config defaults."""
    ignored = set(trial_settings({"parameters": {}, "epochs": 1}, Path(".")))
    for trial in manifest["trials"]:
        ignored.update(trial["parameters"])
        if "LOOKBACK_FRAMES" in trial["parameters"]:
            ignored.add("MAX_MISSED_FRAMES")
    signature = {key: manifest[key] for key in (
        "mode", "trials", "rank_metric", "rank_benchmark", "data_split")}
    signature["checkpoint"] = (manifest["source_checkpoint"] if manifest["mode"] == "tracking"
                               else manifest["evaluate_checkpoint"])
    # Old manifests predate architecture tracking; never resume them as a new model.
    signature["model_version"] = manifest.get("model_version")
    # Older searches omitted this field and trained on all frames.
    signature["train_max_frames_per_sequence"] = (manifest.get("train_max_frames_per_sequence")
                                                 if manifest["mode"] == "train" else None)
    signature["settings"] = {key: value for key, value in manifest["base_settings"].items()
                             if key not in ignored}
    return json.dumps(signature, sort_keys=True, default=str)


def find_search_folder(manifest):
    """Find a compatible old search, including searches created before resume support."""
    if RESUME_DIR is not None and not RESUME:
        raise ValueError("Set RESUME=True to use RESUME_DIR")
    root = Path(SEARCH_ROOT).resolve()
    explicit = Path(RESUME_DIR).resolve() if RESUME_DIR is not None else None
    candidates = ([explicit] if explicit else sorted(
        (p.parent for p in root.glob("*/search.json") if p.parent.name.startswith(SEARCH_NAME + "_")),
        reverse=True)) if RESUME else []
    for folder in candidates:
        try:
            saved = json.loads((folder / "search.json").read_text())
            compatible = search_signature(saved) == search_signature(manifest)
            if compatible and MODE == "tracking":
                old_digest = saved.get("source_checkpoint_sha256")
                compatible = (old_digest == manifest["source_checkpoint_sha256"] if old_digest else
                              Path(SOURCE_CHECKPOINT).stat().st_mtime_ns <= (folder / "search.json").stat().st_mtime_ns)
        except (OSError, ValueError, KeyError, TypeError):
            if explicit:
                raise ValueError(f"Cannot read a valid search manifest in {folder}")
            continue
        if compatible:
            return folder, True
        if explicit:
            raise ValueError("RESUME_DIR has different search/data/model settings or checkpoint contents. "
                             "Restore the original settings or start a fresh search.")
    return root / f"{SEARCH_NAME}_{datetime.now():%Y%m%d_%H%M%S_%f}", False


@contextmanager
def search_lock(folder):
    """Prevent two resume-aware processes from using the same search simultaneously."""
    with (folder / ".search.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another search is already running in {folder}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def completed_result(folder, index, trial):
    """Reuse only a matching, finite result with its checkpoint and report present."""
    path = folder / "result.json"
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text())
        row.setdefault("train_max_frames_per_sequence", None)  # Legacy full-video results.
        expected = {"trial": index, "mode": MODE, "epochs": trial["epochs"],
                    "train_max_frames_per_sequence": TRAIN_MAX_FRAMES_PER_SEQUENCE if MODE == "train" else None,
                    "benchmark": RANK_BENCHMARK, "folder": str(folder), **trial["parameters"]}
        # JSON canonicalization also makes tuple-valued parameter candidates comparable.
        if json.dumps({k: row[k] for k in expected}, sort_keys=True) != json.dumps(expected, sort_keys=True):
            return None
        if not all(isinstance(row[k], (int, float)) and math.isfinite(row[k])
                   for k in ("HOTA", "IDF1", "MOTA", "IDSW", "seconds")):
            return None
        if not isinstance(row["checkpoint_epoch"], int) or row["checkpoint_epoch"] < 1:
            return None
        if not Path(row["checkpoint"]).is_file() or not Path(row["report"]).is_file():
            return None
        return row
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_results(folder, rows):
    """Refresh a ranked CSV and best.json after each completed evaluation."""
    ranked = sorted(rows, key=lambda row: (-row[RANK_METRIC], -row["IDF1"], row["trial"]))
    fields = ["rank", "trial", "mode", "epochs", "train_max_frames_per_sequence", "checkpoint_epoch", "benchmark",
              "HOTA", "IDF1", "MOTA", "IDSW", "seconds", *PARAM_GRID,
              "checkpoint", "report", "folder"]
    temporary = folder / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow({"rank": rank, **row})
    temporary.replace(folder / "results.csv")
    best = ranked[0]
    write_json(folder / "best.json", {
        "rank_metric": RANK_METRIC, "benchmark": RANK_BENCHMARK,
        "parameters": {name: best[name] for name in PARAM_GRID},
        "result": best,
        "note": "Best completed validation trial; settings have NOT been applied to config.py.",
    })
    return best


def main():
    """Run isolated sequential trials, preserving all completed results on failure."""
    from Utils.splits import validate_video_splits, split_manifest
    from Utils.track_state import tracker_settings

    trials = build_trials()
    validate_video_splits()
    if not any(name.startswith(RANK_BENCHMARK + "-") for name in config.VAL_SEQUENCES):
        raise ValueError(f"No {RANK_BENCHMARK} validation sequences to rank")
    if MODE == "tracking" and not Path(SOURCE_CHECKPOINT).is_file():
        raise FileNotFoundError(f"Missing SOURCE_CHECKPOINT: {SOURCE_CHECKPOINT}")
    # Validate all tracker combinations before starting any training.
    for trial in trials:
        with temporary_settings(trial_settings(trial, Path(SEARCH_ROOT) / "preflight")):
            tracker_settings()
            if not isinstance(config.MATCH_THRESHOLD, (int, float)) or not 0 <= config.MATCH_THRESHOLD <= 1:
                raise ValueError("MATCH_THRESHOLD must be between 0 and 1")

    manifest = {
        "model_version": MODEL_VERSION,
        "mode": MODE, "trials": trials, "rank_metric": RANK_METRIC,
        "rank_benchmark": RANK_BENCHMARK, "evaluate_checkpoint": EVALUATE_CHECKPOINT,
        "source_checkpoint": str(Path(SOURCE_CHECKPOINT).resolve()) if MODE == "tracking" else None,
        "base_settings": {k: v for k, v in vars(config).items() if k.isupper()},
        "data_split": split_manifest(),
        "train_max_frames_per_sequence": TRAIN_MAX_FRAMES_PER_SEQUENCE if MODE == "train" else None,
    }
    if MODE == "tracking":
        digest = hashlib.sha256()
        with Path(SOURCE_CHECKPOINT).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["source_checkpoint_sha256"] = digest.hexdigest()
    folder, resumed = find_search_folder(manifest)
    folder.mkdir(parents=True, exist_ok=resumed)
    with search_lock(folder):
        if not resumed:
            write_json(folder / "search.json", manifest)
        return run_trials(folder, trials, resumed)


def run_trials(folder, trials, resumed):
    """Skip completed trials; archive and restart the first unfinished trial."""
    import train
    import tracking
    import evaluate

    tqdm.write(f"Search: {len(trials)} trials; mode={MODE}; ranking by {RANK_BENCHMARK} {RANK_METRIC}\n{folder}")
    rows = []
    for index, trial in enumerate(trials, 1):
        row = completed_result(folder / f"trial_{index:03d}", index, trial)
        if row is not None:
            rows.append(row)
    done = {row["trial"] for row in rows}
    tqdm.write(f"{'Resuming' if resumed else 'Starting'}: {len(done)} completed; {len(trials)-len(done)} remaining.")
    if rows:
        best = save_results(folder, rows)  # Also repairs a ranking interrupted after result.json was saved.
    if MODE == "train":
        tqdm.write(f"Remaining fresh training epochs: {sum(t['epochs'] for i, t in enumerate(trials, 1) if i not in done)}. "
                   "Each trial is followed by full validation tracking/evaluation.")
        tqdm.write(f"Training frames per sequence: {TRAIN_MAX_FRAMES_PER_SEQUENCE or 'all'}; "
                   "shorter videos use all their frames. Validation remains full-length.")
    # Existing phase/frame bars occupy rows 0/1. Search ETA updates after a trial.
    with tqdm(total=len(trials), initial=len(done), desc="Search overall", unit="trial", position=2,
              smoothing=0, dynamic_ncols=True, disable=not SHOW_PROGRESS) as bar:
        for index, trial in enumerate(trials, 1):
            if index in done:
                continue
            trial_dir = folder / f"trial_{index:03d}"
            if trial_dir.exists():
                archive = folder / "interrupted" / f"{trial_dir.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
                archive.parent.mkdir(exist_ok=True)
                trial_dir.rename(archive)
                tqdm.write(f"Preserved unfinished trial in {archive}; restarting this trial from the beginning.")
            trial_dir.mkdir()
            settings = trial_settings(trial, trial_dir)
            write_json(trial_dir / "settings.json", settings)
            start = time.perf_counter()
            tqdm.write(f"\nTrial {index}/{len(trials)}: {trial}")
            try:
                with temporary_settings(settings):
                    if MODE == "train":
                        bar.set_postfix(trial=index, phase="train")
                        train.main(max_train_frames_per_sequence=TRAIN_MAX_FRAMES_PER_SEQUENCE)
                        if EVALUATE_CHECKPOINT == "last":
                            config.CHECKPOINT_PATH = trial_dir / "last.pt"
                    checkpoint_path = Path(config.CHECKPOINT_PATH).resolve()
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    checkpoint_epoch = checkpoint["epoch"]
                    del checkpoint
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    bar.set_postfix(trial=index, phase="track")
                    tracking.main()
                    bar.set_postfix(trial=index, phase="evaluate")
                    metrics = evaluate.main()
                    values = {name: float(metrics[RANK_BENCHMARK][name])
                              for name in ("HOTA", "IDF1", "MOTA", "IDSW")}
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError("Non-finite evaluation metrics; trial cannot be ranked")
                    row = {
                        "trial": index, "mode": MODE, "epochs": trial["epochs"],
                        "train_max_frames_per_sequence": TRAIN_MAX_FRAMES_PER_SEQUENCE if MODE == "train" else None,
                        "checkpoint_epoch": checkpoint_epoch, "benchmark": RANK_BENCHMARK,
                        **values, "seconds": round(time.perf_counter() - start, 2),
                        **trial["parameters"], "checkpoint": str(checkpoint_path),
                        "report": str(config.EVALUATION_OUTPUT_DIR / "results.md"), "folder": str(trial_dir),
                    }
                    write_json(trial_dir / "result.json", row)
                rows.append(row)
                best = save_results(folder, rows)
                bar.update(1)
                bar.set_postfix(best_trial=best["trial"], **{RANK_METRIC: best[RANK_METRIC]})
            except (Exception, KeyboardInterrupt) as error:
                (trial_dir / "error.txt").write_text(f"{type(error).__name__}: {error}\n")
                tqdm.write(f"Search stopped at trial {index}. Earlier results are preserved in {folder}.")
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    tqdm.write(f"\nBest trial: {best['trial']}; {RANK_METRIC}={best[RANK_METRIC]:.4f}; "
               f"IDF1={best['IDF1']:.4f}\nRanking: {folder / 'results.csv'}\n"
               f"Settings/checkpoint: {folder / 'best.json'}\nconfig.py was not changed.")
    return sorted(rows, key=lambda row: row["trial"])


if __name__ == "__main__":
    main()
