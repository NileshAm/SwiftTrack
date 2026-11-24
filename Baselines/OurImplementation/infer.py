"""Run this project's trained association tracker on GT, Tracktor and YOLOX boxes.

GT person IDs are removed before tracking. Each detector gets its own MOT tracks
and summary files; run evaluate.py afterward to score the saved predictions.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from Utils import config  # noqa: E402
from Utils.benchmark_boxes import gt_person_cache  # noqa: E402
from Utils.detections import file_digest, validate_tracktor  # noqa: E402
from Utils.features import load_encoder  # noqa: E402
from Utils.splits import validate_video_splits  # noqa: E402
from Utils.yolox_osnet import cache_for_sequence as validate_yolox  # noqa: E402
from model import AssociationModel, MODEL_VERSION  # noqa: E402
import tracking  # noqa: E402


SAVED_SETTINGS = (
    "ENCODER_NAME", "FEATURE_DIM", "FREEZE_ENCODER", "USE_PRETRAINED_WEIGHTS",
    "CROP_SIZE", "DETR_MODEL_NAME", "DETR_REVISION", "DETR_SHORT_SIDE",
    "DETR_MAX_SIZE", "GRAPH_HIDDEN_DIMS", "ASSOCIATION_DISTANCE_PX",
    "SINKHORN_ITERATIONS", "SINKHORN_SCORE_SCALE", "SINKHORN_UNMATCHED_LOGIT",
    "LOOKBACK_FRAMES", "MAX_MISSED_FRAMES", "USE_MOTION_PREDICTION",
    "USE_APPEARANCE_HISTORY", "APPEARANCE_HISTORY_SIZE",
    "APPEARANCE_HISTORY_MIN_SCORE", "TRACK_CONFIRM_MIN_HITS",
    "TRACK_CONFIRM_WINDOW", "TRACKING_VARIANT", "BOX_ORIGIN",
)


def restore_model_settings(checkpoint):
    settings = checkpoint["settings"]
    missing = set(SAVED_SETTINGS) - settings.keys()
    if missing:
        raise ValueError(f"Checkpoint lacks tracking settings: {', '.join(sorted(missing))}")
    for name in SAVED_SETTINGS:
        setattr(config, name, settings[name])
    if "REID_WEIGHTS_PATH" in settings:
        config.REID_WEIGHTS_PATH = Path(settings["REID_WEIGHTS_PATH"])
    split = checkpoint["data_split"]
    if split.get("protocol") != "whole_video_v1":
        raise ValueError("Checkpoint does not record a disjoint whole-video split")
    config.DATASET_ROOT = Path(split["dataset_root"])
    config.DATASET_SPLIT = split["dataset_split"]
    config.TRAIN_SEQUENCES = list(split["sequences"]["train"])
    config.VAL_SEQUENCES = list(split["sequences"]["val"])
    config.TEST_SEQUENCES = []
    config.TRACK_SPLIT = "val"
    config.RUN_SPLIT_TRACKING = True
    config.SAVE_VIDEO = False
    config.SAVE_TEST_VIDEO = False
    validate_video_splits()
    temporal = checkpoint.get("temporal_sampling", {})
    if (temporal.get("protocol") not in ("historical_pair_window_v1", "chronological_track_state_v1")
            or temporal.get("max_frame_gap") != config.MAX_MISSED_FRAMES + 1):
        raise ValueError("Checkpoint look-back differs from this tracker")


def load_models(checkpoint, device, osnet_weights):
    model = AssociationModel(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    encoder = load_encoder(device)
    if checkpoint.get("encoder_state_dict") is not None:
        encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    if encoder.feature_settings != checkpoint["feature_settings"]:
        raise ValueError("Pretrained encoder weights or preprocessing differ from the checkpoint")
    transfer = None
    if osnet_weights is not None:
        if config.ENCODER_NAME != "osnet_reid":
            raise ValueError("--finetuned-osnet-weights needs an OSNet association checkpoint")
        saved = torch.load(osnet_weights, map_location="cpu", weights_only=True)
        recipe = saved.get("recipe", {})
        if (recipe.get("version") != "mot_osnet_triplet_v1"
                or recipe.get("base_features", {}).get("weights_sha256")
                   != encoder.feature_settings.get("weights_sha256")):
            raise ValueError("Fine-tuned OSNet weights are incompatible with this base encoder")
        encoder.network.load_state_dict(saved["encoder_state_dict"], strict=True)
        transfer = {"file": str(osnet_weights.resolve()),
                    "sha256": file_digest(osnet_weights), "epoch": saved["epoch"],
                    "association_trained_on_frozen_features": True}
        encoder.feature_settings = {**encoder.feature_settings,
                                    "fine_tuned_checkpoint_sha256": transfer["sha256"],
                                    "fine_tuned_epoch": saved["epoch"]}
    encoder.eval()
    return model, encoder, transfer


def source_file(detector, sequence_path, args, output):
    if detector == "gt":
        return gt_person_cache(sequence_path, output / "detection_inputs/gt" / f"{sequence_path.name}.txt")
    if detector == "tracktor":
        return validate_tracktor(sequence_path)
    cache, _ = validate_yolox(args.yolox_cache_dir, sequence_path)
    return cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="model.pt or last.pt from our association trainer")
    parser.add_argument("--detectors", nargs="+", choices=("gt", "tracktor", "yolox"),
                        default=("gt", "tracktor", "yolox"))
    parser.add_argument("--tracktor-cache-dir", type=Path, default=config.TRACKTOR_CACHE_DIR)
    parser.add_argument("--yolox-cache-dir", type=Path,
                        default=PROJECT / "detections/yolox_x_mot20_val_v1")
    parser.add_argument("--finetuned-osnet-weights", type=Path,
                        help="Optional best.pt from osnet-train.py; transfers into a frozen-OSNet association model")
    parser.add_argument("--graph", choices=("on", "off"), default="on")
    parser.add_argument("--detection-min-score", type=float, default=0.5)
    parser.add_argument("--new-track-min-score", type=float, default=0.7)
    parser.add_argument("--match-threshold", type=float,
                        help="Association threshold; default: value saved in checkpoint")
    parser.add_argument("--confirm-min-hits", type=int,
                        help="Override checkpoint track-confirmation count for a comparison")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if len(set(args.detectors)) != len(args.detectors):
        parser.error("List each detector only once")
    if any(not 0 <= value <= 1 for value in
           (args.detection_min_score, args.new_track_min_score)) or (
            args.match_threshold is not None and not 0 <= args.match_threshold <= 1):
        parser.error("All score thresholds must be in [0,1]")
    if args.confirm_min_hits is not None and args.confirm_min_hits < 1:
        parser.error("--confirm-min-hits must be at least 1")
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        parser.error(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION:
        parser.error("Checkpoint model version differs from the current implementation")
    restore_model_settings(checkpoint)
    config.CHECKPOINT_PATH = checkpoint_path
    config.DEVICE = args.device
    config.MATCH_THRESHOLD = (args.match_threshold if args.match_threshold is not None
                              else checkpoint["settings"]["MATCH_THRESHOLD"])
    config.DETECTION_MIN_SCORE = args.detection_min_score
    config.NEW_TRACK_MIN_SCORE = args.new_track_min_score
    config.TRACKTOR_CACHE_DIR = args.tracktor_cache_dir.resolve()
    if args.confirm_min_hits is not None:
        config.TRACK_CONFIRM_MIN_HITS = args.confirm_min_hits
    osnet_weights = args.finetuned_osnet_weights.resolve() if args.finetuned_osnet_weights else None
    model, encoder, transfer = load_models(checkpoint, args.device, osnet_weights)
    output = (args.output_dir or PROJECT / "outputs/our_implementation" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True)
    sources = {}
    for detector in args.detectors:
        sources[detector] = {}
        for sequence in config.VAL_SEQUENCES:
            sequence_path = config.DATASET_ROOT / config.DATASET_SPLIT / sequence
            path = source_file(detector, sequence_path, args, output)
            if detector == "yolox":
                summary = json.loads(path.with_name("summary.json").read_text())
                cache_floor = summary["identity"]["detector"]["min_score"]
                if args.detection_min_score < cache_floor:
                    raise ValueError(f"{sequence}: requested score is below YOLOX cache floor {cache_floor}")
            sources[detector][sequence] = {"file": str(path), "sha256": file_digest(path)}
    with tqdm(total=len(args.detectors) * len(config.VAL_SEQUENCES),
              desc="Our tracker inference", unit="video") as overall:
        for detector in args.detectors:
            folder = output / detector
            config.OUTPUT_DIR = folder
            config.TRACK_OUTPUT_DIR = folder / "tracking/val"
            config.EVALUATION_OUTPUT_DIR = folder / "evaluation/val"
            for sequence in config.VAL_SEQUENCES:
                config.DETECTION_BACKEND = "supplied"
                config.DETECTION_FILE = sources[detector][sequence]["file"]
                tracking.run_sequence(sequence, model, encoder, split="val",
                                      use_graph=args.graph == "on")
                overall.update(1)
    (output / "manifest.json").write_text(json.dumps({
        "implementation": "local_association_model", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_digest(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
        "model_version": MODEL_VERSION, "dataset_root": str(config.DATASET_ROOT.resolve()),
        "dataset_split": config.DATASET_SPLIT,
        "train_sequences": config.TRAIN_SEQUENCES, "sequences": config.VAL_SEQUENCES,
        "detectors": args.detectors, "sources": sources,
        "detection_min_score": args.detection_min_score,
        "new_track_min_score": args.new_track_min_score,
        "match_threshold": config.MATCH_THRESHOLD,
        "confirm_min_hits": config.TRACK_CONFIRM_MIN_HITS,
        "graph": args.graph, "encoder": encoder.feature_settings,
        "finetuned_osnet_transfer": transfer, "gt_ids_passed_to_tracker": False,
        "training_detection_source": checkpoint["data_split"].get("detections"),
    }, indent=2, default=str) + "\n")
    print(f"Inference output: {output}")
    return output


if __name__ == "__main__":
    main()
