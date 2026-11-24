"""Train this project's association model with explicit run and epoch settings.

Run from the project root: python Baselines/OurImplementation/train.py --output-dir ...
Each requested detector trains a separate association checkpoint in its own folder.
The detector models themselves stay frozen; this trains the association model.
"""

import argparse
import importlib.util
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from Utils import config  # noqa: E402
from Utils.benchmark_boxes import gt_person_cache  # noqa: E402
from Utils.detections import file_digest, validate_tracktor, validate_cache  # noqa: E402
from Utils.splits import validate_video_splits  # noqa: E402
from Utils.yolox_osnet import (cache_for_sequence as validate_yolox,
                              cache_for_checkpoint)  # noqa: E402
from Utils.itere2edet import cache_for_sequence as validate_iter  # noqa: E402
import train as association_training  # noqa: E402


ENCODERS = {"detr": ("detr_resnet50", 256),
            "osnet": ("osnet_reid", 512),
            "densenet": ("densenet121", 1024)}


def call_script(filename, arguments):
    path = PROJECT / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(arguments)


def prepare_detections(detector, sequences, output, args):
    """Resolve complete per-video box files before an expensive training run."""
    if detector == "config":
        return
    config.DETECTION_FILES = {}
    config.MAPPED_DETECTION_SOURCE = detector
    for sequence in sequences:
        path = config.DATASET_ROOT / config.DATASET_SPLIT / sequence
        if detector == "gt":
            file = gt_person_cache(path, output / "detection_inputs/gt" / f"{sequence}.txt")
        elif detector == "tracktor":
            config.TRACKTOR_CACHE_DIR = args.tracktor_cache_dir.resolve()
            file = validate_tracktor(path)
        elif detector == "yolox":
            if args.yolox_checkpoint is None:
                file, summary = validate_yolox(args.yolox_cache_dir, path)
            else:
                file, summary = cache_for_checkpoint(
                    args.yolox_cache_dir, path, file_digest(args.yolox_checkpoint))
            floor = summary["identity"]["detector"]["min_score"]
            if args.detection_min_score is not None and args.detection_min_score < floor:
                raise ValueError(f"{sequence}: requested score is below YOLOX cache floor {floor}")
        elif detector == "iter":
            file, summary = validate_iter(args.iter_cache_dir, path, args.iter_checkpoint)
            recipe = summary["identity"]["detector"]
            if (recipe.get("short_side"), recipe.get("max_size")) != (args.iter_short_side, args.iter_max_size):
                raise ValueError(f"{sequence}: Iter cache resolution differs from --iter-short-side/--iter-max-size")
            floor = recipe["min_score"]
            if args.detection_min_score is not None and args.detection_min_score < floor:
                raise ValueError(f"{sequence}: requested score is below Iter cache floor {floor}")
        elif detector == "rtdetr":
            file = validate_cache(path)
            if args.detection_min_score is not None and args.detection_min_score < config.DETECTOR_MIN_SCORE:
                raise ValueError(f"{sequence}: requested score is below RT-DETR cache floor")
        config.DETECTION_FILES[sequence] = str(file)
    config.DETECTION_BACKEND = "mapped"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="One detector: exact run folder; multiple: parent folder with one run per detector")
    parser.add_argument("--detectors", nargs="+", choices=("config", "gt", "tracktor", "yolox", "iter", "rtdetr"),
                        default=("config",), help="Box sources; each gets a separate association model")
    parser.add_argument("--encoder", choices=("config", *ENCODERS), default="config")
    parser.add_argument("--graph", choices=("on", "off"), default="on",
                        help="Train graph attention or train the matcher directly on appearance")
    parser.add_argument("--finetuned-osnet-weights", type=Path,
                        help="Train association on this newly fine-tuned OSNet checkpoint")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Total target epoch, including epochs already completed")
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--val-batch-size", type=int, default=config.VAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--lookback", type=int, default=config.LOOKBACK_FRAMES)
    parser.add_argument("--detection-min-score", type=float, default=config.DETECTION_MIN_SCORE)
    parser.add_argument("--new-track-min-score", type=float, default=config.NEW_TRACK_MIN_SCORE,
                        help="Track birth threshold used by chronological training")
    parser.add_argument("--gt-match-iou", type=float, default=config.GT_MATCH_IOU)
    parser.add_argument("--association-distance-px", type=float,
                        default=config.ASSOCIATION_DISTANCE_PX)
    parser.add_argument("--loss-alpha", type=float, default=config.LOSS_ALPHA)
    parser.add_argument("--negative-ratio", type=int, default=config.NEGATIVE_RATIO)
    parser.add_argument("--match-threshold", type=float, default=config.MATCH_THRESHOLD,
                        help="Saved for inference; not used to optimize training loss")
    parser.add_argument("--tracktor-cache-dir", type=Path, default=config.TRACKTOR_CACHE_DIR)
    parser.add_argument("--rtdetr-cache-dir", type=Path)
    parser.add_argument("--rtdetr-weights-dir", type=Path)
    parser.add_argument("--rtdetr-image-size", type=int, default=960)
    parser.add_argument("--rtdetr-cache-min-score", type=float, default=0.05)
    parser.add_argument("--yolox-cache-dir", type=Path,
                        help="YOLOX cache to use or create")
    parser.add_argument("--yolox-checkpoint", type=Path,
                        help="Existing fine-tuned YOLOX best.pth for cache creation or reuse")
    parser.add_argument("--train-yolox", action="store_true",
                        help="Fine-tune YOLOX, export its best weights and train association")
    parser.add_argument("--make-yolox-cache", action="store_true",
                        help="Export YOLOX boxes for all train and validation videos")
    parser.add_argument("--yolox-train-output-dir", type=Path)
    parser.add_argument("--yolox-train-epochs", type=int, default=5)
    parser.add_argument("--yolox-train-batch-size", type=int, default=1)
    parser.add_argument("--yolox-train-accumulation", type=int, default=4)
    parser.add_argument("--yolox-train-lr", type=float, default=5e-5)
    parser.add_argument("--yolox-train-backbone-lr", type=float, default=5e-6)
    parser.add_argument("--yolox-train-weight-decay", type=float, default=1e-5)
    parser.add_argument("--yolox-train-score", type=float, default=0.25)
    parser.add_argument("--yolox-train-min-precision", type=float, default=0.7)
    parser.add_argument("--yolox-train-min-recall", type=float, default=0.55)
    parser.add_argument("--yolox-train-fraction", type=float, default=1.0)
    parser.add_argument("--yolox-train-frame-stride", type=int, default=1)
    parser.add_argument("--yolox-width", type=int, default=960)
    parser.add_argument("--yolox-height", type=int, default=544)
    parser.add_argument("--yolox-cache-min-score", type=float, default=0.1)
    parser.add_argument("--yolox-nms-iou", type=float, default=0.7)
    parser.add_argument("--iter-cache-dir", type=Path, help="Complete Iter cache; with --make-iter-cache, export here")
    parser.add_argument("--iter-checkpoint", type=Path,
                        help="Official CrowdHuman model.pth or a fine-tuned Iter best.pth")
    parser.add_argument("--make-iter-cache", action="store_true",
                        help="Export Iter detections for all train and validation videos")
    parser.add_argument("--train-iter", action="store_true",
                        help="Fine-tune Iter, export its best weights, then train association")
    parser.add_argument("--iter-train-output-dir", type=Path)
    parser.add_argument("--iter-train-epochs", type=int, default=5)
    parser.add_argument("--iter-train-batch-size", type=int, default=1)
    parser.add_argument("--iter-train-accumulation", type=int, default=4)
    parser.add_argument("--iter-train-frame-stride", type=int, default=1)
    parser.add_argument("--iter-train-lr", type=float, default=1e-5)
    parser.add_argument("--iter-train-backbone-lr", type=float, default=1e-6)
    parser.add_argument("--iter-train-weight-decay", type=float, default=1e-4)
    parser.add_argument("--iter-short-side", type=int, default=960)
    parser.add_argument("--iter-max-size", type=int, default=1600)
    parser.add_argument("--iter-cache-min-score", type=float, default=0.05)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--train-sequences", nargs="+", default=config.TRAIN_SEQUENCES)
    parser.add_argument("--val-sequences", nargs="+", default=config.VAL_SEQUENCES)
    parser.add_argument("--resume", action="store_true", help="Continue this run's last.pt checkpoint")
    args = parser.parse_args(argv)
    if len(set(args.detectors)) != len(args.detectors) or ("config" in args.detectors and len(args.detectors) > 1):
        parser.error("List each detector once; 'config' cannot be combined with another detector")
    if (args.train_yolox or args.make_yolox_cache or args.yolox_checkpoint) and "yolox" not in args.detectors:
        parser.error("YOLOX options require --detectors yolox")
    if args.train_yolox and args.yolox_checkpoint is not None:
        parser.error("Choose --train-yolox or --yolox-checkpoint, not both")
    if (args.train_iter or args.make_iter_cache or args.iter_checkpoint or args.iter_cache_dir) and "iter" not in args.detectors:
        parser.error("Iter options require --detectors iter")
    if args.train_iter and args.iter_checkpoint is not None:
        parser.error("Choose --train-iter or --iter-checkpoint, not both")
    if (args.rtdetr_cache_dir or args.rtdetr_weights_dir) and "rtdetr" not in args.detectors:
        parser.error("RT-DETR options require --detectors rtdetr")
    if "rtdetr" in args.detectors and args.rtdetr_cache_dir is None:
        parser.error("--detectors rtdetr requires --rtdetr-cache-dir")
    if args.finetuned_osnet_weights is not None and args.encoder != "osnet":
        parser.error("--finetuned-osnet-weights requires --encoder osnet")
    if (args.epochs < 1 or args.batch_size < 1 or args.val_batch_size < 1 or args.lookback < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("Epochs, batch sizes, lookback and learning rate must be positive")
    if args.detection_min_score is not None and not 0 <= args.detection_min_score <= 1:
        parser.error("--detection-min-score must be in [0,1]")
    if not 0 <= args.new_track_min_score <= 1:
        parser.error("--new-track-min-score must be in [0,1]")
    if not 0 < args.gt_match_iou <= 1:
        parser.error("--gt-match-iou must be in (0,1]")
    if (not math.isfinite(args.association_distance_px) or args.association_distance_px <= 0
            or not 0 <= args.loss_alpha <= 1 or args.negative_ratio < 1
            or not 0 <= args.match_threshold <= 1):
        parser.error("Invalid association distance, loss alpha, negative ratio or match threshold")
    if (min(args.yolox_train_epochs, args.yolox_train_batch_size, args.yolox_train_accumulation,
            args.yolox_train_frame_stride, args.yolox_width, args.yolox_height) < 1
            or not 0 < args.yolox_train_fraction <= 1
            or not 0 < args.yolox_train_lr < float("inf")
            or not 0 < args.yolox_train_backbone_lr < float("inf")
            or not 0 <= args.yolox_train_weight_decay < float("inf")
            or not 0 < args.yolox_train_score < 1
            or not 0 <= args.yolox_train_min_precision <= 1
            or not 0 <= args.yolox_train_min_recall <= 1
            or not 0 <= args.yolox_cache_min_score <= 1
            or not 0 < args.yolox_nms_iou <= 1):
        parser.error("Invalid YOLOX fine-tuning or cache settings")
    if (min(args.iter_train_epochs, args.iter_train_batch_size, args.iter_train_accumulation,
            args.iter_train_frame_stride, args.iter_short_side) < 1
            or args.iter_max_size < args.iter_short_side
            or not 0 < args.iter_train_lr < float("inf")
            or not 0 < args.iter_train_backbone_lr < float("inf")
            or not 0 <= args.iter_train_weight_decay < float("inf")
            or not 0 <= args.iter_cache_min_score <= 1):
        parser.error("Invalid Iter fine-tuning, resolution or cache settings")
    if args.encoder != "config":
        config.ENCODER_NAME, config.FEATURE_DIM = ENCODERS[args.encoder]
        config.FREEZE_ENCODER = True
        config.USE_PRETRAINED_WEIGHTS = True
    if args.finetuned_osnet_weights is not None:
        config.FINETUNED_OSNET_WEIGHTS_PATH = args.finetuned_osnet_weights.resolve()
        if not config.FINETUNED_OSNET_WEIGHTS_PATH.is_file():
            parser.error(f"Missing fine-tuned OSNet checkpoint: {config.FINETUNED_OSNET_WEIGHTS_PATH}")
    config.TRAIN_SEQUENCES = args.train_sequences
    config.VAL_SEQUENCES = args.val_sequences
    config.EPOCHS = args.epochs
    config.TRAIN_BATCH_SIZE = args.batch_size
    config.VAL_BATCH_SIZE = args.val_batch_size
    config.LEARNING_RATE = args.learning_rate
    config.LOOKBACK_FRAMES = args.lookback
    config.MAX_MISSED_FRAMES = args.lookback - 1
    config.DETECTION_MIN_SCORE = args.detection_min_score
    config.TRAIN_USE_GRAPH = args.graph == "on"
    config.NEW_TRACK_MIN_SCORE = args.new_track_min_score
    config.GT_MATCH_IOU = args.gt_match_iou
    config.ASSOCIATION_DISTANCE_PX = args.association_distance_px
    config.LOSS_ALPHA = args.loss_alpha
    config.NEGATIVE_RATIO = args.negative_ratio
    config.MATCH_THRESHOLD = args.match_threshold
    config.DEVICE = args.device
    config.RESUME_TRAINING = args.resume
    config.TEST_SEQUENCES = []
    config.RUN_SPLIT_TRACKING = False
    config.SAVE_VIDEO = False
    config.SAVE_TEST_VIDEO = False
    validate_video_splits()
    outputs = {detector: (args.output_dir / detector if len(args.detectors) > 1
                          else args.output_dir).resolve() for detector in args.detectors}
    if not args.resume:
        for output in outputs.values():
            if any((output / name).exists() for name in ("model.pt", "last.pt", "history.csv")):
                parser.error(f"Training files already exist in {output}; use --resume or a new output directory")
    if "yolox" in args.detectors:
        args.yolox_cache_dir = (args.yolox_cache_dir or
            (args.output_dir / "yolox_cache" if args.train_yolox or args.make_yolox_cache
             else PROJECT / "detections/yolox_x_mot20_val_v1")).resolve()
        if args.train_yolox:
            detector_dir = (args.yolox_train_output_dir or
                            args.output_dir / "yolox_detector").resolve()
            call_script("detector-train-yolox.py", [
                "--output-dir", str(detector_dir), "--epochs", str(args.yolox_train_epochs),
                "--width", str(args.yolox_width), "--height", str(args.yolox_height),
                "--batch-size", str(args.yolox_train_batch_size),
                "--accumulation", str(args.yolox_train_accumulation),
                "--lr", str(args.yolox_train_lr),
                "--backbone-lr", str(args.yolox_train_backbone_lr),
                "--weight-decay", str(args.yolox_train_weight_decay),
                "--score", str(args.yolox_train_score),
                "--min-precision", str(args.yolox_train_min_precision),
                "--min-recall", str(args.yolox_train_min_recall),
                "--train-fraction", str(args.yolox_train_fraction),
                "--frame-stride", str(args.yolox_train_frame_stride),
                "--device", args.device])
            args.yolox_checkpoint = detector_dir / "best.pth"
        if args.yolox_checkpoint is not None:
            args.yolox_checkpoint = args.yolox_checkpoint.resolve()
            if not args.yolox_checkpoint.is_file():
                parser.error(f"Missing YOLOX checkpoint: {args.yolox_checkpoint}")
        if args.make_yolox_cache or args.train_yolox:
            export_args = ["--split", "all", "--cache-dir", str(args.yolox_cache_dir),
                           "--width", str(args.yolox_width), "--height", str(args.yolox_height),
                           "--min-score", str(args.yolox_cache_min_score),
                           "--nms-iou", str(args.yolox_nms_iou), "--device", args.device]
            if args.yolox_checkpoint is not None:
                export_args += ["--checkpoint", str(args.yolox_checkpoint)]
            call_script("detect-yolox.py", export_args)
    if "iter" in args.detectors:
        args.iter_cache_dir = (args.iter_cache_dir or
            (args.output_dir / "iter_cache" if args.train_iter or args.make_iter_cache
             else PROJECT / "detections/iter_e2edet_resolution_v2/higher_960_1600")).resolve()
        if args.train_iter:
            detector_dir = (args.iter_train_output_dir or args.output_dir / "iter_detector").resolve()
            train_args = ["--output-dir", str(detector_dir), "--epochs", str(args.iter_train_epochs),
                          "--batch-size", str(args.iter_train_batch_size),
                          "--accumulation", str(args.iter_train_accumulation),
                          "--frame-stride", str(args.iter_train_frame_stride),
                          "--train-short-side", str(args.iter_short_side),
                          "--train-max-size", str(args.iter_max_size),
                          "--lr", str(args.iter_train_lr),
                          "--backbone-lr", str(args.iter_train_backbone_lr),
                          "--weight-decay", str(args.iter_train_weight_decay),
                          "--device", args.device]
            if args.resume:
                train_args.append("--resume")
            call_script("detector-train-itere2edet.py", train_args)
            args.iter_checkpoint = detector_dir / "best.pth"
        args.iter_checkpoint = (args.iter_checkpoint or PROJECT / "weights/iter_e2edet/model.pth").resolve()
        if not args.iter_checkpoint.is_file():
            parser.error(f"Missing Iter checkpoint: {args.iter_checkpoint}")
        if args.make_iter_cache or args.train_iter:
            call_script("detect-itere2edet.py", [
                "--checkpoint", str(args.iter_checkpoint), "--split", "all",
                "--cache-dir", str(args.iter_cache_dir),
                "--short-side", str(args.iter_short_side), "--max-size", str(args.iter_max_size),
                "--min-score", str(args.iter_cache_min_score), "--device", args.device])
    if "rtdetr" in args.detectors:
        if args.rtdetr_image_size < 32 or args.rtdetr_image_size % 32 or not 0 <= args.rtdetr_cache_min_score <= 1:
            parser.error("Invalid RT-DETR cache settings")
        config.DETECTION_CACHE_DIR = args.rtdetr_cache_dir.resolve()
        config.DETECTOR_WEIGHTS_DIR = args.rtdetr_weights_dir.resolve() if args.rtdetr_weights_dir else None
        config.DETECTOR_INPUT_MODE = "square"
        config.DETECTOR_IMAGE_SIZE = args.rtdetr_image_size
        config.DETECTOR_MIN_SCORE = args.rtdetr_cache_min_score
    results = {}
    for detector in args.detectors:
        output = outputs[detector]
        config.OUTPUT_DIR = output
        config.CHECKPOINT_PATH = output / "model.pt"
        prepare_detections(detector, config.TRAIN_SEQUENCES + config.VAL_SEQUENCES, output, args)
        print(f"Training association model: detector={detector}, encoder={config.ENCODER_NAME}, output={output}")
        association_training.main()
        print(f"Best checkpoint: {config.CHECKPOINT_PATH}")
        print(f"Latest checkpoint: {output / 'last.pt'}")
        results[detector] = output
    return results


if __name__ == "__main__":
    main()
