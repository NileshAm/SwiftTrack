"""Train this project's association model with explicit run and epoch settings.

Run from the project root: python Baselines/OurImplementation/train.py --output-dir ...
Each requested detector trains a separate association checkpoint in its own folder.
The detector models themselves stay frozen; this trains the association model.
"""

import argparse
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from Utils import config  # noqa: E402
from Utils.benchmark_boxes import gt_person_cache  # noqa: E402
from Utils.detections import validate_tracktor  # noqa: E402
from Utils.splits import validate_video_splits  # noqa: E402
from Utils.yolox_osnet import cache_for_sequence as validate_yolox  # noqa: E402
import train as association_training  # noqa: E402


ENCODERS = {"detr": ("detr_resnet50", 256),
            "osnet": ("osnet_reid", 512),
            "densenet": ("densenet121", 1024)}


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
        else:
            file, summary = validate_yolox(args.yolox_cache_dir, path)
            floor = summary["identity"]["detector"]["min_score"]
            if args.detection_min_score is not None and args.detection_min_score < floor:
                raise ValueError(f"{sequence}: requested score is below YOLOX cache floor {floor}")
        config.DETECTION_FILES[sequence] = str(file)
    config.DETECTION_BACKEND = "mapped"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="One detector: exact run folder; multiple: parent folder with one run per detector")
    parser.add_argument("--detectors", nargs="+", choices=("config", "gt", "tracktor", "yolox"),
                        default=("config",), help="Box sources; each gets a separate association model")
    parser.add_argument("--encoder", choices=("config", *ENCODERS), default="config")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Total target epoch, including epochs already completed")
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--val-batch-size", type=int, default=config.VAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--lookback", type=int, default=config.LOOKBACK_FRAMES)
    parser.add_argument("--detection-min-score", type=float, default=config.DETECTION_MIN_SCORE)
    parser.add_argument("--gt-match-iou", type=float, default=config.GT_MATCH_IOU)
    parser.add_argument("--association-distance-px", type=float,
                        default=config.ASSOCIATION_DISTANCE_PX)
    parser.add_argument("--loss-alpha", type=float, default=config.LOSS_ALPHA)
    parser.add_argument("--negative-ratio", type=int, default=config.NEGATIVE_RATIO)
    parser.add_argument("--match-threshold", type=float, default=config.MATCH_THRESHOLD,
                        help="Saved for inference; not used to optimize training loss")
    parser.add_argument("--tracktor-cache-dir", type=Path, default=config.TRACKTOR_CACHE_DIR)
    parser.add_argument("--yolox-cache-dir", type=Path,
                        default=PROJECT / "detections/yolox_x_mot20_val_v1")
    parser.add_argument("--train-sequences", nargs="+", default=config.TRAIN_SEQUENCES)
    parser.add_argument("--val-sequences", nargs="+", default=config.VAL_SEQUENCES)
    parser.add_argument("--resume", action="store_true", help="Continue this run's last.pt checkpoint")
    args = parser.parse_args(argv)
    if len(set(args.detectors)) != len(args.detectors) or ("config" in args.detectors and len(args.detectors) > 1):
        parser.error("List each detector once; 'config' cannot be combined with another detector")
    if (args.epochs < 1 or args.batch_size < 1 or args.val_batch_size < 1 or args.lookback < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("Epochs, batch sizes, lookback and learning rate must be positive")
    if args.detection_min_score is not None and not 0 <= args.detection_min_score <= 1:
        parser.error("--detection-min-score must be in [0,1]")
    if not 0 < args.gt_match_iou <= 1:
        parser.error("--gt-match-iou must be in (0,1]")
    if (not math.isfinite(args.association_distance_px) or args.association_distance_px <= 0
            or not 0 <= args.loss_alpha <= 1 or args.negative_ratio < 1
            or not 0 <= args.match_threshold <= 1):
        parser.error("Invalid association distance, loss alpha, negative ratio or match threshold")
    if args.encoder != "config":
        config.ENCODER_NAME, config.FEATURE_DIM = ENCODERS[args.encoder]
        config.FREEZE_ENCODER = True
        config.USE_PRETRAINED_WEIGHTS = True
    config.TRAIN_SEQUENCES = args.train_sequences
    config.VAL_SEQUENCES = args.val_sequences
    config.EPOCHS = args.epochs
    config.TRAIN_BATCH_SIZE = args.batch_size
    config.VAL_BATCH_SIZE = args.val_batch_size
    config.LEARNING_RATE = args.learning_rate
    config.LOOKBACK_FRAMES = args.lookback
    config.MAX_MISSED_FRAMES = args.lookback - 1
    config.DETECTION_MIN_SCORE = args.detection_min_score
    config.GT_MATCH_IOU = args.gt_match_iou
    config.ASSOCIATION_DISTANCE_PX = args.association_distance_px
    config.LOSS_ALPHA = args.loss_alpha
    config.NEGATIVE_RATIO = args.negative_ratio
    config.MATCH_THRESHOLD = args.match_threshold
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
