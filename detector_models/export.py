"""Export one official pretrained detector to verified MOT caches. No training."""

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from Utils import config
from Utils.data import _sequence_info
from .adapters import AvailabilityError, load_detector
from .cache import cache_path, export_sequence
from .registry import ORDER, get


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=ORDER, required=True)
    parser.add_argument("--output-root", type=Path, default=config.PROJECT_ROOT / "outputs/crowded_detector_suite_v1")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--cache-floor", type=float, default=.05)
    parser.add_argument("--nms-iou", type=float, default=.7)
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args(argv)
    spec = get(args.detector)
    args.image_size = args.image_size or spec.input_size
    if not 0 <= args.cache_floor <= 1 or args.image_size < 32:
        parser.error("invalid cache floor or image size")
    arm = args.output_root / args.detector
    arm.mkdir(parents=True, exist_ok=True)
    try:
        detector = load_detector(args.detector, args)
    except AvailabilityError as error:
        limitation = {"detector": args.detector, "status": "unavailable", "reason": str(error),
                      "official_repo": spec.official_repo, "checkpoint": spec.checkpoint,
                      "substitution_used": False,
                      "retryable": spec.runtime not in ("legacy_crowddet", "legacy_detectron2")}
        (arm / "limitation.json").write_text(json.dumps(limitation, indent=2) + "\n")
        print(f"UNAVAILABLE {spec.title}: {error}")
        return 20
    sequences = [config.DATASET_ROOT / config.DATASET_SPLIT / name for name in config.VAL_SEQUENCES]
    cache = arm / "detections"
    total = sum(_sequence_info(path).getint("seqLength") for path in sequences)
    with tqdm(total=total, desc=f"{spec.title} detection", unit="frame") as overall:
        for sequence in sequences:
            export_sequence(sequence, cache_path(cache, sequence), detector, overall)
    (arm / "detector_manifest.json").write_text(json.dumps({
        "status": "complete", "detector": args.detector, "recipe": detector.recipe,
        "sequences": list(config.VAL_SEQUENCES), "trained_or_finetuned_here": False}, indent=2) + "\n")
    (arm / "limitation.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
