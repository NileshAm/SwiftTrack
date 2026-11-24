"""Run the authors' graph/model_testing tracker on GT, Tracktor or YOLOX boxes.

Run from the project root: python Baselines/DragonTrackGithub/infer.py --help
No detection identity is passed to the model. Outputs use MOTChallenge rows.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.nn import DataParallel
from tqdm.auto import tqdm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from Utils import config  # noqa: E402
from Utils.data import _sequence_info, load_ground_truth  # noqa: E402
from Utils.detections import file_digest, validate_tracktor  # noqa: E402
from Utils.yolox_osnet import cache_for_sequence as validate_yolox  # noqa: E402
from model_testing import model_testing  # noqa: E402
from network.complete_net import completeNet  # noqa: E402

UPSTREAM_COMMIT = "111e263c650f2f3a6ef39a261582eef87e09e8d8"


def gt_boxes_without_ids(sequence_path, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    truth = load_ground_truth(sequence_path)
    with target.open("x") as stream:
        for frame, record in truth.items():
            for x1, y1, x2, y2 in record["boxes"][record["valid_mask"]]:
                stream.write(f"{frame},-1,{x1+1:.3f},{y1+1:.3f},{x2-x1:.3f},{y2-y1:.3f},1,-1,-1,-1\n")
    return target


def cache_for(detector, sequence_path, args, output):
    if detector == "gt":
        path = gt_boxes_without_ids(sequence_path, output / "detection_inputs/gt" / f"{sequence_path.name}.txt")
    elif detector == "tracktor":
        config.TRACKTOR_CACHE_DIR = args.tracktor_cache_dir
        path = validate_tracktor(sequence_path)
    else:
        path, _ = validate_yolox(args.yolox_cache_dir, sequence_path)
    return path.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="models_v3/epoch_N.pth from the authors' trainer")
    parser.add_argument("--detectors", nargs="+", choices=("gt", "tracktor", "yolox"),
                        default=("gt", "tracktor", "yolox"))
    parser.add_argument("--sequences", nargs="+", default=config.VAL_SEQUENCES)
    parser.add_argument("--tracktor-cache-dir", type=Path, default=config.TRACKTOR_CACHE_DIR)
    parser.add_argument("--yolox-cache-dir", type=Path, default=PROJECT / "detections/yolox_x_mot20_val_v1")
    parser.add_argument("--detection-min-score", type=float, default=0.5)
    parser.add_argument("--match-threshold", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.detection_min_score <= 1 or not 0 <= args.match_threshold <= 1:
        parser.error("Score thresholds must be in [0, 1]")
    if len(set(args.sequences)) != len(args.sequences) or len(set(args.detectors)) != len(args.detectors):
        parser.error("Sequences and detector sources must be distinct")
    if not torch.cuda.is_available():
        parser.error("The authors' released model requires CUDA")
    checkpoint_path = args.checkpoint.resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in saved:
        parser.error("Checkpoint must be an authors-format epoch_N.pth")
    model = DataParallel(completeNet()).to("cuda")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    output = args.output_dir or PROJECT / "outputs/github_dragontrack" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True)
    source_info = {}
    root = config.DATASET_ROOT / config.DATASET_SPLIT
    for detector in tqdm(args.detectors, desc="Authors' tracker sources", unit="source"):
        source_info[detector] = {}
        for sequence in tqdm(args.sequences, desc=detector, unit="video", leave=False):
            sequence_path = root / sequence
            info = _sequence_info(sequence_path)
            cache = cache_for(detector, sequence_path, args, output)
            detections = np.loadtxt(cache, delimiter=",", ndmin=2)
            if detections.shape[1] < 7:
                raise ValueError(f"Expected MOT detections: {cache}")
            detections = detections[detections[:, 6] >= args.detection_min_score].copy()
            detections[:, 1] = -1  # Never pass upstream IDs into inference.
            detections = detections[np.argsort(detections[:, 0], kind="stable")]
            if not len(detections):
                raise ValueError(f"No boxes pass score {args.detection_min_score}: {cache}")
            source_info[detector][sequence] = {"file": str(cache), "sha256": file_digest(cache),
                                               "boxes_after_score_filter": len(detections)}
            tracks = model_testing(sequence, detections, str(sequence_path / "img1"),
                info.getint("seqLength"), 30, model, 200, 3, args.match_threshold,
                0.0, 15, 10, 10, info.getint("frameRate"))
            target = output / "tracks" / detector / f"{sequence}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as stream:
                for frame, track_id, x, y, width, height, *_ in sorted(tracks, key=lambda row: row[0]):
                    stream.write(f"{int(frame)},{int(track_id)},{x:.3f},{y:.3f},{width:.3f},{height:.3f},-1,-1,-1,-1\n")
            print(f"{detector} {sequence}: {len(tracks)} MOT rows -> {target}")
    (output / "manifest.json").write_text(json.dumps({
        "upstream_commit": UPSTREAM_COMMIT, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_digest(checkpoint_path), "checkpoint_epoch": saved.get("epoch"),
        "dataset_root": str(root.resolve()), "sequences": args.sequences,
        "detectors": args.detectors, "detection_min_score": args.detection_min_score,
        "match_threshold": args.match_threshold, "sources": source_info,
        "gt_ids_passed_to_tracker": False,
        "original_tracking_parameters": {"lookback": 30, "distance_limit": 200,
            "fp_min_times_seen": 3, "fp_lookback": 15, "fp_recent_frame_limit": 10,
            "min_height": 10},
    }, indent=2) + "\n")
    print(f"Inference run: {output}")
    return output


if __name__ == "__main__":
    main()
