"""Train the pinned authors' DragonTrack release on disjoint MOT17 videos.

Run from the project root: python Baselines/DragonTrackGithub/train.py --help
The authors' graph/model/training modules live beside this adapter.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from Utils import config  # noqa: E402
from Utils.data import _sequence_info  # noqa: E402
from data_preparation_train import data_prep_train  # noqa: E402
from model_training import model_training  # noqa: E402

UPSTREAM_COMMIT = "111e263c650f2f3a6ef39a261582eef87e09e8d8"


def prepare(sequences, kind, root, args):
    graphs = []
    for sequence in tqdm(sequences, desc=f"Original DragonTrack {kind} graphs", unit="video"):
        path = root / sequence
        length = _sequence_info(path).getint("seqLength")
        rows = np.loadtxt(path / "gt/gt.txt", delimiter=",", ndmin=2)
        if rows.shape[1] < 9:
            raise ValueError(f"Invalid MOT ground truth: {path / 'gt/gt.txt'}")
        graphs.extend(data_prep_train(
            sequence, rows, str(path / "img1"), args.lookback, length,
            args.lookback, args.graph_jump,
            2, length if kind == "training" else 2,
            args.distance_limit, _sequence_info(path).getint("frameRate"), kind))
    if not graphs:
        raise ValueError(f"No usable {kind} graphs; inspect the selected sequences")
    return graphs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sequences", nargs="+", default=config.TRAIN_SEQUENCES)
    parser.add_argument("--val-sequences", nargs="+", default=config.VAL_SEQUENCES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--graph-jump", type=int, default=5)
    parser.add_argument("--distance-limit", type=float, default=250)
    parser.add_argument("--checkpoint", type=Path, help="Authors-format epoch_N.pth to resume")
    parser.add_argument("--resume", action="store_true", help="Resume latest completed epoch in the same output folder")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.lookback, args.graph_jump) <= 0 or args.distance_limit <= 0:
        parser.error("Epochs, batch size, lookback, graph jump and distance limit must be positive")
    if set(args.train_sequences) & set(args.val_sequences):
        parser.error("Training and validation must use different videos")
    if not torch.cuda.is_available():
        parser.error("The authors' released model requires CUDA; activate the GPU-enabled dragon-track environment")
    root = config.DATASET_ROOT / config.DATASET_SPLIT
    output = args.output_dir or PROJECT / "outputs/github_dragontrack" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output.exists() and not args.resume:
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models_v3").mkdir(exist_ok=True)
    (output / "logfiles").mkdir(exist_ok=True)
    checkpoints = sorted((output / "models_v3").glob("epoch_*.pth"),
                         key=lambda path: int(path.stem.split("_")[-1]))
    checkpoint = args.checkpoint.resolve() if args.checkpoint else (checkpoints[-1] if args.resume and checkpoints else None)
    if args.resume and args.checkpoint is None and not checkpoints:
        raise FileNotFoundError("No completed authors' epoch to resume")
    if args.resume and checkpoint is not None and checkpoint.parent != (output / "models_v3").resolve():
        raise ValueError("Resume checkpoint must belong to this output folder")
    (output / "settings.json").write_text(json.dumps({
        "upstream_commit": UPSTREAM_COMMIT,
        "train_sequences": args.train_sequences, "val_sequences": args.val_sequences,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lookback": args.lookback, "graph_jump": args.graph_jump,
        "distance_limit": args.distance_limit, "dataset_root": str(root.resolve()),
        "resume_checkpoint": str(checkpoint) if checkpoint else None,
        "training_boxes": "MOT GT, as in the authors' released training path",
    }, indent=2) + "\n")
    training = prepare(args.train_sequences, "training", root, args)
    validation = prepare(args.val_sequences, "validation", root, args)
    print(f"Prepared {len(training)} training and {len(validation)} validation graphs")
    previous = Path.cwd()
    try:
        os.chdir(output)
        model_training(training, validation, args.epochs, args.epochs, 2, 1,
                       2, args.batch_size, "training.log", str(checkpoint) if checkpoint else None)
    finally:
        os.chdir(previous)
    print(f"Authors-format checkpoints: {output / 'models_v3'}")
    return output


if __name__ == "__main__":
    main()
