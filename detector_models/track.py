"""Track one verified cache with frozen OSNet, the baseline association checkpoint, and no GCN."""

import argparse
import json
from pathlib import Path
import shutil

import torch
from tqdm.auto import tqdm

from Utils import config, mot_metrics
from Utils.data import _sequence_info
from Utils.detections import file_digest
from Utils.features import load_encoder
from Utils.outputs import tracking_context
from model import AssociationModel, MODEL_VERSION
import tracking


BASELINE = {"HOTA": 44.206, "IDF1": 51.182, "MOTA": 50.310, "IDSW": 226,
            "CLR_FP": 48, "CLR_FN": 11605}
DEFAULT_CHECKPOINT = config.PROJECT_ROOT / "outputs/detector_suite_v1/sources/tracktor/association/model.pt"


def verified_cache(cache_root, sequence):
    sequence_path = config.DATASET_ROOT / config.DATASET_SPLIT / sequence
    path = Path(cache_root) / sequence_path.parent.name / sequence / "det.txt"
    summary_path = path.with_name("summary.json")
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing completed cache: {path}")
    summary = json.loads(summary_path.read_text())
    info = _sequence_info(sequence_path)
    if (summary.get("complete") is not True or summary.get("sha256") != file_digest(path)
            or summary.get("identity", {}).get("frames") != info.getint("seqLength")
            or summary.get("identity", {}).get("detector", {}).get("format") != "mot_person_v1"):
        raise ValueError(f"Incomplete or changed cache: {path}")
    return path.resolve(), summary


def load_baseline(checkpoint_path, device):
    config.ENCODER_NAME = "osnet_reid"
    config.FEATURE_DIM = 512
    config.FREEZE_ENCODER = True
    config.USE_PRETRAINED_WEIGHTS = True
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("model_version") != MODEL_VERSION:
        raise ValueError("Association checkpoint has a different model version")
    if saved["settings"]["ENCODER_NAME"] != "osnet_reid":
        raise ValueError("Association checkpoint is not the frozen OSNet checkpoint")
    for key in ("CROP_SIZE", "GRAPH_HIDDEN_DIMS", "ASSOCIATION_DISTANCE_PX",
                "SINKHORN_ITERATIONS", "SINKHORN_SCORE_SCALE", "SINKHORN_UNMATCHED_LOGIT",
                "MATCH_THRESHOLD"):
        if key in saved["settings"]:
            setattr(config, key, saved["settings"][key])
    encoder = load_encoder(device)
    if encoder.feature_settings != saved["feature_settings"]:
        raise ValueError("OSNet weights/preprocessing differ from association training")
    model = AssociationModel(**saved["model_settings"]).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval(); encoder.eval()
    return model, encoder, saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--operating-point", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Tracking output already exists: {args.output_dir}")
    selected = json.loads(args.operating_point.read_text())["operating_point"]
    score = float(selected["score"])
    config.DEVICE = args.device
    config.DETECTION_BACKEND = "supplied"
    config.DETECTION_MIN_SCORE = score
    config.NEW_TRACK_MIN_SCORE = .5
    config.BOX_ORIGIN = 1
    config.OUTPUT_DIR = args.output_dir
    config.TRACK_OUTPUT_DIR = args.output_dir / "tracking/val"
    config.EVALUATION_OUTPUT_DIR = args.output_dir / "evaluation/val"
    config.SAVE_VIDEO = False
    model, encoder, checkpoint = load_baseline(args.checkpoint, args.device)
    sources = {name: verified_cache(args.cache_dir, name) for name in config.VAL_SEQUENCES}
    args.output_dir.mkdir(parents=True)
    manifest = {"encoder": encoder.feature_settings, "use_graph": False,
                "association_checkpoint": str(args.checkpoint.resolve()),
                "association_sha256": file_digest(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
                "detection_score": score, "new_track_min_score": .5,
                "baseline": BASELINE, "sources": {k: {"path": str(v[0]), "summary": v[1]} for k, v in sources.items()}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    total = sum(_sequence_info(config.DATASET_ROOT / config.DATASET_SPLIT / name).getint("seqLength")
                for name in config.VAL_SEQUENCES)
    with tqdm(total=total, desc="OSNet no-GCN tracking", unit="frame") as overall:
        for sequence, (path, _) in sources.items():
            config.DETECTION_FILE = path
            tracking.run_sequence(sequence, model, encoder, split="val", overall=overall, use_graph=False)
            inputs = args.output_dir / "evaluation/val/inputs/MOT17"
            gt = config.DATASET_ROOT / config.DATASET_SPLIT / sequence / "gt/gt.txt"
            gt_target = inputs / "gt" / sequence / "gt/gt.txt"
            gt_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(gt, gt_target)
            track_target = inputs / "trackers/DragonTrack/data" / f"{sequence}.txt"
            track_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config.TRACK_OUTPUT_DIR / sequence / "tracks.txt", track_target)
    measured = mot_metrics.load(args.output_dir)
    scores = mot_metrics.score(measured)
    sequence_summaries = [json.loads((config.TRACK_OUTPUT_DIR / name / "summary.json").read_text())
                          for name in config.VAL_SEQUENCES]
    tracker_fps = sum(item["frames"] for item in sequence_summaries) / sum(item["seconds"] for item in sequence_summaries)
    result = {"metrics": scores, "baseline": BASELINE, "tracker_fps": tracker_fps,
              "delta": {key: scores[key] - BASELINE[key] for key in BASELINE if key in scores},
              "improves_end_to_end": scores["HOTA"] > BASELINE["HOTA"] and scores["IDF1"] > BASELINE["IDF1"]}
    (args.output_dir / "tracking_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

