"""Search tracking thresholds on the pretrained Iter-E2EDET 960/1600 cache.

Uses frozen OSNet, the existing association checkpoint, and no GCN. Does not
train, regenerate detections, or modify config.py. Completed trials are reused.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

from tqdm.auto import tqdm

from Utils import config
from Utils.detections import file_digest
from Utils.itere2edet import cache_for_sequence


ROOT = Path(__file__).resolve().parent
FIELDS = ("detection_score", "new_track_score", "HOTA", "DetA", "AssA",
          "IDF1", "MOTA", "IDSW", "FP", "FN", "tracker_fps", "result_dir")


def trial_name(detection_score, new_track_score):
    return f"det_{detection_score}_new_{new_track_score}".replace(".", "p")


def read_result(folder, detection_score, new_track_score):
    arm = folder/"iter_e2edet/frozen_osnet/without_gcn"
    metrics = json.loads((arm/"report.json").read_text())["metrics"]
    summaries = [json.loads((arm/"tracking/val"/name/"summary.json").read_text())
                 for name in config.VAL_SEQUENCES]
    fps = sum(item["frames"] for item in summaries)/sum(item["seconds"] for item in summaries)
    return dict(detection_score=detection_score, new_track_score=new_track_score,
                HOTA=metrics["HOTA"], DetA=metrics["DetA"], AssA=metrics["AssA"],
                IDF1=metrics["IDF1"], MOTA=metrics["MOTA"], IDSW=metrics["IDSW"],
                FP=metrics["CLR_FP"], FN=metrics["CLR_FN"],
                tracker_fps=round(fps, 2), result_dir=str(folder.resolve()))


def save_results(root, rows):
    ranked = sorted(rows, key=lambda row: (-row["HOTA"], -row["IDF1"]))
    pending = root/"results.csv.partial"
    with pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ranked)
    pending.replace(root/"results.csv")
    lines = ["# Iter-E2EDET 960/1600 tracking threshold search", "",
             "Frozen OSNet, the same association checkpoint, GCN disabled, full",
             "MOT17-02-SDP and MOT17-09-SDP validation videos. Detections come from",
             "the existing 960/1600 cache; no detector or tracker weights are trained.",
             "Ranks by HOTA, then IDF1. These are development validation scores.", "",
             "| Detection score | New-track score | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Tracker FPS |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in ranked:
        lines.append("| " + " | ".join(str(row[key]) for key in FIELDS[:-1]) + " |")
    (root/"report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT/"detections/iter_e2edet_resolution_v2/higher_960_1600")
    parser.add_argument("--iter-checkpoint", type=Path, default=ROOT/"weights/iter_e2edet/model.pth")
    parser.add_argument("--association-checkpoint", type=Path,
                        default=ROOT/"outputs/encoder_comparison_reid_v1/model.pt")
    parser.add_argument("--detection-scores", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    parser.add_argument("--new-track-scores", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/iter_e2edet_960_threshold_search")
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()
    if any(not 0 <= value <= 1 for value in (*args.detection_scores, *args.new_track_scores)):
        parser.error("Scores must be between 0 and 1")
    if not args.iter_checkpoint.is_file() or not args.association_checkpoint.is_file():
        parser.error("The detector and association checkpoints must exist")
    if tuple(config.VAL_SEQUENCES) != ("MOT17-02-SDP", "MOT17-09-SDP"):
        parser.error("This comparison expects MOT17-02-SDP and MOT17-09-SDP validation videos")
    cache = args.cache_dir.resolve()
    for name in config.VAL_SEQUENCES:
        sequence = config.DATASET_ROOT/config.DATASET_SPLIT/name
        _, summary = cache_for_sequence(cache, sequence, args.iter_checkpoint)
        recipe = summary["identity"]["detector"]
        if (recipe.get("short_side"), recipe.get("max_size")) != (960, 1600):
            parser.error(f"{name}: expected 960/1600 pretrained Iter-E2EDET cache")
        if min(args.detection_scores) < recipe["min_score"]:
            parser.error(f"{name}: detection score below cache export floor")
    trials = sorted(set(itertools.product(args.detection_scores, args.new_track_scores)))
    if any(new < detection for detection, new in trials):
        parser.error("Each new-track score must be at least its detection score")
    root = args.output_dir.resolve()
    manifest = {"cache_dir": str(cache), "detector_checkpoint_sha256": file_digest(args.iter_checkpoint),
                "association_checkpoint_sha256": file_digest(args.association_checkpoint),
                "trials": [list(pair) for pair in trials], "device": args.device,
                "encoder": "frozen_osnet", "graph": "off",
                "validation_sequences": config.VAL_SEQUENCES}
    root.mkdir(parents=True, exist_ok=True)
    manifest_file = root/"manifest.json"
    if manifest_file.is_file():
        if json.loads(manifest_file.read_text()) != manifest:
            parser.error(f"{root}: existing search has different settings; choose a new --output-dir")
    elif any(root.iterdir()):
        parser.error(f"{root}: nonempty directory without manifest; choose a new --output-dir")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    with tqdm(total=len(trials), desc="Tracking threshold search", unit="trial") as overall:
        for index, (detection, new) in enumerate(trials, 1):
            name = trial_name(detection, new)
            folder = root/name
            report = folder/"iter_e2edet/frozen_osnet/without_gcn/report.json"
            if not report.is_file():
                if folder.exists():
                    parser.error(f"{folder}: incomplete trial; inspect its files and resume in a new output directory")
                tqdm.write(f"Trial {index}/{len(trials)}: detection {detection:.2f}, new track {new:.2f}")
                command = [sys.executable, ROOT/"evaluate-tracker-variants.py", "--detections", "iter_e2edet",
                           "--iter-checkpoint", args.iter_checkpoint, "--iter-cache-dir", cache,
                           "--encoder", "frozen_osnet", "--graph", "off",
                           "--frozen-osnet-association", args.association_checkpoint,
                           "--detection-min-score", str(detection), "--new-track-min-score", str(new),
                           "--device", args.device, "--output-dir", folder]
                result = subprocess.run(command, cwd=ROOT, check=False)
                if result.returncode:
                    raise RuntimeError(f"Trial {name} failed with exit code {result.returncode}; see {folder}")
            else:
                tqdm.write(f"Reusing completed trial {index}/{len(trials)}: {name}")
            rows.append(read_result(folder, detection, new))
            save_results(root, rows)
            overall.update(1)
    print(f"Ranked results: {root/'report.md'}")


if __name__ == "__main__":
    main()
