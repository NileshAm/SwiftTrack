"""Resumable sequential runner: export -> box evaluation -> threshold -> OSNet/no-GCN tracking."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from Utils import config
from .registry import ORDER

SCORES = ("0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40", "0.50", "0.60", "0.70")


def execute(command, log):
    print("\n$ " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=config.PROJECT_ROOT)
    log.append({"command": command, "returncode": completed.returncode})
    return completed.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("all", *ORDER), default="all")
    parser.add_argument("--output-root", type=Path, default=config.PROJECT_ROOT / "outputs/crowded_detector_suite_v1")
    parser.add_argument("--deim-repo", type=Path, default=config.PROJECT_ROOT / "detector_models/official/DEIMv2")
    parser.add_argument("--deim-weights", type=Path, default=config.PROJECT_ROOT / "detector_models/weights/deimv2_dinov3_s_coco.pth")
    parser.add_argument("--association-checkpoint", type=Path,
                        default=config.PROJECT_ROOT / "outputs/detector_suite_v1/sources/tracktor/association/model.pt")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--min-precision", type=float, default=.70)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    names = ORDER if args.detector == "all" else (args.detector,)
    log = []
    for name in names:
        arm = args.output_root / name
        limitation, manifest = arm / "limitation.json", arm / "detector_manifest.json"
        retryable = (limitation.is_file() and json.loads(limitation.read_text()).get("retryable", False))
        if (not limitation.is_file() or retryable) and not manifest.is_file():
            command = [sys.executable, "-m", "detector_models.export", "--detector", name,
                       "--output-root", str(args.output_root), "--device", args.device]
            if name == "deimv2_s":
                command += ["--official-repo", str(args.deim_repo), "--weights", str(args.deim_weights)]
            code = execute(command, log)
            if code not in (0, 20):
                print(f"FAILED {name}; fix the recorded error and rerun. Later detectors are not started.")
                break
        if limitation.is_file() and not manifest.is_file():
            print(f"Skipping unavailable {name}; see {limitation}")
            continue
        evaluation = arm / "detector_evaluation"
        summary = evaluation / "summary.json"
        if not summary.is_file():
            code = execute([sys.executable, "evaluate_detector.py", "--backend", "cached",
                "--cache-dir", str(arm / "detections"), "--split", "val", "--scores", *SCORES,
                "--ious", "0.5", "0.75", "--output-dir", str(evaluation)], log)
            if code:
                break
        selected = evaluation / "selected_operating_point.json"
        if not selected.is_file():
            if execute([sys.executable, "-m", "detector_models.select_threshold", str(summary),
                        "--min-precision", str(args.min_precision), "--output", str(selected)], log):
                break
        tracking_result = arm / "tracking_osnet_no_gcn/tracking_result.json"
        if not tracking_result.is_file():
            if execute([sys.executable, "-m", "detector_models.track",
                        "--cache-dir", str(arm / "detections"),
                        "--output-dir", str(arm / "tracking_osnet_no_gcn"),
                        "--operating-point", str(selected), "--checkpoint", str(args.association_checkpoint),
                        "--device", args.device], log):
                break
        execute([sys.executable, "-m", "detector_models.report", "--output-root", str(args.output_root)], log)
    execute([sys.executable, "-m", "detector_models.report", "--output-root", str(args.output_root)], log)
    (args.output_root / "commands.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"\nConsolidated report: {args.output_root / 'comparison.md'}")


if __name__ == "__main__":
    main()

