"""Compare three pretrained Iter-E2EDET inference resolutions on MOT17 validation."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from Utils import config
from Utils.detections import file_digest
from Utils.splits import validate_video_splits


ROOT = Path(__file__).resolve().parent
RESOLUTIONS = {
    "original_800_1333": (800, 1333),
    "higher_960_1600": (960, 1600),
    "native_1080_1920": (1080, 1920),
}
SCORES = (0.1, 0.2, 0.3, 0.5, 0.7)
YOLOX_REPORT = ROOT / "outputs/detector_evaluation/20261123_215328_261233_yolox/summary.csv"
BASELINE_TRACKERS = {
    "original_previous": ROOT / "outputs/phase1_iter_frozen_050_070/iter_e2edet/frozen_osnet/without_gcn",
    "yolox_x": ROOT / "outputs/phase1_yolox_frozen_050_070/yolox/frozen_osnet/without_gcn",
}


def run(command, *, capture=False, log=None):
    print("\n$ " + " ".join(map(str, command)), flush=True)
    if capture:
        return subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
    with Path(log).open("wb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, bufsize=0)
        tail = bytearray()
        try:
            while chunk := process.stdout.read(4096):
                stream.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                tail.extend(chunk)
                del tail[:-16000]
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return subprocess.CompletedProcess(command, process.wait(),
                                           tail.decode(errors="replace"))


def failure_status(result):
    if result.returncode == 0:
        return "ok"
    return "OOM" if "out of memory" in (result.stdout or "").lower() else f"failed ({result.returncode})"


def row_at(path, score, iou=0.5):
    with Path(path).open(newline="") as stream:
        rows = csv.DictReader(stream)
        return next((row for row in rows if row["sequence"] == "COMBINED"
                     and float(row["iou"]) == iou and row["score"] == str(score)), None)


def detector_fps(cache):
    summaries = [json.loads((cache/config.DATASET_SPLIT/name/"summary.json").read_text())
                 for name in config.VAL_SEQUENCES]
    frames = sum(item["identity"]["frames"] for item in summaries)
    seconds = sum(item["seconds"] for item in summaries)
    return round(frames/seconds, 2)


def tracker_metrics(path):
    report = json.loads((path/"report.json").read_text())["metrics"]
    summaries = [json.loads((path/"tracking/val"/name/"summary.json").read_text())
                 for name in config.VAL_SEQUENCES]
    report["tracker_fps"] = round(sum(item["frames"] for item in summaries) /
                                  sum(item["seconds"] for item in summaries), 2)
    return report


def percent(row, key):
    return f"{100*float(row[key]):.2f}" if row is not None and row.get(key) not in (None, "") else "—"


def write_report(root, status, cache_root):
    lines = ["# Pretrained Iter-E2EDET inference resolution experiment", "",
             "Same official CrowdHuman checkpoint, official RGB preprocessing, top-500 proposals,",
             "original-image box postprocessing, and no NMS in all three arms. Full MOT17-02-SDP",
             "and MOT17-09-SDP validation videos. Detector GT is used only for scoring.", "",
             "## GPU single-frame checks", "",
             "| Resolution | Size | Status | Detail |", "| --- | --- | --- | --- |"]
    for name, (short, maximum) in RESOLUTIONS.items():
        result = status[name]
        detail = result.get("smoke_error", "")[-180:].replace("|", "/").replace("\n", " ")
        if result.get("smoke_result"):
            check = result["smoke_result"]
            detail = f"{check['proposals']} proposals; {check['seconds']:.2f} s"
        lines.append(f"| {name} | {short}/{maximum} | {result['smoke']} | {detail} |")
    lines += ["", "## Detection at IoU 0.5, confidence 0.30", "",
              "| Source | Precision | Recall | F1 | F1 at IoU 0.75 | Small <50px recall | Heavy occlusion recall | Detector FPS |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in (*RESOLUTIONS, "yolox_x"):
        csv_path = (YOLOX_REPORT if name == "yolox_x" else root/"detector"/name/"summary.csv")
        if not csv_path.is_file():
            lines.append(f"| {name} | {' | '.join(['—']*7)} |")
            continue
        row = row_at(csv_path, 0.3)
        strict = row_at(csv_path, 0.3, 0.75)
        cache = ROOT/"detections/yolox_x_mot20_val_v1" if name == "yolox_x" else cache_root/name
        fps = detector_fps(cache)
        lines.append(f"| {name} | {percent(row,'precision')} | {percent(row,'recall')} | "
                     f"{percent(row,'f1')} | {percent(strict,'f1')} | "
                     f"{percent(row,'height_lt_50_recall')} | "
                     f"{percent(row,'visibility_lt_0.3_recall')} | {fps:.2f} |")
    lines += ["", "All arms sweep confidence 0.10, 0.20, 0.30, 0.50 and 0.70 at IoU 0.5 and 0.75.",
              "Each detector report has the full sweep and per-video counts.", "",
              "## Frozen OSNet tracker, GCN off; read score 0.50, new-track score 0.70", "",
              "| Source | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Tracker FPS |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in (*RESOLUTIONS, *BASELINE_TRACKERS):
        path = BASELINE_TRACKERS.get(name, root/"tracker"/name/"iter_e2edet/frozen_osnet/without_gcn")
        if not (path/"report.json").is_file():
            lines.append(f"| {name} | {' | '.join(['—']*9)} |")
            continue
        m = tracker_metrics(path)
        lines.append(f"| {name} | {m['HOTA']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} | "
                     f"{m['IDF1']:.3f} | {m['MOTA']:.3f} | {m['IDSW']} | "
                     f"{m['CLR_FP']} | {m['CLR_FN']} | {m['tracker_fps']:.2f} |")
    lines += ["", "Detector FPS includes image reads, inference and export; tracker FPS includes image reads and tracking,",
              "but excludes model loading. They are separate measurements. YOLOX applies NMS at IoU 0.7;",
              "Iter-E2EDET does not. Checkpoint and cache hashes are recorded in each cache summary and tracker manifest.",
              "The original_previous row is a separate earlier run with the same 800/1333 recipe.",
              "The pretrained YOLOX recipe reports upstream validation on MOT17 train, so these local videos",
              "are development comparisons, not untouched test data.", "",
              "## Run status", "", "```json", json.dumps(status, indent=2), "```", ""]
    (root/"report.md").write_text("\n".join(lines))
    (root/"status.json").write_text(json.dumps(status, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"weights/iter_e2edet/model.pth")
    parser.add_argument("--cache-root", type=Path, default=ROOT/"detections/iter_e2edet_resolution")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/iter_e2edet_resolution" /
                        datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("This controlled experiment requires GPU smoke tests; use --device cuda")
    if not args.checkpoint.is_file():
        parser.error(f"Missing official checkpoint: {args.checkpoint}")
    if args.output_dir.exists():
        parser.error(f"Choose a new output directory: {args.output_dir}")
    validate_video_splits()
    if tuple(config.VAL_SEQUENCES) != ("MOT17-02-SDP", "MOT17-09-SDP"):
        parser.error("Expected MOT17-02-SDP and MOT17-09-SDP as validation videos")
    root, cache_root = args.output_dir.resolve(), args.cache_root.resolve()
    root.mkdir(parents=True)
    status = {name: {"short_side": short, "max_size": maximum, "smoke": "pending",
                     "export": "pending", "detector_evaluation": "pending", "tracking": "pending"}
              for name, (short, maximum) in RESOLUTIONS.items()}
    (root/"checkpoint.json").write_text(json.dumps({"path": str(args.checkpoint.resolve()),
        "sha256": file_digest(args.checkpoint), "device": args.device}, indent=2)+"\n")
    try:
        # Finish every single-frame GPU check before any full-video export.
        for name, (short, maximum) in RESOLUTIONS.items():
            command = [sys.executable, ROOT/"detect-itere2edet.py", "--checkpoint", args.checkpoint,
                       "--split", "val", "--device", args.device, "--short-side", str(short),
                       "--max-size", str(maximum), "--smoke-test"]
            result = run(command, capture=True)
            output = result.stdout or ""
            if result.returncode == 0:
                status[name]["smoke"] = "ok"
                status[name]["smoke_result"] = json.loads(output.strip().splitlines()[-1])
                print(output, flush=True)
            else:
                status[name]["smoke"] = "OOM" if "out of memory" in output.lower() else "failed"
                status[name]["smoke_error"] = output[-3000:]
                print(output[-1000:], flush=True)
            write_report(root, status, cache_root)
        for name, (short, maximum) in RESOLUTIONS.items():
            if status[name]["smoke"] != "ok":
                continue
            cache = cache_root/name
            export = run([sys.executable, ROOT/"detect-itere2edet.py", "--checkpoint", args.checkpoint,
                          "--split", "val", "--cache-dir", cache, "--device", args.device,
                          "--min-score", "0.05", "--short-side", str(short), "--max-size", str(maximum)],
                         log=root/f"{name}_export.log")
            status[name]["export"] = failure_status(export)
            if export.returncode:
                status[name]["export_error"] = export.stdout[-3000:]
            write_report(root, status, cache_root)
            if export.returncode:
                continue
            evaluated = run([sys.executable, ROOT/"evaluate_detector.py", "--backend", "iter_e2edet",
                             "--cache-dir", cache, "--split", "val", "--scores", *map(str, SCORES),
                             "--ious", "0.5", "0.75", "--output-dir", root/"detector"/name],
                            log=root/f"{name}_detector.log")
            status[name]["detector_evaluation"] = failure_status(evaluated)
            if evaluated.returncode:
                status[name]["detector_error"] = evaluated.stdout[-3000:]
            write_report(root, status, cache_root)
            if evaluated.returncode:
                continue
            tracked = run([sys.executable, ROOT/"evaluate-tracker-variants.py", "--detections", "iter_e2edet",
                           "--iter-checkpoint", args.checkpoint, "--iter-cache-dir", cache,
                           "--encoder", "frozen_osnet", "--graph", "off",
                           "--detection-min-score", "0.5", "--new-track-min-score", "0.7",
                           "--device", args.device, "--output-dir", root/"tracker"/name],
                          log=root/f"{name}_tracker.log")
            status[name]["tracking"] = failure_status(tracked)
            if tracked.returncode:
                status[name]["tracker_error"] = tracked.stdout[-3000:]
            write_report(root, status, cache_root)
    finally:
        write_report(root, status, cache_root)
        print(f"Consolidated report: {root/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
