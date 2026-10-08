"""Evaluate person detection caches without training or tracking.

Run python evaluate_detector.py --help. See Documentation/Implementation/24-detector-evaluation.md.
Metrics are a detector diagnostic, not official MOT scores or COCO AP.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from Utils import config
from Utils.data import (_sequence_info, load_sequence, load_ground_truth,
                        match_detections_to_gt, pairwise_iou)
from Utils.detections import cache_path, file_digest, validate_tracktor
from Utils.splits import validate_video_splits

HEIGHT_BINS = ("height_lt_50", "height_50_to_100", "height_ge_100")
VISIBILITY_BINS = ("visibility_lt_0.3", "visibility_0.3_to_0.7", "visibility_ge_0.7")
COUNT_FIELDS = ("frames", "gt", "detections", "tp", "fp", "fn", "ignored", "duplicates") + tuple(
    f"{name}_{count}" for name in HEIGHT_BINS + VISIBILITY_BINS for count in ("gt", "tp"))


def resolve_source(sequence_path, backend):
    """Verify a saved cache using its own recipe, so old/new resolutions can be compared.

    Supplied files have no completion manifest; assume they cover seqinfo's full video.
    RT-DETR manifests must prove coverage and integrity, but need not match the active
    training config. Record their exact recipe and hash in this evaluation's report.
    """
    info = _sequence_info(sequence_path)
    if backend == "tracktor":
        path = validate_tracktor(sequence_path)
        metadata = json.loads(path.with_name("summary.json").read_text())
    elif backend == "rtdetrv2":
        if config.BOX_ORIGIN != 1:
            raise ValueError("RT-DETR caches use MOT's BOX_ORIGIN=1")
        path = cache_path(sequence_path).resolve()
        metadata = json.loads(path.with_name("summary.json").read_text())
        identity = metadata.get("identity", {})
        expected = {"sequence_path": str(sequence_path.resolve()), "frames": info.getint("seqLength"),
                    "width": info.getint("imWidth"), "height": info.getint("imHeight")}
        recipe = identity.get("detector", {})
        if (metadata.get("complete") is not True or any(identity.get(k) != v for k, v in expected.items())
                or recipe.get("format") != "mot_person_v1" or recipe.get("box_origin") != 1
                or metadata.get("sha256") != file_digest(path)):
            raise ValueError(f"Incomplete, changed, or incompatible detection cache: {path}")
    else:
        path = Path(config.DETECTION_FILE)
        if not path.is_absolute():
            path = sequence_path / path
        metadata = {"coverage": "Assumed full video for supplied MOT detections; no completion manifest"}
    return path.resolve(), {"path": str(path.resolve()), "sha256": file_digest(path), "metadata": metadata}


def score_frame(boxes, gt, iou_threshold):
    """Maximum-cardinality one-to-one matches, then IoU as tie-breaker.

    Marked class-1 people are targets. Unmatched boxes overlapping another annotation
    at the same IoU are ignored, unless they also overlap a target (duplicate FP).
    This explicit diagnostic policy differs from official MOT distractor preprocessing.
    """
    result = match_detections_to_gt(boxes, gt, iou_threshold)
    valid = gt["valid_mask"]
    matched = result["valid_mask"]
    overlaps_target = (pairwise_iou(boxes, gt["boxes"][valid]) >= iou_threshold).any(axis=1)
    duplicate = ~matched & overlaps_target
    ignored = result["ignored_mask"] & ~duplicate
    hit = np.isin(gt["ids"][valid], result["ids"][matched])
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts.update(frames=1, gt=int(valid.sum()), detections=len(boxes), tp=int(matched.sum()),
                  fp=int((~matched & ~ignored).sum()), fn=int(valid.sum()-matched.sum()),
                  ignored=int(ignored.sum()), duplicates=int(duplicate.sum()))
    heights = gt["boxes"][valid, 3] - gt["boxes"][valid, 1]
    visibility = gt["visibility"][valid]
    buckets = (heights < 50, (heights >= 50) & (heights < 100), heights >= 100,
               visibility < 0.3, (visibility >= 0.3) & (visibility < 0.7), visibility >= 0.7)
    for name, mask in zip(HEIGHT_BINS + VISIBILITY_BINS, buckets, strict=True):
        counts[f"{name}_gt"] = int(mask.sum())
        counts[f"{name}_tp"] = int((hit & mask).sum())
    return counts


def metrics(counts):
    """Count-weighted rates; undefined denominators are null/blank, not fake zeros."""
    result = dict(counts)
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    result.update(precision=tp/(tp+fp) if tp+fp else None,
                  recall=tp/(tp+fn) if tp+fn else None,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    for name in HEIGHT_BINS + VISIBILITY_BINS:
        result[f"{name}_recall"] = (counts[f"{name}_tp"] / counts[f"{name}_gt"]
                                    if counts[f"{name}_gt"] else None)
    return result


def write_reports(output_dir, rows, provenance, args):
    """Save comparable counts/rates, source recipes and confidence/recall plots."""
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"backend": args.backend, "split": args.split, "iou_thresholds": args.ious,
              "score_thresholds": [None] + args.scores, "max_frames": args.max_frames,
              "rates": "0 to 1; null means undefined", "sources": provenance, "results": rows,
              "policy": "Marked class-1 GT; maximum-cardinality IoU matching. Non-target annotation overlaps "
                        "ignored only for unmatched non-duplicate detections. Not MOT/COCO benchmark metrics.",
              "read_filter": "DETECTION_MIN_SCORE ignored; thresholds applied explicitly to every cached row"}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    combined = [row for row in rows if row["sequence"] == "COMBINED"]
    lines = ["# Detector evaluation", "", f"Source: **{args.backend}**, split: **{args.split}**.", "",
             "This measures boxes only, not identities. These are diagnostic precision/recall/F1, not HOTA, "
             "MOTA, IDF1 or COCO AP. GT: marked class-1 pedestrians. Matches maximize count at each IoU; "
             "unmatched overlaps with non-target annotations are ignored except duplicates of target people.", "",
             "All = every cached box, regardless of DETECTION_MIN_SCORE. Cached-out boxes cannot be recovered. "
             "Tracktor's substituted score 1 is not calibrated confidence. Supplied SDP scores may also be "
             "uncalibrated. Compare sources at similar precision, using identical videos/frame limits.", "",
             "| IoU | Minimum score | Precision | Recall | F1 | TP | FP | FN | Duplicates | Ignored |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in combined:
        values = ["n/a" if row[key] is None else f"{100*row[key]:.2f}%" for key in ("precision", "recall", "f1")]
        lines.append(f"| {row['iou']} | {row['score']} | " + " | ".join(values) +
                     " | " + " | ".join(str(row[k]) for k in ("tp", "fp", "fn", "duplicates", "ignored")) + " |")
    lines += ["", "Size uses GT box height at original resolution: <50, 50–<100, >=100 pixels. "
              "Visibility bins: <0.3, 0.3–<0.7, >=0.7. All valid GT contribute to their bins; unmatched GT "
              "count as misses. See summary.csv for per-video counts and bin recalls.", "",
              "The combined result sums counts before calculating rates. Plots use IoU " + str(args.ious[0]) + ".",
              "", "Source paths, checksums, full cache recipes and original cache timings are in summary.json. "
              "Those timings measure cache generation, not this evaluator or necessarily GPU-only inference."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in combined if r["iou"] == args.ious[0] and r["score"] != "all"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("precision", "recall", "f1"):
        axes[0].plot([r["score"] for r in selected], [np.nan if r[key] is None else r[key] for r in selected],
                     marker="o", label=key)
    axes[0].set(xlabel="Minimum cached score", ylabel="Rate", ylim=(0, 1), title="Confidence sweep")
    axes[0].legend()
    raw = next(r for r in combined if r["iou"] == args.ious[0] and r["score"] == "all")
    axes[1].bar(["<50px", "50–<100px", ">=100px"],
                [np.nan if raw[f"{b}_recall"] is None else raw[f"{b}_recall"] for b in HEIGHT_BINS])
    axes[1].set(xlabel="GT person height (original image)", ylabel="Recall", ylim=(0, 1), title="All cached boxes")
    fig.suptitle(f"{args.backend}: combined detector diagnostic at IoU {args.ious[0]}")
    fig.tight_layout()
    fig.savefig(output_dir / "detector_metrics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("supplied", "tracktor", "rtdetrv2"), default=config.DETECTION_BACKEND)
    parser.add_argument("--cache-dir", type=Path, help="RT-DETR cache directory; reads its recorded recipe")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Use val when tuning")
    parser.add_argument("--scores", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7],
                        help="Explicit confidence sweep; all cached boxes are always also evaluated")
    parser.add_argument("--ious", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--max-frames", type=int, help="First N frames per video, for a quick check")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing reports are never overwritten")
    args = parser.parse_args()
    if any(not np.isfinite(x) for x in args.scores) or any(not 0 < x <= 1 for x in args.ious):
        parser.error("scores must be finite; IoUs must be in (0,1]")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.cache_dir is not None:
        if args.backend != "rtdetrv2":
            parser.error("--cache-dir is for --backend rtdetrv2")
        config.DETECTION_CACHE_DIR = args.cache_dir
    args.scores, args.ious = sorted(set(args.scores)), sorted(set(args.ious))
    validate_video_splits()
    sequences = config.VAL_SEQUENCES if args.split == "val" else config.TRAIN_SEQUENCES
    output_dir = args.output_dir or (config.PROJECT_ROOT / "outputs" / "detector_evaluation" /
                                    f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.backend}")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    jobs = []
    for sequence in sequences:
        path = Path(config.DATASET_ROOT) / config.DATASET_SPLIT / sequence
        length = _sequence_info(path).getint("seqLength")
        jobs.append((path, min(length, args.max_frames) if args.max_frames else length))
    settings = [(iou, score) for iou in args.ious for score in [None] + args.scores]
    combined = {setting: dict.fromkeys(COUNT_FIELDS, 0) for setting in settings}
    rows, provenance = [], {}
    with tqdm(total=sum(n for _, n in jobs)*len(settings), desc="Detector evaluation overall", unit="frame-setting",
              smoothing=0, position=0, disable=not config.SHOW_PROGRESS) as overall:
        for path, length in jobs:
            source, provenance[path.name] = resolve_source(path, args.backend)
            provenance[path.name]["evaluated_frames"] = [1, length]
            frames = load_sequence(path, source, (1, length), box_origin=config.BOX_ORIGIN, min_score=None)
            gt = load_ground_truth(path, config.GT_FILE, box_origin=config.BOX_ORIGIN)
            counts = {setting: dict.fromkeys(COUNT_FIELDS, 0) for setting in settings}
            with tqdm(frames.items(), desc=path.name, unit="frame", position=1, leave=False,
                      disable=not config.SHOW_PROGRESS) as progress:
                for frame_id, frame in progress:
                    for setting in settings:
                        iou, score = setting
                        boxes = frame["boxes"] if score is None else frame["boxes"][frame["scores"] >= score]
                        result = score_frame(boxes, gt[frame_id], iou)
                        for key, value in result.items():
                            counts[setting][key] += value
                        overall.update(1)
            for (iou, score), result in counts.items():
                rows.append({"sequence": path.name, "iou": iou, "score": "all" if score is None else score,
                             **metrics(result)})
                for key, value in result.items():
                    combined[iou, score][key] += value
    for (iou, score), result in combined.items():
        rows.append({"sequence": "COMBINED", "iou": iou, "score": "all" if score is None else score,
                     **metrics(result)})
    write_reports(output_dir, rows, provenance, args)
    for row in rows:
        if row["sequence"] == "COMBINED" and row["score"] == "all":
            tqdm.write(f"All boxes, IoU {row['iou']}: TP={row['tp']}, FP={row['fp']}, FN={row['fn']}")
    tqdm.write(f"Detector report: {output_dir / 'report.md'}")
    return output_dir


if __name__ == "__main__":
    main()
