
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_mot(path):
    """Read MOT-format CSV: frame,id,x,y,w,h,score,..."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        return np.empty((0, 10))

    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] < 6:
        raise ValueError(f"{path}: expected at least 6 columns")
    return data


def boxes_for_frame(data, frame):
    rows = data[data[:, 0].astype(int) == frame]
    boxes = rows[:, 2:6].astype(float).copy()
    boxes[:, 2:4] += boxes[:, 0:2]  # xywh -> xyxy
    return boxes


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(br - tl, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    return inter / np.maximum(
        area_a[:, None] + area_b[None, :] - inter, 1e-9
    )


def match(a, b, threshold=0.5):
    ious = iou_matrix(a, b)
    if not ious.size:
        return []

    # Maximize valid IoU matches; invalid edges receive no reward.
    scores = np.where(ious >= threshold, 1.0 + ious, 0.0)
    rows, cols = linear_sum_assignment(scores, maximize=True)
    return [
        (int(i), int(j))
        for i, j in zip(rows, cols)
        if ious[i, j] >= threshold
    ]


def gt_boxes(data, frame):
    rows = data[data[:, 0].astype(int) == frame]

    # MOT17: column 7 = marked, column 8 = class.
    # Retain marked pedestrian (class 1) annotations.
    if rows.shape[1] >= 8:
        rows = rows[
            (rows[:, 6] == 1) &
            (rows[:, 7] == 1)
        ]

    boxes = rows[:, 2:6].astype(float).copy()
    boxes[:, 2:4] += boxes[:, :2]
    return boxes


def evaluate_stage(gt, pred, frames, threshold):
    tp = fp = fn = 0
    matched_gt = set()
    total_pred = 0

    for frame in frames:
        g = gt_boxes(gt, frame)
        p = boxes_for_frame(pred, frame)
        pairs = match(g, p, threshold)

        tp += len(pairs)
        fp += len(p) - len(pairs)
        fn += len(g) - len(pairs)
        total_pred += len(p)

        matched_gt.update((frame, i) for i, _ in pairs)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "P": 100 * tp / max(tp + fp, 1),
        "R": 100 * tp / max(tp + fn, 1),
        "F1": 200 * tp / max(2 * tp + fp + fn, 1),
        "boxes": total_pred,
        "matched_gt": matched_gt,
    }


def compare_stages(a, b, frames, threshold):
    """Approximate how many boxes survive between stages."""
    total_a = total_b = paired = 0

    for frame in frames:
        aa = boxes_for_frame(a, frame)
        bb = boxes_for_frame(b, frame)
        total_a += len(aa)
        total_b += len(bb)
        paired += len(match(aa, bb, threshold))

    print(
        f"  Previous boxes: {total_a}\n"
        f"  Next boxes:     {total_b}\n"
        f"  Matched boxes:  {paired}\n"
        f"  Unmatched previous: {total_a - paired}\n"
        f"  New/unmatched next: {total_b - paired}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--input", help="Actual boxes entering tracker")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    gt = load_mot(args.gt)
    raw = load_mot(args.raw)
    tracks = load_mot(args.tracks)
    tracker_input = load_mot(args.input) if args.input else None

    frames = sorted(set(gt[:, 0].astype(int)))
    stages = [("Raw detections", raw)]
    if tracker_input is not None:
        stages.append(("Tracker input", tracker_input))
    stages.append(("Tracker output", tracks))

    results = []

    print("\n=== DETECTION LOSS DIAGNOSTIC ===")
    print(f"GT frames: {len(frames)} | IoU: {args.iou}")

    for name, data in stages:
        result = evaluate_stage(gt, data, frames, args.iou)
        results.append((name, data, result))

        print(f"\n{name}")
        print(f"  Boxes: {result['boxes']}")
        print(f"  TP={result['TP']} FP={result['FP']} FN={result['FN']}")
        print(
            f"  Precision={result['P']:.2f}% "
            f"Recall={result['R']:.2f}% "
            f"F1={result['F1']:.2f}%"
        )

    for previous, current in zip(results, results[1:]):
        print(f"\n=== {previous[0]} -> {current[0]} ===")
        compare_stages(previous[1], current[1], frames, args.iou)

        before = previous[2]["matched_gt"]
        after = current[2]["matched_gt"]

        print(f"  GT matches lost: {len(before - after)}")
        print(f"  GT matches gained: {len(after - before)}")

    print("\nNOTE:")
    print("Independent framewise matching, NOT official TrackEval.")
    print("Box matching estimates stage losses, not exact provenance.")


if __name__ == "__main__":
    main()
