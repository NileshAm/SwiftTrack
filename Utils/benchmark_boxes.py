"""Prepare comparable MOT person-box inputs without exposing ground-truth IDs."""

from pathlib import Path

from Utils.data import load_ground_truth


def gt_person_cache(sequence_path, target):
    """Write marked class-1 person boxes with identity -1; reuse a saved cache."""
    target = Path(target).resolve()
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    truth = load_ground_truth(sequence_path)
    with target.open("x") as stream:
        for frame, record in truth.items():
            for x1, y1, x2, y2 in record["boxes"][record["valid_mask"]]:
                stream.write(f"{frame},-1,{x1+1:.6f},{y1+1:.6f},{x2-x1:.6f},{y2-y1:.6f},1,-1,-1,-1\n")
    return target
