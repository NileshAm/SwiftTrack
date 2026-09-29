"""Read MOT17/MOT20 files and prepare association-training labels.

Boxes returned here are zero-origin xyxy NumPy arrays. Images are loaded only
when read_image() is called. Ground-truth labels never enter load_sequence().
"""

from configparser import ConfigParser
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def _sequence_info(sequence_path):
    """Read the dataset's dimensions, filenames, FPS, and frame count."""
    path = Path(sequence_path) / "seqinfo.ini"
    parser = ConfigParser()
    with path.open() as file:
        parser.read_file(file)
    info = parser["Sequence"]
    for key in ("imWidth", "imHeight", "seqLength", "frameRate"):
        if info.getint(key) <= 0:
            raise ValueError(f"{path}: {key} must be positive")
    return info


def _read_rows(path, columns, frame_count, box_origin):
    """Read required MOT columns, retaining line numbers for useful errors."""
    if box_origin not in (0, 1):
        raise ValueError("box_origin must be 0 or 1")
    rows_by_frame = {}
    with Path(path).open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = np.asarray(line.strip().split(",")[:columns], dtype=np.float64)
                if len(row) != columns or not np.isfinite(row).all():
                    raise ValueError(f"expected at least {columns} finite numeric columns")
                frame_id = int(row[0])
                if row[0] != frame_id or not 1 <= frame_id <= frame_count:
                    raise ValueError("frame ID is outside the sequence or is not an integer")
                if row[4] <= 0 or row[5] <= 0:
                    raise ValueError("box width and height must be positive")
            except ValueError as error:
                raise ValueError(f"{path}, line {line_number}: {error}") from error
            # Convert xywh -> xyxy, then change the coordinate origin once.
            row[4:6] += row[2:4]
            row[2:6] -= box_origin
            rows_by_frame.setdefault(frame_id, []).append((line_number, row))
    return rows_by_frame


def load_sequence(sequence_path, detection_path=None, frame_range=None,
                  *, box_origin=1, min_score=None):
    """Return {frame_id: frame_record}, including frames without detections.

    sequence_path is a folder such as data/MOT17/train/MOT17-02-FRCNN.
    detection_path defaults to det/det.txt; relative paths are sequence-relative.
    frame_range is an inclusive (first, last) pair, or None for the full video.
    Each record contains image_path, width, height, fps, boxes [D,4], scores [D],
    and row_ids [D] (original detection-file line numbers). Images are not read.
    Boxes are not clipped here: crop extraction should clip its own copy.
    """
    sequence_path = Path(sequence_path)
    info = _sequence_info(sequence_path)
    frame_count = info.getint("seqLength")
    first, last = (1, frame_count) if frame_range is None else frame_range
    if (not isinstance(first, (int, np.integer))
            or not isinstance(last, (int, np.integer))
            or not 1 <= first <= last <= frame_count):
        raise ValueError(f"frame_range must be within 1..{frame_count}, inclusive")
    if min_score is not None and not np.isfinite(min_score):
        raise ValueError("min_score must be finite or None")

    detection_path = Path("det/det.txt" if detection_path is None else detection_path)
    if not detection_path.is_absolute():
        detection_path = sequence_path / detection_path
    detections = _read_rows(detection_path, 7, frame_count, box_origin)

    frames = load_image_frames(sequence_path, (first, last))
    for frame_id, frame in frames.items():
        entries = detections.get(frame_id, [])
        if min_score is not None:
            entries = [(line, row) for line, row in entries if row[6] >= min_score]
        rows = np.asarray([row for _, row in entries], dtype=np.float64).reshape(-1, 7)
        frame.update({
            "boxes": rows[:, 2:6].astype(np.float32),
            "scores": rows[:, 6].astype(np.float32),
            "row_ids": np.asarray([line for line, _ in entries], dtype=np.int64),
        })
    return frames


def load_image_frames(sequence_path, frame_range=None):
    """Read image metadata only, for detection; never require det.txt or GT."""
    sequence_path = Path(sequence_path)
    info = _sequence_info(sequence_path)
    frame_count = info.getint("seqLength")
    first, last = (1, frame_count) if frame_range is None else frame_range
    if (not isinstance(first, (int, np.integer)) or not isinstance(last, (int, np.integer))
            or not 1 <= first <= last <= frame_count):
        raise ValueError(f"frame_range must be within 1..{frame_count}, inclusive")
    # Discover actual names, avoiding any four/six-digit padding assumption.
    image_paths = {}
    image_dir = sequence_path / info["imDir"]
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() == info["imExt"].lower():
            if not path.stem.isdigit():
                continue
            frame_id = int(path.stem)
            if frame_id in image_paths:
                raise ValueError(f"Duplicate image frame ID {frame_id} in {image_dir}")
            image_paths[frame_id] = path

    frames = {}
    for frame_id in range(first, last + 1):
        if frame_id not in image_paths:
            raise FileNotFoundError(f"Missing image for frame {frame_id} in {image_dir}")
        frames[frame_id] = {
            "frame_id": frame_id,
            "sequence": sequence_path.name,
            "image_path": image_paths[frame_id],
            "width": info.getint("imWidth"),
            "height": info.getint("imHeight"),
            "fps": info.getint("frameRate"),
        }
    return frames


def read_image(frame):
    """Read one frame as an RGB uint8 array [height, width, 3]."""
    with Image.open(frame["image_path"]) as image:
        if image.size != (frame["width"], frame["height"]):
            raise ValueError(f"Image dimensions disagree with seqinfo.ini: {frame['image_path']}")
        return np.array(image.convert("RGB"))


def load_ground_truth(sequence_path, gt_path="gt/gt.txt", *, box_origin=1):
    """Return GT records keyed by original frame ID, separate from detections.

    Each record has boxes, ids, classes, visibility, valid_mask, ignored_mask.
    For training, only marked class-1 pedestrians are valid. Conservatively
    treat all other annotations as ignored. This is a training policy, not a
    replacement for TrackEval's benchmark-specific evaluation preprocessing.
    """
    sequence_path = Path(sequence_path)
    frame_count = _sequence_info(sequence_path).getint("seqLength")
    gt_path = Path(gt_path)
    if not gt_path.is_absolute():
        gt_path = sequence_path / gt_path
    rows_by_frame = _read_rows(gt_path, 9, frame_count, box_origin)
    ground_truth = {}
    for frame_id in range(1, frame_count + 1):
        entries = rows_by_frame.get(frame_id, [])
        rows = np.asarray([row for _, row in entries], dtype=np.float64).reshape(-1, 9)
        for line, row in entries:
            if any(row[column] != int(row[column]) for column in (1, 6, 7)):
                raise ValueError(f"{gt_path}, line {line}: ID, mark and class must be integers")
            if row[6] not in (0, 1):
                raise ValueError(f"{gt_path}, line {line}: mark must be 0 or 1")
        valid = (rows[:, 6] == 1) & (rows[:, 7] == 1)
        ids = rows[:, 1].astype(np.int64)
        if np.any(ids[valid] <= 0) or len(np.unique(ids[valid])) != valid.sum():
            raise ValueError(f"{gt_path}, frame {frame_id}: valid person IDs must be positive and unique")
        ground_truth[frame_id] = {
            "boxes": rows[:, 2:6].astype(np.float32),
            "ids": ids,
            "classes": rows[:, 7].astype(np.int64),
            "visibility": rows[:, 8].astype(np.float32),
            "valid_mask": valid,
            "ignored_mask": ~valid,
        }
    return ground_truth


def pairwise_iou(boxes_a, boxes_b):
    """Return pairwise overlaps [A,B] for zero-origin xyxy boxes.

    Kept here so data loading works before graph.py exists; graph.py can import
    this same helper later. Zero-area boxes have zero overlap, including 0/0.
    """
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Boxes must contain finite coordinates")
    if np.any(a[:, 2:] < a[:, :2]) or np.any(b[:, 2:] < b[:, :2]):
        raise ValueError("Boxes must use ordered xyxy coordinates")
    intersection_size = np.maximum(
        np.minimum(a[:, None, 2:], b[None, :, 2:])
        - np.maximum(a[:, None, :2], b[None, :, :2]), 0,
    )
    intersection = intersection_size.prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_detections_to_gt(boxes, gt, iou_threshold=0.5):
    """Return ids/valid_mask/ignored_mask in the original detection row order.

    Match valid pedestrians one-to-one, maximizing the number of eligible
    matches first, then their total IoU. Unmatched detections get ID -1 and
    valid_mask=False: never use them as fabricated negative identities.
    ignored_mask identifies unmatched detections overlapping ignored GT.
    A valid pedestrian match takes priority over ignored-region overlap.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    overlap = pairwise_iou(boxes, gt["boxes"])
    ids = np.full(len(boxes), -1, dtype=np.int64)
    valid = gt["valid_mask"]
    person_overlap = overlap[:, valid]
    if len(boxes) and person_overlap.shape[1]:
        # Extra dummy columns allow every detection to stay unmatched.
        # The bonus ensures a larger valid matching beats fewer high-IoU pairs.
        costs = np.zeros((len(boxes), person_overlap.shape[1] + len(boxes)))
        bonus = min(person_overlap.shape) + 1
        costs[:, :person_overlap.shape[1]] = np.where(
            person_overlap >= iou_threshold, -(bonus + person_overlap), 1.0,
        )
        detection_rows, gt_columns = linear_sum_assignment(costs)
        matched = gt_columns < person_overlap.shape[1]
        detection_rows, gt_columns = detection_rows[matched], gt_columns[matched]
        ids[detection_rows] = gt["ids"][valid][gt_columns]
    valid_mask = ids != -1
    ignored_mask = (~valid_mask) & (overlap[:, gt["ignored_mask"]] >= iou_threshold).any(axis=1)
    return {"ids": ids, "valid_mask": valid_mask, "ignored_mask": ignored_mask}
