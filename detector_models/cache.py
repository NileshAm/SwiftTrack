"""Atomic MOT-format detection cache writer with provenance and restart checks."""

import csv
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from Utils.data import _sequence_info, load_image_frames, read_image
from Utils.detections import file_digest


def identity(sequence_path, recipe):
    info = _sequence_info(sequence_path)
    return {"sequence_path": str(Path(sequence_path).resolve()),
            "frames": info.getint("seqLength"), "width": info.getint("imWidth"),
            "height": info.getint("imHeight"), "detector": recipe}


def cache_path(root, sequence_path):
    sequence_path = Path(sequence_path)
    return Path(root) / sequence_path.parent.name / sequence_path.name / "det.txt"


def validate(path, expected):
    path = Path(path)
    summary = path.with_name("summary.json")
    if not path.is_file() or not summary.is_file():
        return False
    saved = json.loads(summary.read_text())
    if (saved.get("complete") is not True or saved.get("identity") != expected
            or saved.get("sha256") != file_digest(path)):
        raise ValueError(f"Changed or incompatible cache: {path}. Use a new output root.")
    return True


def preview(image, boxes, scores, path):
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    for box, score in zip(boxes, scores, strict=True):
        draw.rectangle(tuple(map(float, box)), outline="lime", width=2)
        draw.text(tuple(map(float, box[:2])), f"person {score:.2f}", fill="lime")
    picture.save(path)


def export_sequence(sequence_path, destination, detector, overall):
    """Export all frames. A partial video is replaced; a verified complete video is skipped."""
    destination = Path(destination)
    expected = identity(sequence_path, detector.recipe)
    if validate(destination, expected):
        tqdm.write(f"Skipping verified cache: {sequence_path.name}")
        overall.update(expected["frames"])
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    frames = load_image_frames(sequence_path)
    start, rows = time.perf_counter(), 0
    with partial.open("w", newline="") as stream, tqdm(
            frames.values(), desc=sequence_path.name, unit="frame", position=1, leave=False) as bar:
        writer = csv.writer(stream)
        for frame in bar:
            image = read_image(frame)
            boxes, scores = detector.predict(image)
            for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
                writer.writerow([frame["frame_id"], -1, f"{x1+1:.6f}", f"{y1+1:.6f}",
                                 f"{x2-x1:.6f}", f"{y2-y1:.6f}", f"{score:.8f}", -1, -1, -1])
            if frame["frame_id"] == 1:
                preview(image, boxes, scores, destination.with_name("preview.jpg"))
            rows += len(boxes)
            overall.update(1)
    partial.replace(destination)
    seconds = time.perf_counter() - start
    summary = {"complete": True, "identity": expected, "sha256": file_digest(destination),
               "mot_rows": rows, "seconds": seconds, "fps": len(frames) / seconds,
               "timing": "image read + preprocessing + GPU inference + MOT export"}
    pending = destination.with_name("summary.json.partial")
    pending.write_text(json.dumps(summary, indent=2) + "\n")
    pending.replace(destination.with_name("summary.json"))

