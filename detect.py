"""Cache RT-DETRv2-S person boxes: python detect.py [--split all|train|val|test].

This is pretrained inference, not detector fine-tuning. No ground truth or
supplied detections are read. Training/tracking later read the saved MOT files.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm.auto import tqdm

from Utils import config
from Utils.data import load_image_frames, read_image
from Utils.detections import cache_path, cache_identity, detector_settings, file_digest, validate_cache
from Utils.splits import validate_video_splits, video_key


class PersonDetector:
    """Frozen RT-DETRv2-S, RGB square resize, original-size xyxy person boxes."""

    def __init__(self):
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        detector_settings()
        self.device = torch.device(config.DEVICE)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Fix GPU access or set DEVICE='cpu' for a small check.")
        self.processor = RTDetrImageProcessor.from_pretrained(
            config.DETECTOR_MODEL_NAME, revision=config.DETECTOR_REVISION,
            size={"height": config.DETECTOR_IMAGE_SIZE, "width": config.DETECTOR_IMAGE_SIZE})
        self.model = RTDetrV2ForObjectDetection.from_pretrained(
            config.DETECTOR_MODEL_NAME, revision=config.DETECTOR_REVISION,
            use_pretrained_backbone=False).eval().requires_grad_(False).to(self.device)
        # Resolve by name: COCO class indices differ between model implementations.
        person_ids = [int(key) for key, value in self.model.config.id2label.items() if value.lower() == "person"]
        if len(person_ids) != 1:
            raise ValueError("Detector must contain exactly one class named person")
        self.person_id = person_ids[0]
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    @torch.inference_mode()
    def predict(self, image):
        """Return clipped [N,4] xyxy boxes and [N] scores; no IDs or GT."""
        height, width = image.shape[:2]
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[height, width]], device=self.device),
            threshold=config.DETECTOR_MIN_SCORE, use_focal_loss=self.model.config.use_focal_loss)[0]
        selected = result["labels"] == self.person_id
        boxes = result["boxes"][selected].cpu().numpy()
        scores = result["scores"][selected].cpu().numpy()
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise RuntimeError("Detector produced nonfinite boxes/scores")
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
        # Avoid widths/heights rounding to zero in the six-decimal MOT export.
        valid = ((boxes[:, 2] - boxes[:, 0] > 1e-5) & (boxes[:, 3] - boxes[:, 1] > 1e-5))
        return boxes[valid], scores[valid]


def run_sequence(sequence_path, detector, *, overall=None):
    """Save a full video atomically; restart incomplete videos, skip complete ones.

    det.txt contains frame,-1,left,top,width,height,score,-1,-1,-1. Coordinates
    are MOT one-origin xywh. summary.json is published only after all frames;
    its recipe/coverage/hash are checked before training can consume the file.
    """
    sequence_path = Path(sequence_path)
    identity = cache_identity(sequence_path)
    path = cache_path(sequence_path)
    summary_path = path.with_name("summary.json")
    if summary_path.exists():
        validate_cache(sequence_path)
        tqdm.write(f"Already complete: {path}")
        return path
    frames = load_image_frames(sequence_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name("det.txt.partial")
    row_count = 0
    start = time.perf_counter()
    with partial.open("w", newline="") as stream, tqdm(
            frames.values(), total=len(frames), desc=sequence_path.name, unit="frame",
            position=1 if overall is not None else 0, leave=overall is None,
            dynamic_ncols=True, disable=not config.SHOW_PROGRESS) as bar:
        writer = csv.writer(stream)
        for frame in bar:
            image = read_image(frame)
            boxes, scores = detector.predict(image)
            for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
                writer.writerow([frame["frame_id"], -1, f"{x1+1:.6f}", f"{y1+1:.6f}",
                                 f"{x2-x1:.6f}", f"{y2-y1:.6f}", f"{score:.8f}", -1, -1, -1])
            row_count += len(boxes)
            if frame["frame_id"] == 1:
                preview = Image.fromarray(image)
                draw = ImageDraw.Draw(preview)
                for box, score in zip(boxes, scores, strict=True):
                    draw.rectangle(box.tolist(), outline="lime", width=2)
                    draw.text((float(box[0]), float(box[1])), f"person {score:.2f}", fill="lime")
                preview.save(path.with_name("preview.jpg"))
            bar.set_postfix(people=len(boxes), saved=row_count, refresh=False)
            if overall is not None:
                overall.update(1)
    seconds = time.perf_counter() - start
    partial.replace(path)
    import transformers
    summary = {"complete": True, "identity": identity, "mot_rows": row_count,
               "sha256": file_digest(path), "seconds": seconds, "fps": len(frames)/seconds,
               "device": str(config.DEVICE), "torch": torch.__version__,
               "transformers": transformers.__version__, "person_class_id": detector.person_id,
               "timing": "image reads, inference, export and first-frame preview; excludes model loading"}
    pending_summary = summary_path.with_suffix(".json.partial")
    pending_summary.write_text(json.dumps(summary, indent=2) + "\n")
    pending_summary.replace(summary_path)
    tqdm.write(f"Saved {row_count} person boxes across {len(frames)} frames: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    args = parser.parse_args()
    validate_video_splits()
    if config.DETECTION_BACKEND in ("supplied", "tracktor"):
        tqdm.write("Using supplied/imported detections; detector inference is skipped.")
        return
    detector_settings()
    if config.DETECTION_BACKEND != "rtdetrv2":
        raise ValueError("Set DETECTION_BACKEND='rtdetrv2' before running detect.py")
    groups = {
        "train": (config.DATASET_ROOT, config.DATASET_SPLIT, config.TRAIN_SEQUENCES),
        "val": (config.DATASET_ROOT, config.DATASET_SPLIT, config.VAL_SEQUENCES),
        "test": (config.TEST_DATASET_ROOT, config.TEST_DATASET_SPLIT, config.TEST_SEQUENCES),
    }
    jobs, seen, destinations = [], set(), set()
    for split, (root, dataset_split, sequences) in groups.items():
        if args.split not in ("all", split):
            continue
        for sequence in sequences:
            video_key(sequence)  # Validate folder names before forming output paths.
            path = Path(root) / dataset_split / sequence
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            destination = cache_path(path).resolve()
            if destination in destinations:
                raise ValueError("Different datasets would share a detection cache path; use separate caches")
            destinations.add(destination)
            identity = cache_identity(path)
            if cache_path(path).with_name("summary.json").exists():
                validate_cache(path)
                tqdm.write(f"Skipping complete cache: {sequence}")
            else:
                jobs.append((split, path, identity["frames"]))
    if not jobs:
        tqdm.write("No pending videos. All selected caches are complete (or no videos were selected).")
        return
    tqdm.write(f"Loading {config.DETECTOR_MODEL_NAME} on {config.DEVICE}; {len(jobs)} videos pending.")
    detector = PersonDetector()
    with tqdm(total=sum(count for _, _, count in jobs), desc="Detection overall", unit="frame",
              position=0, smoothing=0, dynamic_ncols=True, disable=not config.SHOW_PROGRESS) as overall:
        for index, (split, path, _) in enumerate(jobs, 1):
            overall.set_postfix(video=f"{index}/{len(jobs)}", split=split, sequence=path.name, refresh=False)
            run_sequence(path, detector, overall=overall)
    tqdm.write("Detection caches ready. Run python train.py to train the association model.")


if __name__ == "__main__":
    main()
