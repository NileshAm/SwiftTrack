"""Export official Iter-E2EDET CrowdHuman person boxes to full-video MOT caches.

Requires the authors' checkpoint and their vendored Detectron2 fork. Run from
the project root after installing ThirdParty/IterE2EDET in dragon-track.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from tqdm.auto import tqdm

from Utils import config
from Utils.data import _sequence_info, load_image_frames, read_image
from Utils.detections import file_digest
from Utils.splits import validate_video_splits

PROJECT = Path(__file__).resolve().parent
VENDOR = PROJECT / "ThirdParty/IterE2EDET"
SOURCE = VENDOR / "projects/crowd-e2e-sparse-rcnn"
UPSTREAM_COMMIT = "c649a0b29be48e7253048a2c8efd6b28a2e3197b"


def load_predictor(checkpoint, device, short_side, max_size):
    if not (VENDOR / "detectron2").is_dir():
        raise FileNotFoundError(f"Missing pinned Iter-E2EDET source: {VENDOR}")
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(SOURCE))
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from config import add_sparsercnn_config
        from models.detector import SparseRCNN  # noqa: F401: register the model
    except (ImportError, OSError) as error:
        raise RuntimeError(f"Could not import the vendored Iter-E2EDET detector: {error}") from error
    cfg = get_cfg()
    add_sparsercnn_config(cfg)
    yaml = SOURCE / "configs/50e.6h.500pro.ignore.yaml"
    cfg.merge_from_file(str(yaml))
    cfg.INPUT.MIN_SIZE_TEST = short_side
    cfg.INPUT.MAX_SIZE_TEST = max_size
    config_sha256 = hashlib.sha256(cfg.dump().encode()).hexdigest()
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return DefaultPredictor(cfg), config_sha256


def sequence_identity(path, recipe):
    info = _sequence_info(path)
    return {"sequence_path": str(path.resolve()), "frames": info.getint("seqLength"),
            "width": info.getint("imWidth"), "height": info.getint("imHeight"),
            "detector": recipe}


def export_sequence(path, cache, predictor, recipe):
    target = cache / path.parent.name / path.name / "det.txt"
    identity = sequence_identity(path, recipe)
    summary_path = target.with_name("summary.json")
    if target.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if (summary.get("complete") is True and summary.get("identity") == identity
                and summary.get("sha256") == file_digest(target)):
            tqdm.write(f"Verified existing cache: {target}")
            return target
        raise ValueError(f"Existing Iter-E2EDET cache differs: {target}; choose a new --cache-dir")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".partial")
    frames = load_image_frames(path)
    count = 0
    started = time.perf_counter()
    with pending.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for frame in tqdm(frames.values(), desc=path.name, unit="frame", leave=False):
            # Detectron2 DefaultPredictor expects BGR even when cfg.INPUT.FORMAT is RGB.
            bgr = np.ascontiguousarray(read_image(frame)[..., ::-1])
            instances = predictor(bgr)["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                    or (scores < 0).any() or (scores > 1).any()):
                raise ValueError(f"Nonfinite or invalid detector output: {frame['image_path']}")
            for (x1, y1, x2, y2), score, category in zip(boxes, scores, classes, strict=True):
                if int(category) != 0 or score < recipe["min_score"]:
                    continue
                x1, x2 = np.clip((x1, x2), 0, frame["width"])
                y1, y2 = np.clip((y1, y2), 0, frame["height"])
                if x2 <= x1 or y2 <= y1:
                    continue
                writer.writerow([frame["frame_id"], -1, f"{x1+1:.6f}", f"{y1+1:.6f}",
                                 f"{x2-x1:.6f}", f"{y2-y1:.6f}", f"{score:.8f}", -1, -1, -1])
                count += 1
    pending.replace(target)
    elapsed = time.perf_counter() - started
    summary = {"complete": True, "identity": identity, "sha256": file_digest(target),
               "mot_rows": count, "seconds": elapsed, "fps": len(frames) / elapsed}
    pending_summary = summary_path.with_suffix(".partial")
    pending_summary.write_text(json.dumps(summary, indent=2) + "\n")
    pending_summary.replace(summary_path)
    tqdm.write(f"{path.name}: {count} boxes -> {target}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Authors' released CrowdHuman model.pth")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--cache-dir", type=Path,
                        default=PROJECT / "detections/iter_e2edet_resolution_v2/higher_960_1600")
    parser.add_argument("--min-score", type=float, default=0.05,
                        help="Export floor; score sweeps cannot recover boxes below it")
    parser.add_argument("--short-side", type=int, default=960)
    parser.add_argument("--max-size", type=int, default=1600)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one validation frame on the selected device without writing a cache")
    args = parser.parse_args(argv)
    if not 0 <= args.min_score <= 1 or args.short_side < 32 or args.max_size < args.short_side:
        parser.error("Invalid score or image size")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Missing Iter-E2EDET checkpoint: {checkpoint}")
    validate_video_splits()
    names = ([] if args.split == "val" else list(config.TRAIN_SEQUENCES)) + (
        [] if args.split == "train" else list(config.VAL_SEQUENCES))
    paths = [config.DATASET_ROOT / config.DATASET_SPLIT / name for name in names]
    predictor, config_sha256 = load_predictor(checkpoint, args.device, args.short_side, args.max_size)
    if args.smoke_test:
        if args.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; GPU smoke test cannot run")
        frame = next(iter(load_image_frames(paths[0]).values()))
        bgr = np.ascontiguousarray(read_image(frame)[..., ::-1])
        started = time.perf_counter()
        instances = predictor(bgr)["instances"]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        print(json.dumps({"status": "ok", "sequence": paths[0].name,
                          "frame": frame["frame_id"], "short_side": args.short_side,
                          "max_size": args.max_size, "device": args.device,
                          "proposals": len(instances), "seconds": elapsed,
                          "config_sha256": config_sha256}))
        return
    recipe = {"format": "mot_person_v1", "backend": "iter_e2edet", "source_commit": UPSTREAM_COMMIT,
              "checkpoint_sha256": file_digest(checkpoint), "config_sha256": config_sha256,
              "short_side": args.short_side, "max_size": args.max_size,
              "min_score": args.min_score, "box_origin": 1, "person_class": 0,
              "postprocessing": "authors_detectron2_no_extra_nms"}
    cache = args.cache_dir.resolve()
    for path in tqdm(paths, desc="Iter-E2EDET export", unit="video"):
        export_sequence(path, cache, predictor, recipe)
    print(f"Iter-E2EDET cache: {cache}")
    return cache


if __name__ == "__main__":
    main()
