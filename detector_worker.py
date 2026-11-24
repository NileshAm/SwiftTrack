"""Isolated YOLO26/RF-DETR inference worker; invoked by detector_suite.py."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm
from Utils import config
from Utils.data import load_image_frames, read_image
from Utils.detections import file_digest


class ExternalDetector:
    def __init__(self, family, size, directory, image_size, floor):
        self.family, self.floor = family, floor
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if family == 'yolo26':
            from ultralytics import YOLO
            path = directory / f'yolo26{size}.pt'
            self.model = YOLO(str(path))
            classes = self.model.names
            package = 'ultralytics'
            self.resolution = image_size
            preprocessing = 'official_letterbox_square_rect_false'
        else:
            from rfdetr import RFDETRSmall, RFDETRMedium
            # RF-DETR's official downloader writes its default checkpoint to cwd.
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.model = (RFDETRSmall if size == 's' else RFDETRMedium)(device=config.DEVICE)
            finally:
                os.chdir(previous)
            path = directory / Path(self.model.model_config.pretrain_weights).name
            classes = self.model.class_names
            package = 'rfdetr'
            self.resolution = self.model.model.resolution
            preprocessing = 'official_rgb_normalize_square_native_resolution'
        items = classes.items() if isinstance(classes, dict) else enumerate(classes)
        people = [int(k) for k, v in items if str(v).lower() == 'person']
        if len(people) != 1:
            raise ValueError(f'Cannot resolve unique person class: {classes}')
        self.person_id = people[0]
        if not path.is_file():
            raise FileNotFoundError(f'Model loaded but weight file provenance is missing: {path}')
        self.recipe = dict(format='mot_person_v1', box_origin=1, backend=family,
                           variant=size, package=importlib.metadata.version(package),
                           weights_sha256=file_digest(path), min_score=floor,
                           image_size=self.resolution, preprocessing=preprocessing,
                           precision='float32', max_detections=300)

    @torch.inference_mode()
    def predict(self, image):
        if self.family == 'yolo26':
            # PIL avoids the BGR convention of Ultralytics NumPy inputs.
            result = self.model.predict(Image.fromarray(image), conf=self.floor,
                                        classes=[self.person_id], imgsz=self.resolution,
                                        rect=False, device=config.DEVICE, half=False,
                                        max_det=300, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
        else:
            result = self.model.predict(Image.fromarray(image), threshold=self.floor)
            keep = result.class_id == self.person_id
            boxes, scores = result.xyxy[keep], result.confidence[keep]
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32)
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('Nonfinite detector predictions')
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, image.shape[1])
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, image.shape[0])
        valid = (boxes[:, 2:] - boxes[:, :2] > 1e-5).all(1)
        return boxes[valid], scores[valid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    for key, value in job['settings'].items():
        setattr(config, key, Path(value) if key in ('DATASET_ROOT', 'TEST_DATASET_ROOT') else value)
    config.DETECTION_BACKEND = 'cached'
    config.DETECTION_CACHE_DIR = Path(job['cache'])
    config.DETECTOR_MIN_SCORE = job['floor']
    detector = ExternalDetector(job['family'], job['size'], job['weights'], job['image_size'], job['floor'])
    config.CACHED_DETECTOR_RECIPE = detector.recipe
    if args.smoke:
        folder = config.DATASET_ROOT / config.DATASET_SPLIT / config.VAL_SEQUENCES[0]
        frame = next(iter(load_image_frames(folder).values()))
        boxes, scores = detector.predict(read_image(frame))
        print(f'Smoke PASS: {job["family"]}/{job["size"]}: {len(boxes)} person boxes', flush=True)
        return
    from detect import run_sequence
    from Utils.detections import validate_cache, cache_path
    sequences = list(dict.fromkeys(config.TRAIN_SEQUENCES + config.VAL_SEQUENCES))
    for name in sequences:
        folder = config.DATASET_ROOT / config.DATASET_SPLIT / name
        if cache_path(folder).with_name('summary.json').is_file():
            validate_cache(folder)
        else:
            run_sequence(folder, detector)
    Path(job['recipe']).write_text(json.dumps(detector.recipe, indent=2) + '\n')


if __name__ == '__main__':
    main()
