"""MOT17 frames and ignored regions for the authors' Iter-E2EDET mapper."""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from Utils import config
from Utils.data import load_ground_truth, load_image_frames
from Utils.detections import file_digest


class MOTIterDataset(Dataset):
    """Use marked pedestrians as targets and all other GT boxes as ignored regions."""

    def __init__(self, sequences, mapper, frame_stride=1):
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        from detectron2.structures import BoxMode

        self.mapper = mapper
        self.items = []
        self.sources = {}
        for name in sequences:
            path = Path(config.DATASET_ROOT)/config.DATASET_SPLIT/name
            frames = load_image_frames(path)
            ground_truth = load_ground_truth(path, config.GT_FILE, box_origin=1)
            selected = list(frames)[::frame_stride]
            self.sources[name] = {"gt_sha256": file_digest(path/config.GT_FILE),
                                  "frames": selected, "sequence_path": str(path.resolve())}
            for frame_id in selected:
                frame, gt = frames[frame_id], ground_truth[frame_id]
                annotations = []
                for box, valid in zip(gt["boxes"], gt["valid_mask"], strict=True):
                    x1, y1, x2, y2 = map(float, box)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    annotations.append({"bbox": [x1, y1, x2-x1, y2-y1],
                                        "bbox_mode": BoxMode.XYWH_ABS,
                                        "category_id": 0, "iscrowd": 0 if valid else 1})
                if not np.any(gt["valid_mask"]):
                    continue  # The upstream matcher expects at least one positive.
                self.items.append({"file_name": str(frame["image_path"]),
                                   "height": frame["height"], "width": frame["width"],
                                   "image_id": f"{name}:{frame_id}", "annotations": annotations})
        if not self.items:
            raise ValueError("No marked pedestrian training frames found")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.mapper(self.items[index])
