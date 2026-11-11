"""MOT targets and RT-DETR loss with ignored annotations, used by detector-train.py.

Loss adapter follows transformers 4.49 loss_rt_detr.py (Apache-2.0,
Copyright 2020 The HuggingFace Team). Upstream matching, box losses, auxiliary
supervision and denoising are retained; only ignored-region negative VFL is masked.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.image_transforms import center_to_corners_format
from transformers.loss.loss_rt_detr import RTDetrLoss
from transformers.loss.loss_for_object_detection import box_iou

from Utils import config
from Utils.data import load_ground_truth, load_image_frames, read_image
from Utils.detections import file_digest
from Utils.splits import validate_video_splits, video_key


class MOTDetectionDataset(Dataset):
    """Read RGB frames and GT only; never read supplied or generated detections."""

    def __init__(self, sequences, *, stride=1, max_frames=None, fraction=1.):
        if not 0 < fraction <= 1:
            raise ValueError("Training fraction must be in (0, 1]")
        self.items, self.sources = [], {}
        if len({video_key(name) for name in sequences}) != len(sequences):
            raise ValueError("Do not count detector variants of one video as separate training videos")
        for name in sequences:
            path = Path(config.DATASET_ROOT) / config.DATASET_SPLIT / name
            gt_path = path / config.GT_FILE
            truth = load_ground_truth(path, config.GT_FILE, box_origin=config.BOX_ORIGIN)
            frames = load_image_frames(path)
            selected = list(frames)[::stride]
            # Keep frames across the whole video, not just its opening section.
            # Fixed frame IDs make every trial and resumed epoch comparable.
            if selected and fraction < 1:
                count = max(1, int(len(selected) * fraction))
                indices = (np.linspace(0, len(selected)-1, count, dtype=int)
                           if count > 1 else [len(selected)//2])
                selected = [selected[i] for i in indices]
            if max_frames is not None:
                selected = selected[:max_frames]
            self.sources[name] = {"sequence_path": str(path.resolve()), "gt_sha256": file_digest(gt_path),
                                  "seqinfo_sha256": file_digest(path/'seqinfo.ini'), "frames": selected}
            self.items.extend((frames[i], truth[i]) for i in selected)
        if not self.items:
            raise ValueError("No detector training/validation frames selected")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        frame, truth = self.items[index]
        return read_image(frame), truth, frame


def clip_boxes(boxes, width, height):
    boxes = boxes.copy()
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
    return boxes[(boxes[:, 2:] > boxes[:, :2]).all(axis=1)]


class DetectionCollator:
    """Convert marked pedestrians to COCO targets and ignored boxes to normalized cxcywh.

    No GT visibility cutoff: occluded marked pedestrians remain training targets.
    An optional horizontal flip transforms positive and ignored boxes together.
    Evaluation receives the original unmodified GT and original image dimensions.
    """

    def __init__(self, processor, person_id, *, augment=False):
        self.processor, self.person_id, self.augment = processor, person_id, augment

    def __call__(self, samples):
        images, annotations, ignored, records = [], [], [], []
        for image, gt, frame in samples:
            h, w = image.shape[:2]
            boxes = clip_boxes(gt['boxes'][gt['valid_mask']], w, h)
            ignore = clip_boxes(gt['boxes'][gt['ignored_mask']], w, h)
            if self.augment and torch.rand(()).item() < .5:
                image = image[:, ::-1].copy()
                for b in (boxes, ignore):
                    b[:, [0, 2]] = w - b[:, [2, 0]]
            objects = []
            for x1, y1, x2, y2 in boxes:
                objects.append({'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                'category_id': self.person_id, 'area': float((x2-x1)*(y2-y1)), 'iscrowd': 0})
            annotations.append({'image_id': frame['frame_id'], 'annotations': objects})
            normalized = ignore / np.asarray([w, h, w, h], dtype=np.float32)
            normalized[:, :2] = (normalized[:, :2] + normalized[:, 2:]) / 2
            normalized[:, 2:] = ignore[:, 2:] / [w, h] - ignore[:, :2] / [w, h]
            ignored.append(torch.as_tensor(normalized, dtype=torch.float32))
            images.append(image)
            records.append((gt, frame))
        batch = self.processor(images=images, annotations=annotations, return_tensors='pt')
        for target, ignored_boxes in zip(batch['labels'], ignored, strict=True):
            target['ignored_boxes'] = ignored_boxes
        return batch, records


class MOTRTDetrLoss(RTDetrLoss):
    """Prevent unmatched ignored regions being trained as background negatives."""

    def __init__(self, model_config, ignore_iou):
        super().__init__(model_config)
        self.ignore_iou = ignore_iou

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        mask = torch.zeros(outputs['logits'].shape[:2], dtype=torch.bool, device=outputs['logits'].device)
        with torch.no_grad():
            predicted = center_to_corners_format(outputs['pred_boxes'])
            for b, (target, (matched, _)) in enumerate(zip(targets, indices, strict=True)):
                ignore = target.get('ignored_boxes')
                if ignore is None or not len(ignore):
                    continue
                overlap, _ = box_iou(predicted[b], center_to_corners_format(ignore))
                mask[b] = (overlap >= self.ignore_iou).any(dim=1)
                # Preserve positive supervision and duplicate-person negatives.
                if len(target['boxes']):
                    positives, _ = box_iou(predicted[b], center_to_corners_format(target['boxes']))
                    mask[b] &= ~(positives >= self.ignore_iou).any(dim=1)
                mask[b, matched] = False
        # For ignored negatives, sigmoid(-100) and its VFL weight are effectively zero.
        # masked_fill also disconnects their logits from the gradient graph.
        changed = {**outputs, 'logits': outputs['logits'].masked_fill(mask[..., None], -100.)}
        return super().loss_labels_vfl(changed, targets, indices, num_boxes, log)


def install_mot_loss(model, ignore_iou):
    """Attach the official loss packaging with our criterion, without patching Transformers."""
    criterion = MOTRTDetrLoss(model.config, ignore_iou).to(model.device)

    def loss_function(logits, labels, device, pred_boxes, config, outputs_class=None, outputs_coord=None,
                      enc_topk_logits=None, enc_topk_bboxes=None, denoising_meta_values=None, **kwargs):
        outputs = {'logits': logits, 'pred_boxes': pred_boxes}
        auxiliary = None
        if config.auxiliary_loss:
            if denoising_meta_values is not None:
                dn_boxes, outputs_coord = torch.split(outputs_coord, denoising_meta_values['dn_num_split'], dim=2)
                dn_logits, outputs_class = torch.split(outputs_class, denoising_meta_values['dn_num_split'], dim=2)
            auxiliary = [{'logits': c, 'pred_boxes': b} for c, b in
                         zip(outputs_class[:, :-1].unbind(1), outputs_coord[:, :-1].unbind(1), strict=True)]
            auxiliary.append({'logits': enc_topk_logits, 'pred_boxes': enc_topk_bboxes})
            outputs['auxiliary_outputs'] = auxiliary
            if denoising_meta_values is not None:
                outputs['dn_auxiliary_outputs'] = [{'logits': c, 'pred_boxes': b} for c, b in
                                                   zip(dn_logits.unbind(1), dn_boxes.unbind(1), strict=True)]
                outputs['denoising_meta_values'] = denoising_meta_values
        losses = criterion(outputs, labels)
        return sum(losses.values()), losses, auxiliary

    model.loss_function = loss_function


def move_batch(batch, device):
    return {key: ([{k: v.to(device) for k, v in t.items()} for t in value] if key == 'labels'
                  else value.to(device)) for key, value in batch.items()}


def build_datasets(stride, max_frames, *, train_fraction=1.):
    """Sample each training video across time, leaving validation full.

    Stride precedes fraction sampling. max_frames remains a separate smoke-check
    cap on both splits, applied after sampling.
    """
    validate_video_splits()
    return (MOTDetectionDataset(config.TRAIN_SEQUENCES, stride=stride, max_frames=max_frames,
                               fraction=train_fraction),
            MOTDetectionDataset(config.VAL_SEQUENCES, max_frames=max_frames))
