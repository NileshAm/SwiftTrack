"""MOT batches and ignored-region loss adapters for MMDetection 3.3 DDQ.

Keep upstream assignment, focal/box losses, dense auxiliaries and denoising.
Only unmatched classification targets overlapping ignored MOT boxes are masked.
See Documentation/Implementation/28-ddq-training.md.
"""

from types import MethodType

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps

from Utils.detector_training import clip_boxes


def enable_activation_checkpointing(model):
    """Recompute pure network blocks on backward to keep FP32 training within VRAM.

    Do not checkpoint DDQ's decoder as a whole: it mutates distinct-query masks.
    Only its attention/FFN blocks are pure. ResNet's norm_eval keeps BN stats fixed.
    """
    from torch.utils.checkpoint import checkpoint

    def wrap(module):
        forward = module.forward

        def checked(self, *args, **kwargs):
            if self.training and torch.is_grad_enabled():
                return checkpoint(forward, *args, use_reentrant=False, **kwargs)
            return forward(*args, **kwargs)

        module.forward = MethodType(checked, module)

    wrap(model.backbone)
    for layer in model.encoder.layers:
        wrap(layer)
    for layer in model.decoder.layers:
        for name in ('self_attn', 'cross_attn', 'ffn'):
            wrap(getattr(layer, name))


class DDQCollator:
    """Square resize and optional flip; preserve untouched GT for evaluation."""

    def __init__(self, image_size, person_id, *, augment=False):
        self.size, self.person_id, self.augment = image_size, person_id, augment

    def __call__(self, samples):
        inputs, data_samples, records = [], [], []
        for image, gt, frame in samples:
            h, w = image.shape[:2]
            boxes = clip_boxes(gt['boxes'][gt['valid_mask']], w, h)
            ignored = clip_boxes(gt['boxes'][gt['ignored_mask']], w, h)
            if self.augment and torch.rand(()).item() < .5:
                image = image[:, ::-1].copy()
                for b in (boxes, ignored):
                    b[:, [0, 2]] = w - b[:, [2, 0]]
            factor = np.asarray([self.size/w, self.size/h]*2, dtype=np.float32)
            boxes, ignored = boxes*factor, ignored*factor
            # DDQ's data preprocessor normalizes BGR and converts it to RGB.
            resized = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            inputs.append(torch.from_numpy(np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1))))
            sample = DetDataSample(metainfo=dict(
                img_id=frame['frame_id'], img_path=str(frame['image_path']),
                ori_shape=(h, w), img_shape=(self.size, self.size),
                scale_factor=(self.size/w, self.size/h), mot_ignored_boxes=ignored.tolist()))
            sample.gt_instances = InstanceData(
                bboxes=torch.as_tensor(boxes, dtype=torch.float32),
                labels=torch.full((len(boxes),), self.person_id, dtype=torch.long))
            data_samples.append(sample)
            records.append((gt, frame))
        return dict(inputs=inputs, data_samples=data_samples), records


@torch.no_grad()
def mask_ignored_weights(boxes, labels, weights, positives, meta, num_classes, ignore_iou):
    """Zero ignored negatives; keep matched positives and duplicate-person negatives."""
    ignored = boxes.new_tensor(meta.get('mot_ignored_boxes', [])).reshape(-1, 4)
    if not len(boxes) or not len(ignored):
        return
    masked = (bbox_overlaps(boxes.detach(), ignored) >= ignore_iou).any(dim=1)
    if len(positives):
        masked &= ~(bbox_overlaps(boxes.detach(), positives) >= ignore_iou).any(dim=1)
    masked &= labels == num_classes
    weights[masked] = 0


def image_boxes(predictions, meta):
    h, w = meta['img_shape']
    return bbox_cxcywh_to_xyxy(predictions) * predictions.new_tensor([w, h, w, h])


def install_mot_loss(model, ignore_iou):
    """Attach instance-local target adapters; no installed MMDetection files change.

    Distinct decoder and encoder targets share get_targets. Dense targets use
    DDQAuxLoss. Denoising predictions are available only during _loss_dn_single;
    its temporary context is cleared even when upstream raises an exception.
    """
    head = model.bbox_head
    original_targets = head._get_targets_single

    def targets(self, cls_score, bbox_pred, gt_instances, img_meta):
        result = original_targets(cls_score, bbox_pred, gt_instances, img_meta)
        mask_ignored_weights(image_boxes(bbox_pred, img_meta), result[0], result[1],
                             gt_instances.bboxes, img_meta, self.num_classes, ignore_iou)
        return result

    head._get_targets_single = MethodType(targets, head)
    dense = head.aux_loss_for_dense
    original_dense = dense._get_target_single

    def dense_targets(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_meta, **kwargs):
        result = original_dense(cls_scores, bbox_preds, gt_bboxes, gt_labels, img_meta, **kwargs)
        mask_ignored_weights(bbox_preds, result[0], result[1], gt_bboxes,
                             img_meta, cls_scores.shape[-1], ignore_iou)
        return result

    dense._get_target_single = MethodType(dense_targets, dense)
    original_dn_loss, original_dn_targets = head._loss_dn_single, head.get_dn_targets

    def dn_loss(self, dn_cls_scores, dn_bbox_preds, batch_gt_instances, batch_img_metas, dn_meta):
        self._mot_dn_boxes = dn_bbox_preds
        try:
            return original_dn_loss(dn_cls_scores, dn_bbox_preds, batch_gt_instances, batch_img_metas, dn_meta)
        finally:
            self._mot_dn_boxes = None

    def dn_targets(self, batch_gt_instances, batch_img_metas, dn_meta):
        result = original_dn_targets(batch_gt_instances, batch_img_metas, dn_meta)
        for i, (gt, meta) in enumerate(zip(batch_gt_instances, batch_img_metas, strict=True)):
            mask_ignored_weights(image_boxes(self._mot_dn_boxes[i], meta), result[0][i],
                                 result[1][i], gt.bboxes, meta, self.num_classes, ignore_iou)
        return result

    head._loss_dn_single = MethodType(dn_loss, head)
    head.get_dn_targets = MethodType(dn_targets, head)


def total_loss(losses):
    """MMDetection returns tensors or lists of per-image losses."""
    values = [sum(v.mean() for v in value) if isinstance(value, (tuple, list)) else value.mean()
              for key, value in losses.items() if 'loss' in key]
    loss = sum(values)
    if not values or not torch.isfinite(loss):
        raise FloatingPointError('DDQ produced an empty/nonfinite loss')
    return loss


@torch.no_grad()
def validation_forward(model, batch):
    """Eval-mode predictions plus distinct-decoder loss, without denoising or GT queries.

    DDQ's full training loss requires train-only dense queries. Compute decoder
    loss on the actual inference queries instead; never use train mode on val.
    """
    processed = model.data_preprocessor(batch, training=False)
    samples = processed['data_samples']
    head_inputs = model.forward_transformer(model.extract_feat(processed['inputs']), samples)
    outputs = model.bbox_head(**head_inputs)
    metas = [s.metainfo for s in samples]
    losses = model.bbox_head.loss_for_distinct_queries(
        *outputs, batch_gt_instances=[s.gt_instances for s in samples], batch_img_metas=metas)
    predictions = model.bbox_head.predict_by_feat(*outputs, batch_img_metas=metas, rescale=True)
    return total_loss(losses), predictions


def checkpoint_state(model):
    """Omit runtime aliases of shared DDQ heads, keeping portable strict-load weights."""
    return {key: value for key, value in model.state_dict().items()
            if not key.startswith('decoder.aux_reg_branches.')}


def load_training_state(model, state):
    # A previous training forward registers this alias. Remove it before strict
    # loading; DDQ restores it on the next training forward.
    if hasattr(model.decoder, 'aux_reg_branches'):
        del model.decoder.aux_reg_branches
    model.load_state_dict(state, strict=True)
