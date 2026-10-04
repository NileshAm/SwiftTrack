"""DETR full-image transformer features, aligned to supplied person boxes.

Read Documentation/Implementation/11-paper-components.md for the scope: this
uses DETR's contextual encoder, not its detector predictions or query embeddings.
The configured supplied/cached detector boxes determine which people can be tracked.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from torch.utils.checkpoint import checkpoint

from Utils import config


class DETRAppearance(nn.Module):
    """Run a pretrained ResNet50/DETR encoder once per image, then pool each ROI.

    FREEZE_ENCODER controls fine-tuning. Current-frame association loss can
    update these weights; cached track histories remain detached.
    Dependencies are pinned to transformers 4.49.0 and timm 1.0.15.
    """

    def __init__(self, device):
        super().__init__()
        from transformers import DetrImageProcessor, DetrModel

        if not config.USE_PRETRAINED_WEIGHTS:
            raise ValueError("DETR appearance requires pretrained weights")
        if config.FEATURE_DIM != 256:
            raise ValueError("DETR ResNet50 requires FEATURE_DIM=256")
        if not 32 <= config.DETR_SHORT_SIDE <= config.DETR_MAX_SIZE:
            raise ValueError("Require 32 <= DETR_SHORT_SIDE <= DETR_MAX_SIZE")
        self.processor = DetrImageProcessor.from_pretrained(
            config.DETR_MODEL_NAME, revision=config.DETR_REVISION,
            size={"shortest_edge": config.DETR_SHORT_SIDE, "longest_edge": config.DETR_MAX_SIZE})
        # All backbone weights come from DETR; avoid an extra timm weight download.
        model = DetrModel.from_pretrained(config.DETR_MODEL_NAME,
                                         revision=config.DETR_REVISION,
                                         use_pretrained_backbone=False)
        self.backbone = model.backbone
        self.input_projection = model.input_projection
        self.transformer = model.encoder
        self.feature_dim = model.config.d_model
        self.requires_grad_(not config.FREEZE_ENCODER)
        self.eval().to(device)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.feature_settings = {
            "encoder": "detr_resnet50", "model": config.DETR_MODEL_NAME,
            "revision": config.DETR_REVISION, "feature_dim": self.feature_dim,
            "shortest_edge": config.DETR_SHORT_SIDE, "longest_edge": config.DETR_MAX_SIZE,
            "pooling": "encoder_roi_align_3x3_aligned_mean_l2",
            "color": "RGB", "mean": self.processor.image_mean,
            "std": self.processor.image_std, "frozen": config.FREEZE_ENCODER,
            "precision": "float32_tf32_disabled", "transformers": "4.49.0",
        }

    def extract_boxes(self, image, boxes):
        """Single-image interface; uses the same pooling as batched extraction."""
        return self.extract_batch([image], [boxes])[0]

    def extract_batch(self, images, boxes_per_image):
        """Encode same-size frames together; return ordered (features, mask) pairs.

        Images are independent along the batch dimension; no temporal attention
        is added. Only frames with valid boxes enter DETR. Keeping equal image
        sizes avoids padding-dependent changes to positional features/ROI scale.
        Outputs share an autograd graph: sum their losses before backward().
        """
        if len(images) != len(boxes_per_image):
            raise ValueError("Every image needs its own box array")
        results, active_images, regions_per_image = [], [], []
        active_rows = []
        shape = None
        for index, (image, boxes) in enumerate(zip(images, boxes_per_image)):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0):
                raise ValueError("Images must be nonempty RGB uint8 [H,W,3] arrays")
            if shape is not None and image.shape != shape:
                raise ValueError("Batch frames must share image dimensions; batch videos separately")
            shape = image.shape
            boxes = np.asarray(boxes, dtype=np.float32)
            if boxes.shape == (0,):
                boxes = boxes.reshape(0, 4)
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("Boxes must be [D,4] zero-origin xyxy")
            height, width = image.shape[:2]
            clipped = boxes.copy()
            clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, width)
            clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, height)
            valid = (np.isfinite(boxes).all(1) & (clipped[:, 2] > clipped[:, 0])
                     & (clipped[:, 3] > clipped[:, 1]))
            results.append((torch.zeros(len(boxes), self.feature_dim), torch.from_numpy(valid)))
            if valid.any():
                active_rows.append(index)
                active_images.append(image)
                regions_per_image.append(clipped[valid])
        if not active_images:
            return results
        device = next(self.parameters()).device
        inputs = self.processor(images=active_images, return_tensors="pt").to(device)
        checkpointing = self.training and config.DETR_GRADIENT_CHECKPOINTING and torch.is_grad_enabled()
        maps, positions = (checkpoint(self.backbone, inputs.pixel_values, inputs.pixel_mask, use_reentrant=False)
                           if checkpointing else self.backbone(inputs.pixel_values, inputs.pixel_mask))
        last_map, mask = maps[-1]
        projected = self.input_projection(last_map)
        grid_h, grid_w = projected.shape[-2:]
        arguments = dict(
            inputs_embeds=projected.flatten(2).transpose(1, 2),
            attention_mask=mask.flatten(1),
            object_queries=positions[-1].flatten(2).transpose(1, 2),
            return_dict=True)
        encoded = (checkpoint(self.transformer, use_reentrant=False, **arguments)
                   if checkpointing else self.transformer(**arguments)).last_hidden_state
        memory = encoded.transpose(1, 2).reshape(len(active_images), self.feature_dim, grid_h, grid_w)
        regions = [torch.as_tensor(boxes, device=device) for boxes in regions_per_image]
        scale = memory.new_tensor([grid_w/width, grid_h/height]*2)
        regions = [boxes * scale for boxes in regions]
        pooled = roi_align(memory, regions, output_size=(3, 3),
                           spatial_scale=1.0, sampling_ratio=2, aligned=True).mean((2, 3))
        pooled = F.normalize(pooled, dim=1)
        if not torch.isfinite(pooled).all():
            raise RuntimeError("Nonfinite DETR appearance features")
        for index, features in zip(active_rows, pooled.cpu().split([len(x) for x in regions])):
            result, valid = results[index]
            result[valid] = features
        return results
