"""Frozen DETR full-image transformer features, aligned to supplied person boxes.

Read Documentation/Implementation/11-paper-components.md for the scope: this
uses DETR's contextual encoder, not its detector predictions or query embeddings.
The configured supplied/cached detector boxes determine which people can be tracked.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from Utils import config


class DETRAppearance(nn.Module):
    """Run a pretrained ResNet50/DETR encoder once per image, then pool each ROI.

    The encoder stays frozen to fit the current one-graph training pipeline.
    The graph projection, affinities, attention, and matching remain trainable.
    Dependencies are pinned to transformers 4.49.0 and timm 1.0.15.
    """

    def __init__(self, device):
        super().__init__()
        from transformers import DetrImageProcessor, DetrModel

        if not config.FREEZE_ENCODER:
            raise ValueError("DETR is frozen; end-to-end detector fine-tuning is not implemented")
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
        self.requires_grad_(False)
        self.eval().to(device)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.feature_settings = {
            "encoder": "detr_resnet50", "model": config.DETR_MODEL_NAME,
            "revision": config.DETR_REVISION, "feature_dim": self.feature_dim,
            "shortest_edge": config.DETR_SHORT_SIDE, "longest_edge": config.DETR_MAX_SIZE,
            "pooling": "encoder_roi_align_3x3_aligned_mean_l2",
            "color": "RGB", "mean": self.processor.image_mean,
            "std": self.processor.image_std, "frozen": True,
            "precision": "float32_tf32_disabled", "transformers": "4.49.0",
        }

    @torch.no_grad()
    def extract_boxes(self, image, boxes):
        """Return CPU [D,256] features and [D] validity; retain input row order.

        Transformer attention sees the whole image. Invalid/clipped-away boxes
        receive zero features and False; callers must apply the returned mask.
        ROI coordinates are scaled to the actual backbone grid, not guessed.
        """
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0):
            raise ValueError("image must be a nonempty RGB uint8 [H,W,3] array")
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.shape == (0,):
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must be [D,4] zero-origin xyxy")
        height, width = image.shape[:2]
        clipped = boxes.copy()
        clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, width)
        clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, height)
        valid = (np.isfinite(boxes).all(1) & (clipped[:, 2] > clipped[:, 0])
                 & (clipped[:, 3] > clipped[:, 1]))
        result = torch.zeros(len(boxes), self.feature_dim)
        valid_mask = torch.from_numpy(valid)
        if not valid.any():
            return result, valid_mask
        self.eval()
        device = next(self.parameters()).device
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        maps, positions = self.backbone(inputs.pixel_values, inputs.pixel_mask)
        last_map, mask = maps[-1]
        projected = self.input_projection(last_map)
        grid_h, grid_w = projected.shape[-2:]
        encoded = self.transformer(
            inputs_embeds=projected.flatten(2).transpose(1, 2),
            attention_mask=mask.flatten(1),
            object_queries=positions[-1].flatten(2).transpose(1, 2),
            return_dict=True).last_hidden_state
        memory = encoded.transpose(1, 2).reshape(1, self.feature_dim, grid_h, grid_w)
        regions = torch.as_tensor(clipped[valid], device=device)
        regions = regions * regions.new_tensor([grid_w/width, grid_h/height]*2)
        pooled = roi_align(memory, [regions], output_size=(3, 3),
                           spatial_scale=1.0, sampling_ratio=2, aligned=True).mean((2, 3))
        pooled = F.normalize(pooled, dim=1)
        if not torch.isfinite(pooled).all():
            raise RuntimeError("Nonfinite DETR appearance features")
        result[valid_mask] = pooled.cpu()
        return result, valid_mask
