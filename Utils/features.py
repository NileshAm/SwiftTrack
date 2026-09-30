"""Appearance features for supplied boxes: DETR by default, optional DenseNet.

Load the encoder once, then reuse it for every frame. Extraction returns
(features, valid_mask) in detection order; both tensors are on the CPU.
"""

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

from Utils import config
from tqdm.auto import tqdm


def load_encoder(device=None):
    """Load the configured frozen encoder (DETR by default).

    Uses config.DEVICE unless device is given (for example, "cpu"). The first
    pretrained run downloads official weights. DETR details are in appearance.py;
    the optional DenseNet path below remains available for explicit comparisons.
    """
    if config.ENCODER_NAME == "detr_resnet50":
        from Utils.appearance import DETRAppearance
        return DETRAppearance(config.DEVICE if device is None else device)
    if config.ENCODER_NAME != "densenet121":
        raise ValueError("ENCODER_NAME must be detr_resnet50 or densenet121")
    if not config.FREEZE_ENCODER:
        raise ValueError("This extractor is frozen; fine-tuning is not implemented")
    device = torch.device(config.DEVICE if device is None else device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver or use load_encoder('cpu').")
    crop_size = tuple(config.CROP_SIZE)
    if len(crop_size) != 2 or any(not isinstance(n, int) or n < 32 for n in crop_size):
        raise ValueError("CROP_SIZE must be (height, width), each at least 32 pixels")

    # Use full FP32 for the initial correctness baseline. CUDA convolution can
    # otherwise use TF32 even with float32 tensors, changing the embeddings.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # Pin the weight version rather than allowing DEFAULT to change later.
    weights = DenseNet121_Weights.IMAGENET1K_V1
    encoder = densenet121(weights=weights if config.USE_PRETRAINED_WEIGHTS else None)
    encoder.feature_dim = encoder.classifier.in_features
    if config.FEATURE_DIM != encoder.feature_dim:
        raise ValueError(f"FEATURE_DIM must be {encoder.feature_dim} for DenseNet121")
    # DenseNet.forward already applies ReLU, spatial average pooling, and flatten.
    encoder.classifier = nn.Identity()
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)

    # Resize the entire person crop to our configured H/W, without a center crop.
    # Keep the pretrained RGB normalization. This resize is our chosen protocol.
    normalization = weights.transforms()
    encoder.preprocess = transforms.Compose([
        transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization.mean, std=normalization.std),
    ])
    encoder.feature_settings = {
        "encoder": config.ENCODER_NAME,
        "weights": weights.name if config.USE_PRETRAINED_WEIGHTS else "random_debug_only",
        "crop_size": crop_size,
        "color": "RGB",
        "mean": normalization.mean,
        "std": normalization.std,
        "resize": "bilinear_whole_crop",
        "pooling": "relu_global_average",
        "feature_dim": encoder.feature_dim,
        "frozen": True,
        "precision": "float32_tf32_disabled",
    }
    return encoder


@torch.no_grad()
def extract_features(image, boxes, encoder, batch_size=None, *, show_progress=True):
    """Return (features [D,F], valid_mask [D]) as CPU tensors.

    DETR dispatches to appearance.py: whole-frame transformer + ROI pooling.
    The crop/batch details below describe the optional DenseNet fallback only.

    image: RGB uint8 NumPy array, as returned by Utils.data.read_image.
    boxes: zero-origin xyxy array [D,4], as returned by load_sequence.
    encoder: the reusable model returned by load_encoder.

    Fractional boxes use floor(left/top), ceil(right/bottom) after clipping.
    Invalid, nonfinite, or fully outside boxes get zero features and False in
    valid_mask. Always use that mask before graph construction or matching.
    Original boxes and row order are preserved. Only one crop batch is on GPU
    at a time; outputs are normal detached tensors usable by a trainable graph.
    Set show_progress=False inside a loop with its own frame progress bar.
    """
    if hasattr(encoder, "extract_boxes"):
        return encoder.extract_boxes(image, boxes)
    batch_size = config.CROP_BATCH_SIZE if batch_size is None else batch_size
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0):
        raise ValueError("image must be a nonempty RGB uint8 array [height, width, 3]")
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [D,4] in xyxy order")

    height, width = image.shape[:2]
    clipped = boxes.copy()
    clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, width)
    clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, height)
    valid = (np.isfinite(boxes).all(axis=1)
             & (clipped[:, 2] > clipped[:, 0])
             & (clipped[:, 3] > clipped[:, 1]))
    valid_rows = np.flatnonzero(valid)
    features = torch.zeros((len(boxes), encoder.feature_dim), dtype=torch.float32)
    valid_mask = torch.from_numpy(valid)

    # A surrounding training loop may have called train(); restore frozen mode.
    encoder.eval()
    if len(valid_rows) == 0:
        return features, valid_mask
    device = next(encoder.parameters()).device
    pil_image = Image.fromarray(image)
    batches = tqdm(range(0, len(valid_rows), batch_size), desc="Appearance features",
                   unit="batch", dynamic_ncols=True,
                   disable=not (show_progress and config.SHOW_PROGRESS))
    for start in batches:
        rows = valid_rows[start:start + batch_size]
        crops = []
        for row in rows:
            box = clipped[row]
            bounds = (int(np.floor(box[0])), int(np.floor(box[1])),
                      int(np.ceil(box[2])), int(np.ceil(box[3])))
            crops.append(encoder.preprocess(pil_image.crop(bounds)))
        batch = torch.stack(crops).to(device)
        encoded = encoder(batch)
        if not torch.isfinite(encoded).all():
            raise RuntimeError(f"Nonfinite appearance features for detection rows {rows.tolist()}")
        features[torch.from_numpy(rows)] = encoded.cpu()
        batches.set_postfix(device=str(device), crops=f"{min(start+batch_size, len(valid_rows))}/{len(valid_rows)}",
                            refresh=False)
        del batch, encoded
    return features, valid_mask
