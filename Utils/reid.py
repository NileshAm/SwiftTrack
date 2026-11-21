"""Frozen MSMT17 OSNet features, before the existing relationship/affinity layers.

Uses the official Torchreid architecture without importing its training toolkit.
Geometry and position remain box-derived; OSNet supplies appearance only.
"""
import torch
from torch import nn
from torchvision import transforms

from Utils import config
from Utils.detections import file_digest


class ReIDEncoder(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.feature_dim = 512

    def forward(self, images):
        return torch.nn.functional.normalize(self.network(images), dim=1)


def load_reid_encoder(device):
    from Utils.vendor.osnet import osnet_x1_0

    if not config.FREEZE_ENCODER or not config.USE_PRETRAINED_WEIGHTS:
        raise ValueError("OSNet comparison requires frozen pretrained ReID weights")
    if config.FEATURE_DIM != 512:
        raise ValueError("FEATURE_DIM must be 512 for osnet_reid")
    if tuple(config.CROP_SIZE) != (256, 128):
        raise ValueError("MSMT17 OSNet preprocessing requires CROP_SIZE=(256, 128)")
    path = config.REID_WEIGHTS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Missing official MSMT17 OSNet checkpoint: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    # Classifier identity count belongs to the source dataset, never MOT identities.
    if "classifier.weight" not in state:
        raise ValueError("Expected the official OSNet classification checkpoint")
    network = osnet_x1_0(num_classes=state["classifier.weight"].shape[0], pretrained=False)
    network.load_state_dict(state, strict=True)
    encoder = ReIDEncoder(network).requires_grad_(False).eval().to(device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    encoder.preprocess = transforms.Compose([
        transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    encoder.feature_settings = {
        "encoder": "osnet_reid", "architecture": "osnet_x1_0",
        "source_dataset": "MSMT17_combineall", "weights_sha256": file_digest(path),
        "source_revision": "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50",
        "weights_revision": "a5c5cc037c24235cda3b21085b93ad77c9616224",
        "feature_dim": 512, "crop_size": tuple(config.CROP_SIZE), "color": "RGB",
        "mean": mean, "std": std, "resize": "bilinear_whole_crop",
        "pooling": "global_average_fc_l2", "frozen": True,
        "precision": "float32_tf32_disabled",
    }
    return encoder
