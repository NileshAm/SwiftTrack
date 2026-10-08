# Utils/features.py — frozen appearance features

> **Current settings and architecture:** see [paper alignment](17-paper-alignment.md). The notes below describe earlier development stages; frozen-encoder defaults, previous loss/scoring layers, and old run names are superseded. `main.py` now plots saved run metrics.

> **Updated implementation:** use [DETR, attention, matching, and test videos](11-paper-components.md) for the current code, settings, and per-sequence output paths. The notes below record the earlier baseline and its checks; DenseNet defaults, fixed Sinkhorn, percentage splits, and old flat tracking paths are superseded. Current training and validation use separate complete video lists.

[Back to TODO](README.md) · [Implementation](../../Utils/features.py)

The extractor turns each person crop into **1,024 numbers** describing its appearance. Later, the association model can compare these features across frames. It does not assign person IDs itself.

| Function | Behavior |
| --- | --- |
| `load_encoder(device=None)` | Load DenseNet121 with pinned `IMAGENET1K_V1` pretrained weights, remove the classifier, freeze all parameters, and enter evaluation mode. Defaults to `config.DEVICE`; pass `"cpu"` for debugging. Load once and reuse. |
| `extract_features(image, boxes, encoder, batch_size=None)` | Crop an RGB uint8 image using zero-origin xyxy boxes, resize/normalize, and process small batches. Returns **`(features, valid_mask)`** on CPU, in original detection order. Defaults to `config.CROP_BATCH_SIZE`. |

`features` has shape `[D, 1024]` and `valid_mask` has shape `[D]`. Both are detached CPU tensors. Empty boxes return shapes `[0, 1024]` and `[0]`. Invalid/nonfinite boxes and boxes fully outside the image get zero features and a false mask entry. Never feed those zero placeholder rows into the graph as valid people.

Partially outside boxes are clipped on a copy; original detections are unchanged. Fractional coordinates use floor for left/top and ceil for right/bottom. No detection rows are silently dropped or reordered. Use the mask to filter boxes, scores, row IDs, and any training labels together before building the graph.

## Use with the data loader

Run from the project root in the activated `dragon-track` environment:

```python
from Utils import config
from Utils.data import load_sequence, read_image
from Utils.features import load_encoder, extract_features

sequence = config.DATASET_ROOT / config.DATASET_SPLIT / config.SEQUENCES[0]
frames = load_sequence(sequence, frame_range=(1, 1))
frame = frames[1]
image = read_image(frame)

encoder = load_encoder()  # Once, outside the eventual frame loop.
features, valid = extract_features(image, frame["boxes"], encoder, batch_size=2)
print(features.shape)     # torch.Size([13, 1024]) for this local sample.
print(valid.sum().item()) # 13 usable crops.

# Preserve row alignment when invalid crops occur.
keep = valid.numpy()
boxes = frame["boxes"][keep]
scores = frame["scores"][keep]
row_ids = frame["row_ids"][keep]
features = features[valid]
```

The example uses batch size 2 to leave room for other GPU work. The configurable starting batch size remains 16; reduce it if memory is tight. Do not recreate the encoder for each frame. Each extraction call restores evaluation mode, preventing accidental batch-normalization updates during association training.

## Preprocessing and scope

The whole crop is resized to `config.CROP_SIZE`, currently `(256, 128)` in height/width order, using bilinear interpolation. Pixels become floats in `[0, 1]`, then use the pretrained RGB mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. DenseNet applies ReLU and spatial average pooling before returning the 1,024 features. No additional L2 normalization is applied.

This whole-crop resize is our explicit development choice, not torchvision's default classification center crop or a verified paper preprocessing match. TF32 shortcuts are disabled when loading the encoder so the initial run uses full FP32; these PyTorch backend settings apply to the process. Preserve this setting when comparing outputs and performance.

Weights download once into PyTorch's normal cache. `encoder.feature_settings` records the weight version, crop size, normalization, pooling, and precision; save it with future checkpoints/caches. Setting `USE_PRETRAINED_WEIGHTS=False` creates a random encoder for debugging only. Fine-tuning and other backbones are not implemented. No separate cache module is required.

DenseNet remains the repaired-code starting point, not the paper's DETR feature path. A real crowded MOT20 run remains pending local data.

**Checked in dragon-track:** pretrained CPU and RTX 4060 extraction on 13 MOT17 detections; batch-vs-single and reordered inputs; repeatability; frozen parameters and unchanged normalization statistics; downstream trainable-layer backward; RGB normalization and crop dimensions; empty, clipped, invalid, and nonfinite boxes. CPU/GPU maximum absolute feature difference was about `1.72e-5` with TF32 disabled. Batch size 2 peaked at about 40.2 MiB of PyTorch-allocated GPU memory in this check; this is not full-pipeline or crowded-scene memory. No permanent test files were added.
