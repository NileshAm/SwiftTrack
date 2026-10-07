# Batched appearance extraction for training

[Back to TODO](README.md) · [Config](../../Utils/config.py) · [Paper-alignment guide](17-paper-alignment.md)

Training now processes multiple full images in one DETR call. Previously, `TRAIN_BATCH_SIZE` only accumulated gradients from separate one-image calls. It now controls a real image batch and one optimizer update. Graphs are still scored in chronological order to preserve motion/history state.

## Settings

```python
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 4
DETR_GRADIENT_CHECKPOINTING = False
```

These settings are in `Utils/config.py`. Run `python train.py` normally. The progress bar shows the current batch size; a video's final batch may be smaller. A running Python process retains the settings it loaded at startup. Existing results remain protected; choose a fresh `RUN_NAME` if that output folder already contains a checkpoint/history. This change does not add training-resume support.

Batch size 2 retains the paper's training batch size. Increasing it changes the number of optimizer updates per epoch and may affect accuracy; it is not just a speed switch. Validation batching does not update weights. Batching does not change checkpoint architecture or tracking's single-frame interface.

Activation checkpointing recomputes intermediate image activations during backward to reduce VRAM usage. Turning it off uses more memory and was faster in the local check. If memory runs out on a more demanding video, reduce the batch size or turn checkpointing back on. There is no automatic fallback that silently changes the training recipe.

## Measured local check

RTX 4060 Laptop GPU, full configured DETR resolution (short side 800 / maximum 1333), FP32, two PyTorch CPU threads, 12 current frames from MOT17-04 after warmup. Times include image reads, preprocessing, chronological graph/loss work, and optimizer updates.

| Training batch | Checkpointing | Frames/second | Peak PyTorch GPU allocation |
| --- | --- | --- | --- |
| 1 | On | 3.11–3.20 | 1.38 GiB |
| 2 | On | 3.16 | 2.15 GiB |
| 4 | On | 3.13 | 3.76 GiB |
| 1 | Off | 3.87 | 1.56 GiB |
| 2 | Off | 3.96–3.98 | 2.73 GiB |
| 4 | Off | 3.97 | 5.07 GiB |

Batching alone did not give a meaningful speed increase on this short clip. Disabling checkpointing improved throughput by roughly 25%; batch 2 matched batch 4's speed with lower memory usage. These are short-run measurements, not a full-epoch ETA, GPU reservation totals, or maximum memory requirements across every video. Graph construction, state updates, and image processing still take time outside the batched network pass.

## Functions and ordering

- **`DETRAppearance.extract_batch(images, boxes_per_image)` — [appearance.py](../../Utils/appearance.py):** validates equal image dimensions, encodes valid frames together, pools each image's own boxes, and returns aligned CPU features/masks. Empty or entirely invalid box lists retain their positions in the returned list without running through DETR. It rejects mixed image dimensions to avoid padding-dependent feature changes.
- **`extract_feature_batch(...)` — [features.py](../../Utils/features.py):** dispatches to batched DETR with training gradients enabled where appropriate. Optional frozen DenseNet keeps its existing per-image crop processing.
- **`iter_graphs(...)` — [train.py](../../train.py):** batches within a video, precomputes only independent image features, then predicts past track state, constructs the current graph, yields it for scoring, and updates memory. It never puts future labels/observations into an earlier graph. Video boundaries reset history and flush partial batches.
- **`_run_epoch(...)` — [train.py](../../train.py):** sums losses that share encoder activations, performs one backward pass at `batch_end`, divides gradients by the actual number of supervised graphs, then clips and steps. Skipped graphs do not enter the divisor. This avoids repeated backward through the same encoder activations or changing weights before all batch losses have been scored.

DETR treats images as separate batch entries. Its backbone uses frozen BatchNorm statistics, and transformer attention stays within each image. Track history features remain detached. Dropout during training can differ from serial processing, so bit-for-bit identical training trajectories are not promised.

## Checks

- [x] Batched vs single-frame features and gradients, including empty/invalid box rows.
- [x] Changing a later image does not change an earlier image's extracted features in evaluation mode.
- [x] Validation graph inputs, labels, ages, and history sizes agree with serial extraction.
- [x] Shared backward updates the appearance encoder; validation leaves weights unchanged.
- [x] Incomplete batches and video boundaries work without carrying track history across videos.
- [x] Short GPU training/checkpoint reload/tracking/video/TrackEval pipeline.
- [ ] Measure full-epoch speed and full-video tracking accuracy for the next training run.
