# train.py — training, validation, and saved weights

`python train.py` still trains on full videos. For a quick search, the optional
`main(max_train_frames_per_sequence=N)` argument trains on the first N consecutive
frames of each training video (shorter videos use all frames). Validation stays
full-length. Checkpoints record `training_frame_limit` and actual
`training_frame_ranges` separately from the full dataset split. See the
[search frame-limit guide](20-parameter-search.md#reduce-training-time-with-a-frame-limit).

> **Current settings and architecture:** see [paper alignment](17-paper-alignment.md). The notes below describe earlier development stages; frozen-encoder defaults, previous loss/scoring layers, and old run names are superseded. `main.py` now plots saved run metrics.

> **Current training:** [chronological motion/history state](16-state-training.md) supersedes the historical frame-pair protocol below. Use the fresh `mot17_SDP_state30` run.

> **Updated implementation:** use [DETR, attention, matching, and test videos](11-paper-components.md) for the current code, settings, and per-sequence output paths. The notes below record the earlier baseline and its checks; DenseNet defaults, fixed Sinkhorn, percentage splits, and old flat tracking paths are superseded. Current training and validation use separate complete video lists.

[Back to TODO](README.md) · [Implementation](../../train.py)

Implemented using plain PyTorch and the existing helpers. Run from the project root:

```bash
conda activate dragon-track
python train.py
```

## Settings to change

Edit [Utils/config.py](../../Utils/config.py) before running; there are no extra configuration files or command-line options.

### Current final-candidate run

The active configuration now uses **`RUN_NAME="mot17_final"`, all seven FRCNN videos, 20 epochs, learning rate `3e-5`, and 100 Sinkhorn iterations**. It retains the requested 80%/10%/remaining-10% temporal splits, CUDA, 16 crops per encoder batch, and the existing 128-wide/two-layer association network. The crop batch size is not the number of graphs per training step: the training loop still processes one frame pair at a time.

Why these changes: `mot17_all/history.csv` reached its best validation loss at epoch 3 (0.09759), then worsened to 0.20173 at epoch 5 while training loss fell. The lower learning rate is a conservative candidate, not a measured improvement. Twenty epochs provide a budget at that lower rate; `model.pt` still selects the lowest validation loss. The last epoch's weights are not automatically the best. No learning-rate schedule or early stopping has been added.

More Sinkhorn iterations address the large normalization residuals recorded in the previous tracking summaries (maximum about 0.67). A CPU check using the previous best checkpoint on MOT17-04 frames 841–842 (21×21 scores) reduced the maximum mass error from 0.48847 at 20 iterations to 0.05702 at 100. Normalization is still approximate, and this is not an accuracy measurement. More iterations do not add the paper's learnable normalization or change the BCE/contrastive supervision. The training run has not been launched by this configuration update.

The previous validation metrics were HOTA **58.673**, IDF1 **63.831**, MOTA **48.911**, with **5,237 false negatives, 204 false positives, and 73 ID switches**. Misses dominate the MOTA penalty, but these counts alone do not distinguish missing detector boxes from birth/association decisions. Keep the existing detection/matching thresholds for the first comparison, then tune them using validation tracking metrics. Association validation loss is only a proxy for tracking performance. Keep the evaluation remainder untouched until settings are fixed.

Configuration alone cannot reproduce the published metrics: our system uses supplied FRCNN detections, frozen ImageNet DenseNet features, mean message passing, and fixed unmatched normalization. The [paper](https://openaccess.thecvf.com/content/WACV2025/papers/Galoaa_DragonTrack_Transformer-Enhanced_Graphical_Multi-Person_Tracking_in_Complex_Scenarios_WACV_2025_paper.pdf) describes DETR-based detection/features, attention, and learnable Sinkhorn. Our 10% temporal validation subset also differs from a like-for-like paper evaluation. Report this run as the implemented baseline, with measured results rather than a promise of paper-level accuracy.

| Setting | Meaning |
| --- | --- |
| `RUN_NAME` | Output folder name. Use a new name for each fresh run. The normal `OUTPUT_DIR` and `CHECKPOINT_PATH` expressions follow this name. |
| `SEQUENCES` | Shared video list for training, validation, tracking, and evaluation. Current run: all seven MOT17 FRCNN videos. |
| `TRAIN_FRACTION`, `VAL_FRACTION` | Default `0.8` uses the first 80%; `0.1` uses the next 10%; the remainder is held out for evaluation. |
| `EPOCHS`, `LEARNING_RATE` | Current candidate: 20 passes, fixed Adam learning rate `3e-5`. Best weights are selected by validation loss. |
| `DEVICE`, `CROP_BATCH_SIZE` | Default CUDA and 16 crops. Reduce the crop batch to 2 if other programs use GPU memory; use `"cpu"` for small CPU checks. |
| `SAVE_EVERY_EPOCH` | Save/update `last.pt` after every epoch when true; otherwise save it only after the final epoch. The best checkpoint is always saved when validation improves. |

The run uses temporal splits of each selected video. For MOT17-02's 600 frames, training uses 1–480 (479 pairs), validation uses 481–540 (59 pairs), and evaluation reserves 541–600. Training never uses the evaluation portion. No pair crosses a split boundary. This holds out later frames from the same scenes/identities, not entirely unseen videos.

`Utils.data.split_frame_range` reads each video's frame count and computes inclusive boundaries `floor(N*train)` and `floor(N*(train+val))`. Fractions must be positive and sum to less than 1; every partition must have frames, and training/validation each need at least two frames for pairs. Decimal arithmetic prevents rounding surprises such as `0.8+0.1` falling below the intended boundary.

`RUN_NAME="mot17_final"` preserves the old run's files. Train fresh weights for this candidate; checkpoints store both percentages and resolved ranges, and tracking checks those ranges before running.

## Functions and data flow

| Function | What it does |
| --- | --- |
| `load_frame_pairs(sequences, split)` | Resolves `"train"` or `"val"` percentages separately for each video, reads detection metadata/GT once, and returns adjacent pairs within that portion. Empty detection frames remain. |
| `prepare_frame(frame, gt, encoder)` | Loads one image, extracts features, filters invalid crops, and matches GT against exactly those filtered boxes. Returns boxes, features, and labels in the same order. |
| `iter_graphs(frame_pairs, encoder)` | Builds each graph and its targets separately. Reuses the current frame's features for the next ordered pair, keeping only recent features in memory. GT never enters the graph. |
| `train_one_epoch(model, frame_pairs, encoder, optimizer)` | Runs forward → loss → backward → Adam update for each supervised graph. Returns average losses, pair counts, and elapsed seconds. |
| `validate(model, frame_pairs, encoder)` | Computes losses with evaluation mode and gradients disabled. Makes no updates and restores the model's previous mode afterward. |
| `_run_epoch(...)` | Shared small loop behind training/validation. Skips unsupervised graphs, checks finite losses/gradients, prints progress, and collects metrics. |
| `main()` | Sets seeds, loads splits and models, loops over epochs, selects the best weights using validation loss, and saves history/checkpoints. Returns the history rows. |

One graph is processed at a time in FP32. DenseNet stays frozen in evaluation mode. Pairs run in sequence/frame order; there is no shuffle, scheduler, worker pipeline, disk feature cache, or full-sequence graph cache. Frozen features are extracted again each epoch.

Previous-frame detections stand in for tracks. This is a baseline for adjacent-frame association; longer-lived track history, occlusion recovery, and tracking metrics belong to the tracker/evaluation steps.

## Reading the losses

The existing `compute_loss` combines class-balanced BCE on raw logits and a contrastive loss on learned embeddings. Ground-truth labels only define targets and the loss mask. Invalid/unmatched identities are excluded rather than fabricated as negative examples.

Each training/validation pass has a `tqdm` progress bar with completed pairs, elapsed time, estimated time remaining, processing rate, device, average total/BCE/contrastive losses, and skipped pairs. Set `SHOW_PROGRESS=False` for quiet scripted runs. Each epoch still prints its final losses and saves the CSV. Epoch losses are the arithmetic mean over supervised frame-pair graphs; they are not weighted by the number of detection pairs. `positive_pairs`/`negative_pairs` count supervised candidate pairs, while `used_pairs`/`skipped_pairs` count frame-pair graphs.

Training losses are measured before each optimizer update; validation uses the model after the epoch. An unsupervised graph is skipped. If an entire split has no supervised pairs, the run stops with an explanation instead of reporting a misleading zero loss. Nonfinite loss or gradients also stop the run.

## Saved files

Inside `outputs/<RUN_NAME>/`:

- **`model.pt`** (or `CHECKPOINT_PATH`): weights from the epoch with the lowest validation total loss.
- **`last.pt`**: most recent saved epoch, controlled by `SAVE_EVERY_EPOCH`.
- **`history.csv`**: one row per completed epoch with training/validation losses, counts, and elapsed seconds. Open it in a spreadsheet to plot the curves.

Checkpoints include model/optimizer states, constructor settings, epoch, seed, feature weights/preprocessing, the config snapshot including selected sequences/ranges, metrics, and the best epoch/loss so far. Paths are stored as strings. DenseNet weights are referenced by their pinned pretrained version, not duplicated in the checkpoint.

Each launch starts fresh. Existing checkpoint/history files are protected: choose a new `RUN_NAME` before starting another run. There is no automatic resume option; optimizer state is included for a later manual continuation if needed. Graph/model demonstration images do not block training.

Load the best association model into an existing graph workflow:

```python
import torch
from Utils import config
from model import AssociationModel

checkpoint = torch.load(config.CHECKPOINT_PATH, map_location="cpu", weights_only=True)
model = AssociationModel(**checkpoint["model_settings"]).to(config.DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
with torch.no_grad():
    output = model(graph)  # An already prepared graph, with matching feature settings.
```

When preparing new graphs, use the saved `settings` and `feature_settings` to reproduce crop size, encoder, coordinate convention, detection filtering, and Sinkhorn settings. Loading weights alone does not restore these global configuration values. `main.py` still starts its own random model for explanation; it does not automatically load these training weights.

## Verification

A temporary CPU check of the percentage protocol used a 20-frame copy of real MOT17-02 data: training 1–16, validation 17–18, evaluation 19–20. One epoch completed, the checkpoint saved exact boundaries, and tracking plus TrackEval ran on only the validation portion. The original data and configured run directory were not modified by this check.

Earlier verification under the previous separate-video protocol covered best-checkpoint selection, optimizer state, CSV output, unchanged weights during validation, and fresh-process reload with identical logits/scores. Diagnostic checkpoints were saved under `/tmp`.

This verifies the small training runner, not tracking accuracy or the paper's full training schedule. The existing `main.py` provides the separate one-pair learning demonstration. Full training, real MOT20 verification, and held-out tracking evaluation still need to be run.
