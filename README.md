# SwiftTrack
An extension of DragonTrack, a transformer-enhanced graph-based multi-person tracking framework. This project builds upon the original DragonTrack implementation with the goal of investigating, developing, and evaluating further improvements to the tracking framework.

## Run the implemented pipeline

From this folder in the existing Conda environment:

```bash
conda activate dragon-track
python import_tracktor.py  # Verify/reuse the imported public detections.
python train.py
python tracking.py
python evaluate.py
```

Edit [Utils/config.py](Utils/config.py): `DATASET_ROOT` and `DATASET_SPLIT` locate the labelled dataset, while `TRAIN_SEQUENCES` and `VAL_SEQUENCES` select separate complete videos. Both lists can reference folders inside MOT17's `train` directory; there is no need to move or copy videos.

```python
TRAIN_SEQUENCES = [
    "MOT17-04-SDP", "MOT17-05-SDP", "MOT17-10-SDP",
    "MOT17-11-SDP", "MOT17-13-SDP",
]
VAL_SEQUENCES = ["MOT17-02-SDP", "MOT17-09-SDP"]
```

This editable 5/2 example uses every frame of each selected video. There are no percentage splits or leftover eval frames. The code rejects shared videos even across detector variants: `MOT17-02-SDP` in training and `MOT17-02-FRCNN` in validation is not allowed. Track memory is reset between videos and between training and validation. Validation videos never update weights.

The fresh run is `outputs/mot17_fixed_matching/`. **Retrain from scratch:** the graph uses the paper’s attention equations with corrected matching normalization and cosine-aligned contrastive distances, DETR appearance uses frozen original pretrained weights, and boxes come from imported Tracktor-preprocessed SDP detections. Previous outputs are preserved. Tracking rejects checkpoints trained with a different observation window or detector source.

`FREEZE_ENCODER=True` reloads the original pinned pretrained DETR weights and excludes them from optimizer updates. The collapsed fine-tuned checkpoint is preserved for comparison and is not loaded by this fresh run. Retrain the association model with the restored features. Matching now uses fixed row/column totals with a learned unmatched preference, and contrastive loss uses the same normalized embeddings as cosine matching. Old matcher checkpoints are incompatible; see the [matching/loss repair guide](Documentation/Implementation/19-matching-and-loss-fix.md).

`DETECTION_BACKEND="tracktor"` uses the authors-linked preprocessed public detections. The archive and all 11 configured videos are already imported; `import_tracktor.py` checks and reuses them. Upstream IDs are discarded and missing confidence values receive equal weight. `detect.py` skips inference in this mode. Raw SDP and RT-DETR caches remain available for separate comparison runs. See the [paper-alignment guide](Documentation/Implementation/17-paper-alignment.md).

`LOOKBACK_FRAMES=30` controls the shared maximum observation gap. Training and validation walk through videos in order, score current detections against past track states, then update memory using known identities. Tracks can have different ages in one graph, and missing detections create natural gaps. Tracking allows reconnection at exactly a 30-frame gap and expires older observations. See the [state-training guide](Documentation/Implementation/16-state-training.md) for the past-only update order and teacher-forcing limitation.

`tracking.py` tracks complete `VAL_SEQUENCES` when `RUN_SPLIT_TRACKING=True`, then the optional `TEST_SEQUENCES`. `evaluate.py` scores only complete validation videos and saves their metrics. These validation scores are used for model selection/tuning; they are not an independent final test score.

`TEST_SEQUENCES` still selects full videos under `TEST_DATASET_ROOT / TEST_DATASET_SPLIT`. `SAVE_TEST_VIDEO=True` saves their videos automatically. Set `TEST_SEQUENCES=[]` to skip them, or `RUN_SPLIT_TRACKING=False` to track only test videos. No test GT is read.

```text
outputs/mot17_fixed_matching/
  model.pt, last.pt, history.csv
  tracking/paper_motion_history/val/<sequence>/tracks.txt, summary.json, preview.jpg
  tracking/paper_motion_history/test/<sequence>/tracks.txt, summary.json, preview.jpg, video.mp4
  evaluation/paper_motion_history/val/results.md
```

`SAVE_VIDEO=True` also saves validation videos. Existing results are protected; choose a fresh output folder to repeat tracking. Progress bars use `tqdm`. The model uses frozen pretrained full-image DETR appearance features and uses learned relationships, 512→128 graph attention, and learnable matching. Read the [current component guide](Documentation/Implementation/17-paper-alignment.md) and [TODO](Documentation/Implementation/README.md) for details and remaining differences from the paper.

Training, validation loss, and tracking use a per-ID Kalman motion filter and a confidence-weighted history of
up to 30 reliable appearance observations. `USE_MOTION_PREDICTION` and
`USE_APPEARANCE_HISTORY` control these independently. `TRACKING_VARIANT` separates
comparison results while reusing the same compatible checkpoint. Only observed
detection boxes are exported. See the [motion/history guide](Documentation/Implementation/15-motion-and-history.md).

## Batched training

`TRAIN_BATCH_SIZE=2` now runs two full images through DETR together; `VAL_BATCH_SIZE=4` batches validation extraction. Motion/history updates and graph scoring remain chronological. `DETR_GRADIENT_CHECKPOINTING=False` uses spare VRAM to avoid activation recomputation.

A short RTX 4060 check measured about 4.0 frames/second and 2.7 GiB peak PyTorch allocation with these training settings. Batch 4 used about 5.1 GiB without a meaningful speed gain. Full-video speed/memory can differ. See [batch settings and checks](Documentation/Implementation/18-batched-training.md).

## Parameter search without editing config.py

Edit the settings at the top of [search.py](search.py), then run `python search.py`.
`PARAM_GRID` lists parameter names and candidate values; `EPOCH_COUNTS` sets
independent training budgets, such as `[3, 5]`, without using `config.EPOCHS`.
Every combination trains from scratch, tracks the full validation videos, and
runs evaluation. For example, two learning rates with `[3]` create two 3-epoch runs.
`TRAIN_MAX_FRAMES_PER_SEQUENCE=200` optionally trains on the first 200 frames of
each training video; `None` uses full videos. Validation remains full-length.
`MODE="tracking"` instead sweeps tracker settings with an existing checkpoint.

Separate timestamped folders under `outputs/searches/` contain each trial's
outputs, a ranked `results.csv`, and `best.json` with the winning parameters and
checkpoint path. With `RESUME=True`, restarting the same search reuses its folder,
skips completed trials, and restarts the unfinished trial from epoch 1 while
archiving its partial outputs. Set `RESUME=False` for a fresh search.
Settings are overridden only in memory; `Utils/config.py` stays
unchanged. See the [search guide](Documentation/Implementation/20-parameter-search.md)
for grid examples, epoch semantics, progress and checkpoint selection.

## Continue training

The config now has `RESUME_TRAINING=True` and `EPOCHS=40`. Run `python train.py`
to continue the epoch-20 residual checkpoint through epoch 40. The epoch target is
total, not additional. Weights, Adam state, best validation loss and history resume
automatically; existing run files are backed up first. `last.pt` saves every epoch.
Use a new `RUN_NAME` to start fresh. See the [resume guide](Documentation/Implementation/23-resume-training.md).

## Appearance diagnostic

The current model preserves each node's own features through a learned residual
projection in both attention layers. Run `python train.py` for the new
`mot17_appearance_residual_v1` experiment before tracking or running the diagnostic;
old checkpoints are incompatible and remain preserved. See the
[residual guide](Documentation/Implementation/22-appearance-residual.md).

Run `python test.py --max-frames 100` for a short comparison of appearance before
and after graph attention, or `python test.py` for full validation videos.
It uses the configured checkpoint and saves per-video similarity plots, ranking
results and gate exclusions under the checkpoint folder's `feature_diagnostics/`.
It does not train or edit config. See the [diagnostic guide](Documentation/Implementation/21-feature-diagnostic.md).

## Training and evaluation charts

`SAVE_PLOTS=True` saves training/validation loss curves, timings, and supervision counts after every epoch. Evaluation saves combined and per-video HOTA/IDF1/MOTA, detection/association scores, missed detections/false positives/identity switches, and IoU threshold diagnostics.

To plot results you already have, without retraining or evaluating again:

```bash
python main.py
# Or choose an older run:
python main.py --run-dir outputs/mot17_attention
```

Open the PNGs in `outputs/<run>/plots/training/` and `outputs/<run>/evaluation/<TRACKING_VARIANT>/val/plots/<benchmark>/`. Missing evaluation results are skipped until you run `evaluate.py`. New runs also save `losses_per_pair.png` with weighted BCE per sample weight and contrastive loss per pair. The original objective remains a sum per graph. Training loss is not tracking accuracy; tracking scores are for the evaluated checkpoint, not every epoch. See the [plotting guide](Documentation/Implementation/12-plots.md) for chart meanings and helper functions.

## Detector evaluation and full-resolution detection

Compare box recall/precision before retraining. These commands leave the configured
training detector source unchanged:

```bash
python evaluate_detector.py --backend tracktor
python detect.py --backend rtdetrv2 --split val --input-mode original --cache-dir detections/rtdetrv2_s_original_score010
python evaluate_detector.py --backend rtdetrv2 --cache-dir detections/rtdetrv2_s_original_score010
```

The evaluator saves confidence sweeps, missed/duplicate counts, size/visibility
recall, and plots under `outputs/detector_evaluation/`. Use `--max-frames 100` for
a quick evaluation check. Original-resolution inference preserves image pixels
and pads the edges; it costs more memory than 640x640. See the
[detector guide](Documentation/Implementation/24-detector-evaluation.md) for comparisons,
parameter ranges, metric definitions and how to train on the winning cache.

## Fine-tune the detector

Run `python detector-train.py` to fine-tune RT-DETRv2 using the configured training
videos and validate on the held-out videos each epoch. Separate `DETECTOR_TRAIN_*`
settings control this run. It saves the best validation-F1 detector, latest resume
checkpoint, metric reports and curves. The pretrained model is evaluated as epoch
zero, so fine-tuning must beat that baseline to replace the best weights.

Resume by running the command again; use `--epochs 20` to extend the total target.
This does not switch the existing association model's detection source.
See [detector training](Documentation/Implementation/25-detector-training.md) for
settings, ignored-label handling and commands to export detections with the trained model.

## Detector parameter search

`python detector-search.py --mode reports` ranks existing detector epoch/threshold
reports without training. `python detector-search.py --mode train` runs three short
learning-rate trials with a confidence sweep and minimum precision/recall goals.
Edit the grid and independent epoch budget in `detector-search.py`; config.py is
not modified. Completed trials are skipped and interrupted trials resume.
See the [detector search guide](Documentation/Implementation/26-detector-search.md)
for current findings, checkpoint selection and larger-model comparisons.

## Compare DDQ-DETR with RT-DETR

In the same `dragon-track` environment, `python detect-ddq.py --split val` exports
five-scale DDQ person detections separately. Use `--variant 4scale` for the earlier
detector. `python detect-ddq.py --smoke-test` checks one
frame first. Use `evaluate_detector.py --backend ddq --cache-dir <ddq-cache>` for
the same metrics as RT-DETR, then `compare-detectors.py --reports <rt-summary.json>
<ddq-summary.json>` to compare recall and F1. No active detector settings are changed.
See [DDQ setup and comparison commands](Documentation/Implementation/27-ddq.md).

# Acknowledgement

This project builds upon the **DragonTrack** framework introduced in the paper:

**Transformer-Enhanced Graphical Multi-Person Tracking in Complex Scenarios**  
Original paper: https://openaccess.thecvf.com/content/WACV2025/html/Galoaa_DragonTrack_Transformer-Enhanced_Graphical_Multi-Person_Tracking_in_Complex_Scenarios_WACV_2025_paper.html

Original DragonTrack repository:  
https://github.com/ostadabbas/DragonTrack

This repository extends and modifies the original DragonTrack implementation with the goal of investigating, developing, and evaluating further improvements to the framework.

Please refer to the original DragonTrack work and cite the corresponding paper when using this project for research.

# License

This project is released under the MIT License. Portions of the code derived from the original DragonTrack implementation remain subject to the copyright and license notices of the original authors.

## Overall progress and ETA

With `SHOW_PROGRESS=True`, `tqdm` shows an overall bar above the detailed bar:

- **Training overall:** all training and validation frame pairs across every epoch; shows the current epoch and phase. ETA starts during the first epoch.
- **Tracking overall:** all frames across the selected validation and test videos, including video rendering; shows the current video and split.
- **Evaluation overall:** all videos scored by TrackEval; updates after each video's HOTA, CLEAR, and Identity metrics finish. ETA becomes available after the first completed video. Input preparation retains its separate bar.

ETAs use the average observed processing rate and can change with video crowding and training/validation speed. The overall bars start after dataset/model setup; evaluation report writing and final output cleanup are not separate counted work units. `SHOW_PROGRESS=False` hides both overall and detailed bars.
