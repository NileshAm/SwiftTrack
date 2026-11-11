# Fine-tune RT-DETRv2 on MOT training videos

For a joint confidence/training-parameter search, see [detector search](26-detector-search.md).
The default direct command below retains fixed-threshold checkpoint selection for
compatibility. New search trials use `--selection-mode sweep` and record the chosen
threshold plus eligibility in the checkpoint/export manifest.

`detector-train.py` trains the **box detector**. It is separate from `train.py`,
which trains the association model. It reads images and MOT ground truth from
`TRAIN_SEQUENCES`, and evaluates `VAL_SEQUENCES` after every epoch. It never trains
on supplied SDP/Tracktor/generated detection boxes or on validation labels.

## Run

```bash
conda activate dragon-track
python detector-train.py
```

Defaults in `Utils/config.py` under **DETECTOR FINE-TUNING**:

| Setting | Default | Meaning |
| --- | --- | --- |
| `DETECTOR_TRAIN_OUTPUT_DIR` | `outputs/detector_rtdetrv2_s_mot17_v1` | Separate detector run; existing association runs remain intact. |
| `DETECTOR_TRAIN_EPOCHS` | 10 | Total epoch target, not additional epochs. |
| `DETECTOR_TRAIN_IMAGE_SIZE` | 640 | Square training/validation input. Use a new run for 960. |
| `DETECTOR_TRAIN_BATCH_SIZE` | 1 | Images per minibatch, conservative for 8 GB VRAM. |
| `DETECTOR_TRAIN_ACCUMULATION` | 4 | Average gradients over minibatches before updating. Effective full batch = 4 images. |
| `DETECTOR_TRAIN_LR` | 0.00005 | AdamW learning rate for detector heads/encoder/decoder. |
| `DETECTOR_TRAIN_BACKBONE_LR` | 0.000005 | Lower learning rate for trainable backbone parameters. |
| `DETECTOR_TRAIN_WEIGHT_DECAY` | 0.0001 | AdamW weight decay. |
| `DETECTOR_TRAIN_FRAME_STRIDE` | 1 | Use every training frame; 3 uses every third frame. Validation always uses every frame. |
| `DETECTOR_TRAIN_SCORE` | 0.4 | Fixed score threshold for best-checkpoint selection at IoU 0.5. |
| `DETECTOR_TRAIN_IGNORE_IOU` | 0.5 | Ignore-region overlap threshold for unmatched classification predictions. |

The script starts from `DETECTOR_MODEL_NAME` / `DETECTOR_REVISION` (the pinned
COCO RT-DETRv2-S checkpoint), or resumes its own completed checkpoint. It does not
use `DETECTOR_WEIGHTS_DIR` as a training initializer. It keeps the existing COCO
head dimensions/weights and supplies only the resolved person class as positive
labels. It is optimized for person detection; preservation of other COCO categories
is not an objective. All parameters that the loaded model exposes as trainable are
optimized, with the lower backbone learning rate; pretrained backbone freezing
and batch-normalization conventions are retained.

This uses FP32, gradient clipping at norm 1, a constant learning rate, and random
horizontal flips for training only. No added mixed-precision framework or package
installation is needed in the existing Transformers 4.49 environment. The small
loss adapter uses that version's RT-DETR loss interfaces.

Overall and phase `tqdm` bars show progress and ETA. Only one GPU job should run at
a time. Validation has no augmentation, no gradients, and no optimizer updates.

## Resume and checkpoints

Run the same command again after interruption. Training resumes at the next epoch
following the newest complete `last.pt` / `best.pt`, restoring model, AdamW state,
history and random state. An interrupted epoch restarts from its beginning; there
is no mid-epoch recovery. To extend ten epochs to twenty:

```bash
python detector-train.py --epochs 20
```

Changing the image size, learning rates, batch settings, sampling, GT contents or
video lists requires a new `--output-dir`. Increasing the epoch target is allowed.
Checkpoints record GT/sequence-metadata checksums and selected frame IDs. Do not
change source images in place. Reproducibility across different devices/library
versions is not guaranteed.

Before epoch one, the script evaluates the pretrained baseline (epoch 0).
**Best** means highest combined validation F1 at fixed score 0.4 and IoU 0.5,
including this baseline. Ties retain the earlier checkpoint. If fine-tuning never
improves that metric, `best` remains epoch 0; this is deliberate and visible in
the manifest/history. It is not COCO AP, HOTA, IDF1 or a paper benchmark score.

Saved files:

```text
outputs/detector_rtdetrv2_s_mot17_v1/
  settings.json                  # Settings, train/val videos and GT provenance
  last.pt                        # Latest completed epoch, optimizer and RNG
  best.pt                        # Best fixed-threshold validation F1
  best/                          # Hugging Face model + processor for detect.py
    config.json
    preprocessor_config.json
    model.safetensors
    training_manifest.json       # Best epoch/F1 and export integrity hashes
  history.csv                    # Epoch 0 baseline, losses, P/R/F1 and timing
  training.png                   # Loss and fixed-threshold validation curves
  validation_epoch_000.json      # Baseline detection metrics
  validation_epoch_001.json      # Each epoch: per-video/combined counts, rates,
  ...                            # size/visibility recall, IoUs .5/.75, score sweep
```

`best/` is refreshed when F1 improves and at successful completion. Rerunning an
already completed target repairs an interrupted export. Checkpoint files use
atomic replacement. The export manifest is written last, so incomplete exports
are rejected by detection. Stop training before using `best/` for full cache
creation; it can change while training is running.

Train and validation losses are not directly comparable: training includes
augmentation and denoising supervision that validation does not. Look at the
validation metric curves and the pretrained baseline when deciding whether the
run helped. A rising F1 with poor IoU-0.75 performance still needs box inspection.

## MOT targets and ignored annotations

- Marked class-1 pedestrians become targets, including heavily occluded people.
- Positive and ignored boxes are clipped to the image for loss computation;
  degenerate clipped boxes are discarded. Original GT is retained for evaluation.
- Non-target annotations are ignored regions, not extra positive classes.
- Hungarian matching, box/GIoU losses, decoder/encoder auxiliary losses and
  denoising supervision use the Transformers RT-DETR implementation.
- For unmatched predictions overlapping ignored boxes, negative classification
  loss is masked. Matched positives and predictions overlapping valid people keep
  their supervision, so ignored boxes do not suppress duplicate-person negatives.
- Ignored boxes are normalized/transformed alongside targets, including flips.
- This loss/evaluation ignore policy is a local MOT adaptation; the diagnostic
  evaluation uses the same policy as `evaluate_detector.py`, not TrackEval's
  official benchmark-specific preprocessing.

The current targets are full MOT person boxes. Do not remove low-visibility GT to
inflate validation recall. Do not add detector variants of the same video as
independent training examples; overlapping train/validation videos are rejected.

## Small check or alternate experiment

```bash
# First two frames per train/val video, one epoch; isolated from the real run:
python detector-train.py --epochs 1 --max-frames 2 --output-dir outputs/detector_smoke

# An independent 960 experiment; larger inputs need more memory:
python detector-train.py --image-size 960 --output-dir outputs/detector_rtdetrv2_s_960_v1
```

`--max-frames` is only a smoke-check cap, not full validation. It limits the first
N selected frames of each video, after applying the training stride. Output
metadata records the cap so it cannot silently resume as a full-data experiment.
All CLI overrides are process-local and do not edit config.py.

## Generate and evaluate the best detector's boxes

After training finishes:

```bash
python detect.py --backend rtdetrv2 --weights-dir outputs/detector_rtdetrv2_s_mot17_v1/best --input-mode square --image-size 640 --split val --cache-dir detections/rtdetrv2_s_mot17_finetuned_640_v1
python evaluate_detector.py --backend rtdetrv2 --cache-dir detections/rtdetrv2_s_mot17_finetuned_640_v1
```

Use `--image-size 960` for a 960-trained run. Start evaluation with the same input
size used during training. Cache recipes include hashes of the exported weights,
processor, model config and training manifest. Retraining/replacing weights makes
an old cache incompatible: use a new cache directory instead of mixing epochs.
Old pretrained caches remain compatible while `DETECTOR_WEIGHTS_DIR=None`.

If the best detector outperforms the baseline, generate train and validation boxes
with the same weights/recipe (`--split train`, then `--split val`). For association
training set `DETECTOR_WEIGHTS_DIR` to the exported `best/`,
`DETECTION_BACKEND="rtdetrv2"`, and choose the matching cache directory, mode,
input size and cache floor in config. Set a fresh association `RUN_NAME` and
`RESUME_TRAINING=False`; do not resume the Tracktor-trained association model.
Fine-tuning alone does not change the active Tracktor detection source.

## Significant functions

- `Utils/detector_training.MOTDetectionDataset`: image/GT records and split provenance.
- `DetectionCollator`: COCO-style targets, ignored boxes and consistent flips.
- `MOTRTDetrLoss.loss_labels_vfl`: mask ignored-region negative classification loss.
- `install_mot_loss`: attach that criterion to the model, preserving auxiliary/denoising packaging.
- `detector-train.train_epoch`: gradient accumulation, clipping and optimizer updates.
- `validate_epoch`: loss plus original-resolution detection diagnostics on held-out frames.
- `export_best`: export the selected detector and integrity manifest for inference.
- `Utils/detections.detector_settings`: validate/hash local fine-tuned exports in cache recipes.

Reference: [Transformers 4.49 RT-DETRv2 training API](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/rt_detr_v2).
