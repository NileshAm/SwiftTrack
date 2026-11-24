# Fine-tune the CrowdHuman + MOT20 YOLOX-X detector

This uses the existing `dragon-track` environment and the official YOLOX-X weights.
It trains the detector on the videos in `TRAIN_SEQUENCES` and scores it on the
separate videos in `VAL_SEQUENCES` from `Utils/config.py`. It does not change the
tracking model or `config.py`.

## Run

From the project directory:

```bash
conda activate dragon-track
python detector-train-yolox.py
```

Defaults are 5 total epochs, aspect-preserving resize within 960 × 544, batch size
1, gradient accumulation 4, learning rate 5e-5 for the head/neck and 5e-6 for
the backbone, weight decay 1e-5, horizontal flips, and CUDA bfloat16 when on a
GPU. The smaller training resolution fits the available 8 GB GPU in the smoke
check. A larger resolution may run out of memory. Training uses the MOT ground
truth person boxes and passes ignored boxes to YOLOX's native assignment/loss.
It does not train on precomputed SDP/RT-DETR/DDQ/YOLOX detection caches.

For a short code check, use only one frame from each configured video:

```bash
python detector-train-yolox.py --output-dir /tmp/yolox-check --epochs 1 --max-frames 1
```

The short check cannot measure accuracy or predict the outcome of full training.
`--train-fraction 0.25` samples frames evenly across *each* training video;
validation stays complete. `--frame-stride 2` is another way to shorten training.
`--max-frames` caps both training and validation and is only for a small check.

## Results and resume

`outputs/yolox_x_mot17_finetune_v1` contains `best.pth` for detection export,
`best.pt` and `last.pt` for training state, `history.csv`, `settings.json`, and
one validation JSON per epoch. Epoch 0 evaluates the untouched pretrained model.
The best checkpoint is selected by validation F1 at IoU 0.5, sweeping the score
thresholds. Eligible results meet both minimum precision and recall; this takes
priority over F1. If no threshold meets both floors, the best F1 is saved and
marked ineligible in the validation report.

Rerun the same command with a larger total `--epochs` to continue from `last.pt`.
Keep the same output path, data split, sizes and other settings. Changed settings
or data require a new `--output-dir`. The trainer records the split and GT file
hashes to catch accidental changes. A second simultaneous run in the same output
directory is blocked.

## Evaluate exported weights

First export validation detections at the *same resolution* used in training:

```bash
python detect-yolox.py --split val --checkpoint outputs/yolox_x_mot17_finetune_v1/best.pth --width 960 --height 544 --cache-dir detections/yolox_finetuned_960x544_val
python evaluate_detector.py --backend yolox --cache-dir detections/yolox_finetuned_960x544_val --scores 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7
```

Use a different cache directory for the pretrained model, a different checkpoint,
resolution or suppression setting. The full export/evaluator verifies checkpoint
and cache provenance. To compare at the published 1600 × 896 inference size, run
both models at that same size and compare recall at similar precision. Scores in
the training report use 960 × 544, so they are not directly comparable to a
1600 × 896 export. After choosing a detector, create train and validation caches
with the same weights, then train/evaluate the tracker using those caches.

`--score` and `--scores` control checkpoint selection on the held-out videos;
`--min-precision` and `--min-recall` define the eligibility floors. These are
selection settings, not training losses. More detections alone can lower F1.

## Files

- `detector-train-yolox.py`: training, validation, checkpoint selection and resume.
- `Utils/yolox_training.py`: MOT image/box preparation for the native YOLOX loss.
- `Utils/yolox.py`: official or fine-tuned checkpoint loading and inference.
- `detect-yolox.py`: separate detection cache export.
- `evaluate_detector.py`: full-video detection metrics.

## Verification

A one-frame-per-video GPU run completed baseline and two training epochs on the
available GPU; resume began at epoch 2. The exported `best.pth` strictly loaded
into the inference model. Full-sequence training and evaluation have not been run.
