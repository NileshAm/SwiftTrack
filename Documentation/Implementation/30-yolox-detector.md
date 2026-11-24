# Pretrained CrowdHuman + MOT20 YOLOX-X

This is a separate detection and evaluation path in `dragon-track`. It uses only
ByteTrack's YOLOX-X detector; association remains your existing tracking model.
No environment replacement or config.py changes are needed. Optional MOT fine-tuning is described in [YOLOX training](31-yolox-training.md).

## Run a small check, then export and evaluate

From the project directory:

```bash
conda activate dragon-track
python test-yolox.py
python detect-yolox.py --split val --cache-dir detections/yolox_x_mot20_val_v1
python evaluate_detector.py --backend yolox --cache-dir detections/yolox_x_mot20_val_v1 --scores 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7
```

`test-yolox.py` also writes `evaluation.json` with the shared GT matching and
confidence sweep on that one frame. `detect-yolox.py --smoke-test` skips GT scoring.

The smoke check reads the first frame of the first configured validation video,
loads the real pretrained weights, and saves a preview and JSON boxes under
`outputs/yolox_smoke`. It does not create a full-video cache or claim validation
accuracy. Full export reads every frame of VAL_SEQUENCES, with parent/frame tqdm
progress and ETA. Use `--split train`, `test`, or `all` for other configured lists.

The first run downloads the official checkpoint (about 756 MiB) under
`outputs/yolox_weights`, verifies its published SHA-256 prefix `9ce38a60`, and
strictly loads the detector weights. An existing copy may be supplied with
`--checkpoint`; this entry point also accepts `best.pth` exported by `detector-train-yolox.py`.

## Preprocessing and results

Defaults follow the published MOT20 config: resize within width 1600 / height 896
while preserving aspect ratio, pad to a multiple of 32 with pixel value 114,
BGR input without ImageNet normalization, one pedestrian class, NMS IoU 0.7.
Inference uses FP32 with TF32 disabled. Confidence is YOLOX's combined objectness
and class score. The cache floor is 0.10; evaluation can sweep higher thresholds.

Change resolution with `--width` and `--height`, suppression with `--nms-iou`, or
cache floor with `--min-score`. Always use a new cache directory after changes.
Do not assume RT-DETR/DDQ confidence thresholds transfer to YOLOX.

Each video gets `det.txt`, `preview.jpg`, and `summary.json`. MOT boxes are saved
in original-image coordinates, one-origin xywh, with confidence and identity -1.
Manifests record full video coverage, checkpoint/config hashes, versions and time.
Complete compatible videos skip on rerun; an interrupted video restarts. A lock
prevents concurrent writers. The evaluator verifies hashes, coverage and backend,
and rejects DDQ/RT-DETR caches accidentally supplied as YOLOX.

Evaluation reports precision, recall, F1, false positives/misses, duplicate counts,
size/visibility recall, per-video and combined results at IoU 0.5/0.75. These are
detection diagnostics, not HOTA/IDF1 or COCO AP. Compare recall at similar precision,
not raw box counts. After a detection improvement, separately measure tracking.

```bash
python compare-detectors.py --reports /absolute/path/to/yolox/summary.json /absolute/path/to/rtdetr/summary.json --min-precision 0.70 --min-recall 0.55
```

## Data provenance

The released model uses CrowdHuman train+val and MOT20 train. It does not use
MOT17 training images for gradient training in the published config. Upstream
validation is on MOT17 train, so it is not an untouched final benchmark there.
Do not evaluate this checkpoint on MOT20 train as held-out data. The cached
recipe records this distinction. Avoid substituting the MOT17 checkpoint when
using whole MOT17 videos as your validation set.

## Files

- `detect-yolox.py`: one-frame check, configured split export, previews, cache safety.
- `test-yolox.py`: one-frame inference and shared evaluator integration check.
- `Utils/yolox.py`: checkpoint retrieval, official detector configuration,
  strict state loading, preprocessing and original-coordinate predictions.
- `evaluate_detector.py --backend yolox`: shared matching and reporting.

Source: [MMDetection 3.3 ByteTrack model zoo](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/bytetrack/README.md).

## Verification performed

Verified official checkpoint hash and strict loading, GPU inference and GT scoring
on one validation frame, then real two-frame MOT export through the full evaluator.
Inspected the preview. Synthetic checks verify original-coordinate MOT output,
backend rejection and tampered-cache rejection. Existing DDQ caches still load.
Full validation has not been run; these checks do not establish accuracy gains.
