# Detector evaluation and original-resolution inference

These tools measure the box coverage problem before spending time retraining the
association model. `detect.py` runs pretrained RT-DETRv2 inference;
`evaluate_detector.py` scores saved boxes against labeled videos. Neither trains
the detector. Run commands from the project root in `conda activate dragon-track`.

## 1. Measure the existing sources

```bash
python evaluate_detector.py --backend tracktor
python evaluate_detector.py --backend supplied
python evaluate_detector.py --backend rtdetrv2 --cache-dir detections/rtdetrv2_s_coco_640_score010
```

Validation is the default. `--max-frames 100` uses only the first 100 frames of each
validation sequence for a quick check; use the complete videos for decisions.
`--split train` is available for diagnosing training data, not selecting settings.
Unlabeled test videos cannot be evaluated here. The configured train/validation
video lists are retained. Backend/cache overrides do not edit `Utils/config.py`.

Each run creates a separate directory under `outputs/detector_evaluation/` with:

- `report.md`: combined precision, recall, F1, TP, FP, FN, duplicate and ignored counts.
- `summary.csv`: the same counts and rates per video and combined, plus recall by
  original GT person height (<50, 50–<100, >=100 pixels) and visibility
  (<0.3, 0.3–<0.7, >=0.7). Empty denominators are blank/null, not zero.
- `summary.json`: source paths, checksums, saved detector recipes, cache-generation
  timing (when available), evaluated frame ranges and exact metric policy.
- `detector_metrics.png`: confidence sweep and size-specific recall at the first IoU.

The default sweep scores **all cached boxes**, then thresholds 0.1, 0.2, 0.3,
0.5 and 0.7, each at IoU 0.5 and 0.75. Change it with:

```bash
python evaluate_detector.py --backend rtdetrv2 --cache-dir detections/rtdetrv2_s_coco_640_score010 --scores 0.1 0.15 0.2 0.3 0.4 0.5 --ious 0.5 0.75
```

`DETECTION_MIN_SCORE` is deliberately ignored during this explicit sweep. A cache
floor still applies: a cache exported at 0.1 contains no boxes below 0.1. Lowering
an evaluation threshold cannot recover them. Tracktor's substituted confidence of
1 is not a calibrated detector score, and supplied SDP scores need not be calibrated
probabilities. Thresholds have different meanings across sources.

## Metric policy

This is a **detector diagnostic**, not COCO AP or official MOT HOTA/MOTA/IDF1.
It uses marked class-1 GT pedestrians. Each frame uses one-to-one matching that
maximizes the number of matches at the chosen IoU, then total IoU. Unmatched
predictions overlapping non-target annotations at that IoU are ignored, unless
they also overlap a target person: these duplicate candidates count as false
positives. All other unmatched predictions are FP; unmatched target people are FN.
Duplicate count is a subset of FP, not an additional error count. The ignore
policy differs from TrackEval's benchmark-specific distractor preprocessing.

Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN).
Combined rates are calculated from summed counts, not averaged video percentages.
Size/visibility recall includes all target GT, including missed people. Neither
small nor heavily occluded people are removed to improve a score. Ignore annotations
can make diagnostic precision optimistic relative to an official benchmark, so
compare like-for-like and still run `evaluate.py` for the final tracker.

## 2. Generate detections without shrinking images

```bash
python detect.py --backend rtdetrv2 --split val --input-mode original --min-score 0.1 --cache-dir detections/rtdetrv2_s_original_score010
python evaluate_detector.py --backend rtdetrv2 --cache-dir detections/rtdetrv2_s_original_score010
```

`original` preserves the original RGB pixels and aspect ratio. It adds black
padding only on the bottom/right to a multiple of 32. For example, 1920x1080
becomes 1920x1088. Resizing in the processor is disabled, anchors/position embeddings
are generated for the actual dimensions, predictions are mapped to the padded
canvas, and final boxes are clipped to the original image. No GT is used during
inference. Degenerate boxes entirely outside the image are discarded.

One image is processed at a time. This uses more memory/time than 640x640 and is
an experiment: pretrained weights may perform worse at a changed scale. If memory
is insufficient, try square 960 or 1280 in **separate** caches, e.g.:

```bash
python detect.py --backend rtdetrv2 --split val --input-mode square --image-size 960 --cache-dir detections/rtdetrv2_s_960_score010
python evaluate_detector.py --backend rtdetrv2 --cache-dir detections/rtdetrv2_s_960_score010
```

RT-DETR keeps its pretrained query count and official top-query postprocessing.
No added NMS or slicing is enabled. Full-resolution inference alone does not fix
all crowded-scene failures. Completed caches are skipped only if settings, coverage
and checksum match. Existing incompatible caches are rejected, not overwritten.
The old 640 cache format remains supported. Evaluation reads a cache's own recorded
recipe, so you do not need to alter active detector settings to compare experiments.

The config defaults remain `DETECTION_BACKEND="tracktor"` and
`DETECTOR_INPUT_MODE="square"`, preserving the trained model's data source.
The explicit commands above enable original-resolution detection for that process.
To permanently select it later, change the mode/cache/backend together.

## Parameters to tune, in order

| Parameter | First comparison | What to inspect |
| --- | --- | --- |
| `DETECTOR_INPUT_MODE` / `DETECTOR_IMAGE_SIZE` | square 640, square 960, original; optionally square 1280 | Recall for <50px people at similar precision, and cache FPS/memory. Use a new cache per recipe. |
| `DETECTOR_MIN_SCORE` | Export 0.05 if the 0.1 cache still misses low-confidence people | This is a storage floor, not a track birth threshold. Requires regeneration. |
| `DETECTION_MIN_SCORE` | Evaluate 0.05, 0.1, 0.15, 0.2, 0.3, 0.5 on the chosen low-floor cache | Choose recall with acceptable precision; do not maximize recall alone. The evaluator's `--scores` does this without editing config. |
| `NEW_TRACK_MIN_SCORE` | 0.3, 0.5, 0.7 after selecting a detection source | Tracking-only validation: prevent weak false detections from starting identities. Must be at least the read threshold. |
| `MATCH_THRESHOLD` | Recheck 0.02, 0.05, 0.1, 0.2 after retraining on the new source | HOTA/IDF1 on validation. Previous optimum can change with new detections. |

These are starting experiments, not established optimal values. Compare both
validation videos and IoU 0.75, not just the combined IoU-0.5 number. Higher recall
with many extra FP can reduce tracking accuracy. Keep IoU evaluation criteria fixed
rather than tuning them to raise the reported score.

If resolution/threshold changes do not improve the precision–recall tradeoff,
consider MOT detector fine-tuning on **training videos only** (optionally external
person data with no validation overlap). That needs a separate detector training
implementation; `train.py` trains the association model/its configured appearance
encoder, not RT-DETR box prediction. Larger model weights must also have a matching
`DETECTOR_MODEL_NAME` and pinned `DETECTOR_REVISION`.

## 3. Use a winning detector for association training

Generate the chosen cache for **both** train and validation using the same recipe
(`--split all` also generates configured test videos). Then set in `Utils/config.py`:

- `DETECTION_BACKEND="rtdetrv2"`, the winning cache path, input mode/size, cache floor
  and pinned model/revision.
- `DETECTION_MIN_SCORE` to the selected read threshold (or None to retain all).
- A fresh `RUN_NAME` and `RESUME_TRAINING=False`.

Train a fresh association model, then track/evaluate. Do not resume the existing
Tracktor-trained run against a changed detector source; checkpoint compatibility
checks intentionally reject that. Only then retune birth/matching settings on
validation. The existing code does not implement ByteTrack-style two-stage matching.

## Significant functions

- `detect.PersonDetector.predict`: preprocess in square/original mode, infer, map
  padded coordinates and clip; output zero-origin xyxy plus confidence.
- `Utils.detections.detector_settings`: distinct full-resolution recipe; preserves
  legacy square cache compatibility. `validate_cache` protects training inputs.
- `evaluate_detector.resolve_source`: validates cache completeness, source video,
  dimensions and checksum, records the saved recipe independently of training config.
- `evaluate_detector.score_frame`: one-to-one matches, ignore/duplicate handling,
  and GT-based size/visibility counts.
- `evaluate_detector.metrics`: calculate defined rates from summed counts.
- `evaluate_detector.write_reports`: Markdown/CSV/JSON/PNG outputs.
- `evaluate_detector.main`: configured labeled video selection and nested tqdm ETA.

References: [Transformers RT-DETRv2 documentation](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2)
and [official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR).
