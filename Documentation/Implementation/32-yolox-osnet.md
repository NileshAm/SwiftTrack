# Pretrained YOLOX-X + OSNet tracking, without GCN

This is a separate tracking baseline. It reads **only** the official pretrained
CrowdHuman + MOT20 YOLOX-X detection cache, extracts frozen MSMT17 OSNet person
features, and matches tracks to detections by cosine similarity with a motion
position gate. It uses the existing motion and appearance-history track state,
but does not load or train a graph model, attention, Sinkhorn matcher, or
association checkpoint. The project-wide `Utils/config.py` stays unchanged.

## Run on the configured validation videos

The default cache is the already exported full-video pretrained YOLOX cache.
From the project directory:

```bash
conda activate dragon-track
python track-yolox-osnet.py --smoke-test
python track-yolox-osnet.py --evaluate
```

`--smoke-test` checks two frames and writes no tracking results. The full run
tracks both `VAL_SEQUENCES` and then runs the existing TrackEval HOTA/IDF1/MOTA
evaluator. A fresh timestamped folder under `outputs/yolox_osnet/` contains
tracks, previews, summaries, and evaluation results. To save tracking videos,
add `--save-video`. The run does **not** run YOLOX inference; it reads its
precomputed complete-video cache.

If you need a new cache, export the **pretrained** detector without `--checkpoint`:

```bash
python detect-yolox.py --split val --cache-dir detections/yolox_x_mot20_val_v1
```

The tracker verifies each cache file hash, full-video identity, and official
pretrained checkpoint hash. It rejects the fine-tuned `best.pth` cache, even
though the current best fine-tuning epoch happens to be epoch 0.

## Thresholds

```bash
python track-yolox-osnet.py --evaluate --similarity 0.7 --new-track-min-score 0.5 --detection-min-score 0.1
```

- `--similarity`: minimum OSNet cosine similarity for an existing track match.
- `--new-track-min-score`: minimum YOLOX confidence to start a new track.
- `--detection-min-score`: read-time YOLOX confidence cutoff; it cannot be below
  the cache's score floor (0.1 in the default cache).
- `--center-distance`: motion gate in person heights, with a 25-pixel minimum.

Tune these on validation, using a fresh output directory for each run. The
matcher permits a detection to match only one track. It does not treat its
cosine score as the learned graph model's match probability. Motion prediction,
30-frame look-back, appearance history, and track confirmation use the current
config settings. Changing those settings affects this baseline too.

To run on `TEST_SEQUENCES`, first export their separate pretrained cache with
`python detect-yolox.py --split test --cache-dir detections/yolox_x_mot20_test_v1`,
then run `python track-yolox-osnet.py --split test --cache-dir detections/yolox_x_mot20_test_v1`.
Test videos have no GT in this workflow, so `--evaluate` is validation-only.

The official YOLOX checkpoint was validated upstream on MOT17 train. The
configured MOT17 validation videos are therefore useful for local comparison,
but do not provide an untouched independent benchmark of that pretrained model.
Compare tracking variants on the same videos and detector boxes.

## Files

- `track-yolox-osnet.py`: tracking runner and optional evaluation.
- `Utils/yolox_osnet.py`: pretrained-cache verification and direct matching.
- `Utils/reid.py`: frozen MSMT17 OSNet encoder.
- `Utils/track_state.py`: motion and appearance-history lifecycle.
- `evaluate.py`: existing full-video TrackEval metrics.
