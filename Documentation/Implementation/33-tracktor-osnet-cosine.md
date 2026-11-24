# Tracktor + frozen OSNet with direct cosine matching

This is the detector-control run for `track-yolox-osnet.py`. Both runners use the
same OSNet encoder, cosine matching, center-distance gate, Kalman motion,
appearance history, confirmation, and TrackEval protocol. No GCN, learned
Sinkhorn matcher, association checkpoint, or training is used. Only the input
boxes differ. The Tracktor cache contains author-linked preprocessed boxes with
placeholder confidence 1.0, so its score thresholds cannot be interpreted as
calibrated detector confidence.

From the project directory:

```bash
conda activate dragon-track
python track-tracktor-osnet.py --smoke-test --detection-min-score 0.5 --new-track-min-score 0.7
python track-tracktor-osnet.py --detection-min-score 0.5 --new-track-min-score 0.7
python evaluate-tracktor-osnet.py outputs/tracktor_osnet_cosine/RUN_FOLDER
```

The tracking command prints its new timestamped `RUN_FOLDER`. The evaluation
script reads that completed run and verifies its Tracktor cache path, hash, and
direct matcher before invoking the existing TrackEval implementation. You may
instead pass `--evaluate` to the tracking command to run both steps together.
The script leaves `Utils/config.py` unchanged.

The matched YOLOX comparison uses:

```bash
python track-yolox-osnet.py --evaluate --detection-min-score 0.5 --new-track-min-score 0.7
```

The first completed matched run is in
`outputs/tracktor_osnet_cosine/matched_yolox_050_070`. Its results were 44.734
HOTA, 51.110 IDF1, 50.326 MOTA and 222 ID switches. The existing YOLOX
0.5/0.7 run gave 42.045 HOTA, 53.803 IDF1, 49.147 MOTA and 505 switches.
These are local validation results, selected on repeatedly used MOT17 videos.

`Utils/yolox_osnet.py` supplies the exact cosine/Hungarian matching function
shared by both runners; `Utils/track_state.py` supplies motion and history.
