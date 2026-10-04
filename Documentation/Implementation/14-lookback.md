# SDP detections and a shared 30-frame look-back

> Historical record: random frame-pair sampling described below has been
> superseded by [chronological motion/history training](16-state-training.md).
> The 30-frame boundary and supplied SDP detections remain unchanged.

Run `python train.py`, then `python tracking.py`, then `python evaluate.py`.
The new output folder is `outputs/mot17_SDP_lookback30/`. Train fresh weights;
old short-gap checkpoints do not satisfy the new protocol. Existing results stay intact.

## Settings and exact boundary

`LOOKBACK_FRAMES=30` means `current_frame - last_observed_frame <= 30`.
For example, an observation from frame 1 can match again at frame 31, but not 32.
The tracker's existing missing-frame counter is therefore configured as
`MAX_MISSED_FRAMES=LOOKBACK_FRAMES-1` (29). Previously, a missing-frame count
of 30 permitted a 31-frame observation gap. Edit LOOKBACK_FRAMES to change
the window; leave the derived MAX_MISSED_FRAMES expression intact.

`DETECTION_BACKEND="supplied"` and `DETECTION_FILE="det/det.txt"` select original
SDP boxes for the five training videos, two validation videos, and configured
test videos. RT-DETR files remain available but are not read. `detect.py` skips
inference in this mode, including when called by `run.sh`.

## Functions

- `train.load_frame_pairs`: reads full-video metadata and labels, using one current-frame anchor for each frame after the first.
- `train.select_frame_pairs`: picks a historical frame from the same video's previous 30 frames. Training samples uniformly at random each epoch. Validation cycles through gaps in a fixed order without consuming training RNG. Early frames use only available history. GT does not select the gap.
- `train.iter_graphs`: builds the historical/current graph and its separate loss targets. Frozen CPU features are cached for one window plus the current frame and cleared between videos. Each graph records `frame_gap` for monitoring, but it is not an added model input.
- `train._run_epoch`: shows the sampled gap in tqdm and records `mean_frame_gap` and `max_frame_gap` for each phase in history.csv.
- `tracking.main`: verifies the checkpoint's `temporal_sampling` protocol and maximum gap before inference. `run_sequence` records `lookback_frames` in its summary.

Validation evaluates one selected historical frame per current frame, covering
all gaps across a full video. It does not enumerate all 30 possible pairs at
every frame, which would multiply validation work. The selection repeats
identically each epoch, so checkpoint losses remain comparable within this run.

## What this fixes and what it does not

Training and validation now expose the association model to the same maximum
age of stale observations that tracking permits. They still use two-frame
detection pairs as stand-ins for track state. Predicted-track training rollouts,
mixtures of observation ages in one training graph, and false-detection rejection
are not added by this change. Tracking now supports the separately implemented
[motion prediction and appearance history](15-motion-and-history.md).
Lower validation loss still needs confirmation with held-out tracking metrics.
