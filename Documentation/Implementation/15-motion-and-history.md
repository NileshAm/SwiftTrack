# Motion prediction and appearance history

> **Current settings and architecture:** see [paper alignment](17-paper-alignment.md). The notes below describe earlier development stages; frozen-encoder defaults, previous loss/scoring layers, and old run names are superseded. `main.py` now plots saved run metrics.

[Back to TODO](README.md) · [Tracker](../../tracking.py) · [Config](../../Utils/config.py)

These utilities now run during training, validation loss, and tracking. See
[chronological state training](16-state-training.md) for the training protocol
and fresh run. This guide describes the filter/history behavior and inference
comparisons. No new packages are needed.

## Settings and output

| Setting | Default | Meaning |
| --- | --- | --- |
| `USE_MOTION_PREDICTION` | `True` | Predicted boxes supply graph geometry. |
| `USE_APPEARANCE_HISTORY` | `True` | Accumulated features supply track appearance. |
| `APPEARANCE_HISTORY_SIZE` | `30` | Maximum reliable observations retained per ID, not frames. |
| `APPEARANCE_HISTORY_MIN_SCORE` | `0.5` | Minimum detector confidence for adding a feature. |
| `TRACKING_VARIANT` | `"motion_history"` | Tracking/evaluation folder label; checkpoint unchanged. |

Results go under `outputs/<RUN_NAME>/tracking/motion_history/val/<sequence>/`
and `tracking/motion_history/test/<sequence>/`. Evaluation and its plots go under
`outputs/<RUN_NAME>/evaluation/motion_history/val/`. Videos use the same sequence
folders. Summary JSON records actual settings and method versions; evaluation
retains those summaries.

For comparisons, keep the checkpoint, detections, thresholds, and videos fixed.
Run both switches off (`TRACKING_VARIANT="baseline"`), motion only
(`"motion_only"`), then both (`"motion_history"`). Edit config before starting
Python. The folder label does not set switches automatically. Existing outputs
are protected. These switches alone require neither retraining nor a new RUN_NAME.

## `Utils/motion.py`

`MotionState(box, frame_id)` creates one Kalman filter per new ID, with zero
initial velocity. Its eight values are center x/y, width/height, and their
per-frame velocities. Boxes use zero-origin xyxy. State calculations use float64;
returned graph boxes use float32.

- `predict(frame_id)` advances by actual elapsed frames, increasing uncertainty
  through missing detections. Repeated calls at the same frame are safe;
  earlier frame IDs are rejected.
- `update(box, frame_id)` corrects the prediction with an associated detection.
  Successive observations teach velocity. Joseph-form covariance updates help
  preserve numerical stability.
- `box()` returns a new predicted/filtered xyxy array without advancing time.

Per-frame position and measurement noise use height/20; velocity noise uses
height/160. Initial uncertainties use height/10 for position/size and height/16
for velocity. These are starting heuristics, not learned or verified paper
parameters. Width/height stay at least one pixel. Predicted centers may move
outside the image; clipping them would distort motion.

## `Utils/appearance_history.py`

`AppearanceHistory(max_size, min_score)` keeps a bounded queue per ID.
`update(feature, score)` copies reliable features to CPU, normalizes each, and
weights it by detector confidence capped at one. Low-confidence and zero
vectors are skipped. `feature()` returns the normalized weighted mean, or None
before any accepted sample. If vectors cancel, it uses the latest accepted
vector. This is a recent-observation mean, not an exponential moving average.

Low-confidence matches can keep an ID alive without changing memory. If the
birth threshold permits an initial feature below the history threshold, that
feature is a fallback until a reliable match arrives. Unmatched frames add
nothing. Expiring a track discards its memory.

## Integration in `tracking.py`

Lifecycle helpers now live in `Utils/track_state.py` and are imported by both
tracking and chronological training, so they share one implementation.

1. Expire tracks outside the observation window.
2. `predict_track_boxes()` advances surviving filters to the current frame,
   even when it has no detections.
3. Build the graph with predicted boxes and history appearance vectors.
4. Apply existing learned scores, Sinkhorn, and Hungarian assignment.
5. `update_tracks()` corrects motion and updates history for matches and births.
6. Export actual observed detection boxes/scores to MOT/video.

The measured `track["box"]` stays separate from `track["motion"]`. Predictions
never reset `last_seen`, extend track lifetime, or create output detections.
Reconnection at a 30-frame gap and expiry beyond it are unchanged. Both switches
off restore last-observed box/feature matching. `tracker_settings()` validates
and records inference options independently of model weights.

## Limits and verification

Training/validation now uses predicted track states with GT-guided past updates;
it does not simulate wrong past assignments. There is no motion gating, camera-motion compensation,
interpolation, or extra association fallback. The graph still decides identities.
Sudden motion or incorrect matches can harm the filter/history. Compare full
validation HOTA/IDF1 before claiming an improvement.

Checks use temporary scripts/fixtures rather than permanent test infrastructure:
constant-velocity gap prediction, box/covariance validity, bounded memory,
confidence rejection, empty frames, expiry, independent IDs, disabled switches,
and a short real-data checkpoint/tracking/evaluation run.
