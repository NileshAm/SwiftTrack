# Train with motion prediction and appearance history

> **Current settings and architecture:** see [paper alignment](17-paper-alignment.md). The notes below describe earlier development stages; frozen-encoder defaults, previous loss/scoring layers, and old run names are superseded. `main.py` now plots saved run metrics.

[Back to TODO](README.md) · [Training](../../train.py) · [Settings](../../Utils/config.py)

Training and validation loss now process videos chronologically. Each graph
compares live track states against current detections. Track states use the
same Kalman filter, reliable appearance history, birth threshold, and expiry
rules as tracking. This replaces random two-frame historical sampling.

## Run

Use the existing `dragon-track` environment:

```bash
python train.py
python tracking.py
python evaluate.py
```

`RUN_NAME="mot17_SDP_state30"` selects a fresh run. Existing weights/results are
preserved. Train fresh weights to learn from these inputs. Both
`USE_MOTION_PREDICTION=True` and `USE_APPEARANCE_HISTORY=True` now affect training,
validation loss, and tracking. The history size/confidence settings are shared.
No new dependencies or permanent test framework are required.

Weights/history go under `outputs/mot17_SDP_state30/`. Tracking goes under
`tracking/motion_history/val/` and `tracking/motion_history/test/`; evaluation
goes under `evaluation/motion_history/val/`. These paths are inside that run.

## What happens at each frame

1. Start a fresh memory at each video's first frame. Its detections seed tracks;
   the first scored frame is the second frame.
2. Remove observations older than `LOOKBACK_FRAMES` (30), then predict each
   surviving track's position at the current frame using past observations.
3. Extract current detection features. Build the graph from predicted track
   boxes and accumulated track features versus current detections.
4. Build loss targets separately from known identities. Forward/backward uses
   the graph, not identity labels as inputs.
5. Only after scoring/loss, update motion/history with this frame's detections.

Different people can have different last-seen frames in one graph. Missing
detections naturally produce gaps up to 30 frames; gaps are no longer sampled
uniformly. A gap of exactly 30 can reconnect. A gap of 31 starts a new track.
Empty frames still predict/age tracks, and memory is bounded per live track.
Every image's frozen features are extracted once per video per epoch.

## Teacher forcing and remaining difference from inference

After scoring, training and validation-loss memory use known identities to
choose correct state updates. This is **teacher forcing**. GT boxes are only
used to label detections by IoU; motion and appearance receive detection boxes
and image features. No future observation is used. Current-frame labels cannot
update the track inputs for their own loss.

Actual validation/test tracking still uses predicted Hungarian assignments
and reads no GT. Consequently, incorrect past associations are not simulated
during training. Loss validation is deterministic and uses no optimizer, but
is still easier than actual tracking; it must not be reported as tracking
accuracy. Full TrackEval results remain the accuracy check.

Unknown/ignored detections retain label -1. They can create unsupervised tracks
under the same confidence rule as inference, but -1 never matches another -1.
They compete in graphs and expire normally. This retains the existing loss
policy of excluding unknown identities, not explicit false-positive training.

## Files and functions

| File/function | Responsibility |
| --- | --- |
| `Utils/motion.py`, `Utils/appearance_history.py` | Existing per-ID filter and bounded appearance memory. |
| `Utils/track_state.py` | Shared `expire_tracks`, `predict_track_boxes`, and `update_tracks`; no GT access. Moved here from tracking so both paths call identical code. |
| `Utils/track_state.py:state_settings` | Validate/save the motion/history/lifetime recipe, excluding output folder names. |
| `Utils/training_state.py:TrainingTrackState.inputs` | Expire/predict past state; return graph arrays and separate supervision labels/ages. |
| `Utils/training_state.py:TrainingTrackState.update` | Apply GT-guided matches after scoring, including births and unlabelled tracks. |
| `train.py:prepare_frame` | Extract and filter features, boxes, scores and labels together. |
| `train.py:iter_graphs` | Seed/reset video memory, yield past-only inputs, then ingest the current frame when iteration resumes. |
| `train.py:_run_epoch` | Existing losses/optimizer plus track-age/history statistics and frame progress bars. |

`load_frame_pairs` keeps its existing name, but adjacent pairs now describe
chronological frame order only. `select_frame_pairs` and its random-gap feature
cache are removed. The original public tracking helpers remain importable
from `tracking.py` through shared utility imports.

## Checkpoints, progress, and plots

Checkpoints record `temporal_sampling.protocol="chronological_track_state_v1"`
and the shared state recipe. Tracking accepts this protocol and older compatible
30-frame checkpoints for comparison. It reports differing state settings when
an intentional inference-only comparison changes the recipe. Folder labels
alone do not affect compatibility. Older weights do not become state-trained
simply by changing configuration.

Progress shows maximum track age and average retained history length in the
current graph. CSV fields `mean_frame_gap` and `max_frame_gap` now describe track
ages instead of sampled frame-pair gaps; `mean_history_size` records accepted
appearance observations averaged over live tracks, then graphs. Empty track
sets contribute zero. `used_pairs`/`skipped_pairs` retain their CSV names but
count scored/skipped frame graphs. Plots label the new training protocol.

## Verification and accuracy work remaining

Temporary checks cover matching state parity with inference, no current-frame
input leakage, mixed ages, gap-30 reconnection/gap-31 expiry, video reset, unknown
IDs, confidence filtering, empty graphs, finite gradients, and deterministic
validation with unchanged weights. A short real SDP run exercises training,
validation, checkpoint reload, tracking, TrackEval and plots.

The full five-video training and two-video tracking evaluation are still needed
to measure accuracy. No gain over the previous model is claimed by these checks.
