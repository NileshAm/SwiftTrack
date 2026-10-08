# tracking.py — IDs, assignment, output, and demo

> **Current tracking additions:** [motion prediction and appearance history](15-motion-and-history.md) describe the Kalman filter, reliable feature memory, switches, and comparison output folders. Both are enabled by default.

> **Updated implementation:** use [DETR, attention, matching, and test videos](11-paper-components.md) for the current code, settings, and per-sequence output paths. The notes below record the earlier baseline and its checks; DenseNet defaults, fixed Sinkhorn, percentage splits, and old flat tracking paths are superseded. Current training and validation use separate complete video lists.

[Back to TODO](README.md) · [Implementation](../../tracking.py)

The tracker is implemented as `tracking.py` (the earlier plan called it `track.py`). Run in the project root:

```bash
conda activate dragon-track
python tracking.py
```

It loads `CHECKPOINT_PATH`, checks the saved preprocessing settings against the current config, and runs the selected clips without reading ground truth. The architecture comes from the checkpoint. Missing weights produce a message to run `python train.py` first.

## Settings

All settings are in [Utils/config.py](../../Utils/config.py):

| Setting | Current behavior |
| --- | --- |
| `SEQUENCES` | Same selected videos as training. |
| `TRACK_SPLIT` | `"val"` selects the next `VAL_FRACTION` after training; `"eval"` selects the remainder. The tracker starts with empty state at the start of this portion and preserves original frame numbers. |
| `TRACK_OUTPUT_DIR` | Defaults to `OUTPUT_DIR / "tracking" / TRACK_SPLIT`. Use a different folder to rerun; existing MOT files are protected. |
| `MATCH_THRESHOLD` | Provisional floor of `0.1` on the normalized matching score, applied before assignment. It is not a calibrated probability or a verified paper threshold. |
| `NEW_TRACK_MIN_SCORE` | Unmatched detections with detector confidence at least `0.5` start a track immediately. Existing tracks may match lower-confidence detections. |
| `MAX_MISSED_FRAMES` | Retain lost tracks through at most 30 missing observations. Ages use actual frame numbers. |
| `SHOW_PROGRESS` | `tqdm` bars show percentage, elapsed time, ETA, frames/second, current frame, detections, matches, births, retained tracks, and normalization error. |
| `SAVE_VIDEO` | False by default. True writes an MP4 while tracking. OpenCV is now installed in `dragon-track`. Saved results can also be rendered afterward with `make_evaluation_video`. |

Use a trained checkpoint and tune the matching/birth thresholds with `TRACK_SPLIT="val"` before selecting `"eval"` for final evaluation. Changing `TRACK_OUTPUT_DIR` does not change the checkpoint being loaded. Checkpoints from the old frame-range protocol or different percentage boundaries are rejected; train new weights in the new run folder. The tracker accepts runtime changes such as device, batch size, and Sinkhorn iteration count; feature/coordinate mismatches raise an error. Normalization remains approximate, so inspect the reported error.

## Functions

| Function | What it does |
| --- | --- |
| `match_tracks(scores, allowed_mask, unmatched_track_scores, unmatched_detection_scores, match_threshold)` | Solves CPU Hungarian assignment with one unmatched option per track and per detection. Returns matched row tuples and the two lists of unmatched row indices. |
| `update_tracks(tracks, matches, unmatched_tracks, unmatched_detections, frame, features, next_id)` | Updates boxes/features/confidence, ages missed tracks, removes expired ones, and creates eligible new tracks. Returns surviving tracks, the next unused ID, and visible observations. |
| `run_sequence(sequence, model, encoder)` | Loads detections/images, extracts valid crops, builds and scores each graph, assigns/updates tracks, and writes results. Runs with gradients disabled and models in evaluation mode. Starts IDs at 1 for each sequence. |
| `draw_tracks(image, observations)` | Returns an RGB image with coloured boxes and IDs using Pillow. Used for the saved preview and optional video. |
| `make_evaluation_video(sequence=None, split=None)` | Makes an MP4 from saved MOT results and source images without rerunning inference. Defaults to the first configured sequence and current split; use `split="eval"` for held-out frames. |
| `main()` | Validates settings/paths, loads checkpoint and compatible frozen encoder, then runs each configured sequence. |

## Assignment rule

For a track `t` and detection `d`, the matching cost is `-log(scores[t,d])`. Keeping the two items unmatched costs `-log(unmatched_track_scores[t]) + -log(unmatched_detection_scores[d])`. A match must pass the candidate mask and threshold and be strictly cheaper than leaving those two items unmatched. The solver then finds the minimum total cost with one-to-one constraints and explicit dummy rows/columns. This chooses the best combined assignment; it does not independently take each row's largest score.

Zeros are clipped to `1e-12` only when taking logarithms. Forbidden, zero-score, and below-threshold matches remain excluded. Empty sides, rectangular matrices, and fully forbidden masks are supported. This score/cost convention is our partial-reproduction design choice and still needs validation-based tuning.

## Track lifetime

Each track stores `id`, `box`, `feature`, `score`, `last_seen`, and `missed`. Matched tracks take the current observation and reset `missed` to zero. Unmatched tracks keep their last observed box/feature and age. There is no motion prediction, smoothing, or appearance-history average yet.

Before matching, tracks whose gap already exceeds the permitted missing observations are removed. After matching, unmatched tracks expire when `current_frame - last_seen > MAX_MISSED_FRAMES`. For example, with a limit of 1, an observation at frame 1 can reconnect at frame 3 after missing frame 2; it cannot reconnect at frame 4 after missing both frames 2 and 3. Removed IDs are never reused within that sequence.

Only current matched/new observations are exported. Lost tracks remain candidates but have no invisible predicted boxes in the output. Crop filtering is applied consistently to boxes, confidence, row IDs, and features.

## Output files

Each sequence produces files under `TRACK_OUTPUT_DIR`:

- `<sequence>.txt`: MOT rows `frame,id,x,y,width,height,confidence,-1,-1,-1`. IDs are positive, frame IDs are original, box origin is restored exactly once, and confidence is the detector confidence.
- `<sequence>_preview.jpg`: the final selected frame with observed boxes/IDs.
- `<sequence>_summary.json`: frame/row/ID counts, elapsed time, FPS, checkpoint path, thresholds, resolved original frame range, split, dataset location, completion flag, and maximum normalization error. `evaluate.py` verifies this metadata before scoring.
- `<sequence>.mp4`: optional when `SAVE_VIDEO=True` and OpenCV is installed.

FPS covers the frame loop, including image reads, feature extraction, graph/model/assignment work, MOT writes, and optional video writes. It excludes model/metadata loading and final preview saving. Small CPU checks are not benchmark speed measurements. If a run fails midway, its partial MOT file remains; choose a fresh output directory for the next run.

## Checked

Using the existing trained checkpoint on CPU, a 10-frame MOT17-05 validation clip produced 48 observations, 10 created IDs, finite positive boxes, and no duplicate `(frame,id)` rows. A fresh Python process reloaded the checkpoint and produced byte-identical MOT output. Temporary checks compared assignment against exhaustive solutions on 240 small rectangular/masked/empty cases, checked unmatched choices, and exercised missed-frame aging, gap reconnection/expiry, coordinate conversion, and monotonic IDs. Training, validation, feature extraction, and the learning demo also passed short runs with their new `tqdm` bars. No permanent test files were added.

Threshold calibration, full held-out tracking metrics, and real MOT20 verification remain pending. These checks establish working code, not identity accuracy.

## Make a video from saved results

In Python, from the project folder:

```python
from tracking import make_evaluation_video

make_evaluation_video("MOT17-02-FRCNN", split="eval")
```

Run tracking for the requested split first. The function reads the saved MOT rows and completion summary, checks the frame range, draws boxes/IDs with stable colours, and writes `<sequence>.mp4` beside the tracking text file. It preserves source FPS, resolution, original frame order, and frames without observations. A `tqdm` bar shows rendering progress. Existing videos are protected from overwrite.

With no arguments, it uses the first configured sequence and current `TRACK_SPLIT`; this may be `"val"`. To render every configured sequence, loop over `config.SEQUENCES` and call the function for each. No GPU, model loading, ground-truth labels, or metric rerun is needed.

Verified by decoding all 60 frames of the current MOT17-02 validation video at 1920×1080 and 30 FPS. The held-out rendering path also passed on the temporary two-frame eval fixture at 30 FPS.
