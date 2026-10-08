# evaluate.py, comparisons, and final outputs

> **Updated implementation:** use [DETR, attention, matching, and test videos](11-paper-components.md) for the current code, settings, and per-sequence output paths. The notes below record the earlier baseline and its checks; DenseNet defaults, fixed Sinkhorn, percentage splits, and old flat tracking paths are superseded. Current training and validation use separate complete video lists.

[Back to TODO](README.md)

## evaluate.py

[evaluate.py](../../evaluate.py) is implemented using [TrackEval](https://github.com/JonathonLuiten/TrackEval), installed as version 1.3.0 in `dragon-track` without upgrading the existing numeric/PyTorch packages. Its HOTA, CLEAR, and Identity implementations compute HOTA, IDF1, MOTA, identity switches, and related metrics. MOT pedestrian preprocessing remains enabled; no custom metric formulas are used.

Run in the project root:

```bash
conda activate dragon-track
python train.py
python tracking.py
python evaluate.py
```

Start with `TRACK_SPLIT="val"` in `Utils/config.py`. Training uses the first `TRAIN_FRACTION=0.8` of every video in `SEQUENCES`; validation uses the next `VAL_FRACTION=0.1`; the remainder is reserved for evaluation. For 600 frames: train 1–480, validation 481–540, evaluation 541–600. Once tuning is finished, change `TRACK_SPLIT` to `"eval"` and rerun tracking/evaluation. Their default output directories include the split name, keeping both results separate. No local scores can be produced for test data without labels.

The new protocol uses `RUN_NAME="mot17_temporal_split"`. Retrain for this protocol; old checkpoints/output summaries are not silently treated as compatible. These are temporal portions of the same videos, not unseen-video results.

| Function | Behavior |
| --- | --- |
| `copy_mot_slice(source, destination, frame_range, is_gt=...)` | Validates numeric MOT rows, keeps only selected GT rows, requires prediction rows to be inside that same range, and subtracts `first_frame-1` from both copies. Checks positive/unique prediction IDs and positive box sizes. |
| `prepare_inputs(output_dir)` | Requires completed tracking summaries with the exact split, frame coverage, dataset location, and matching row counts. Groups MOT17/MOT20 separately and writes evaluator input copies. |
| `run_trackeval(output_dir, groups)` | Calls TrackEval on the copied data with explicit sequence lengths, MOT preprocessing, HOTA/CLEAR/Identity metrics, and its sequence progress bar. Keeps native metric summaries, detailed CSV files, and logs. |
| `main()` | Checks settings, prepares inputs, calls the evaluator, and writes a readable results table plus run metadata. |

Ground truth is copied in its original MOT format, including mark/class/visibility fields. The training label filter is **not** applied; TrackEval handles ignored/distractor objects itself. Empty prediction files are valid and count as missed detections, provided the completed tracking summary proves every selected frame was processed. Old, partial, wrong-range, or mismatched tracking output is rejected instead of being scored against full-video GT. Original GT, detections, and predictions remain unchanged.

`EVALUATION_OUTPUT_DIR` defaults to `outputs/<run_name>/evaluation/<val-or-eval>/`. It contains:

- `results.md`: selected original ranges and combined HOTA, IDF1, MOTA, IDSW values; the first three use TrackEval's percentage formatting, while IDSW is a count.
- `<MOT17-or-MOT20>/DragonTrack/pedestrian_summary.txt` and `pedestrian_detailed.csv`: native combined/per-sequence metrics, aggregated by TrackEval.
- `trackeval.log`: evaluator version/path, native tables, and diagnostic output; `trackeval_errors.log` is created if evaluation fails.
- `run_info.json`: fractions, resolved ranges, tracking summaries, version, and reported metrics.
- `inputs/`: the cropped/rebased evaluator copies, retained for inspection. No images are duplicated.

Use a new `EVALUATION_OUTPUT_DIR` to retain another successful run. A failed evaluation can be retried in the same folder until `results.md` is written. `TRACKEVAL_ROOT=None` uses the installed package; a source checkout can be supplied instead if it supports the same API. TrackEval may log a missing optional BURST/pycocotools notice; that dataset is not used for MOT box evaluation.

**Verification:** a temporary copy of 20 real MOT17-02 frames passed the full CPU pipeline: train 1–16, validate/track/evaluate 17–18. A fresh process reloaded the checkpoint to track/evaluate the remaining frames 19–20. A controlled two-frame perfect tracker scored HOTA/IDF1/MOTA=100 and IDSW=0, including correct distractor removal; empty predictions scored zero. Checks also covered uneven rounding, duplicate/out-of-range rows, wrong coverage metadata, source preservation, and rebasing. These checks establish code/alignment correctness; the full configured dataset run remains to be performed.

## ByteTrack and BoT-SORT

Start setup while training runs. Use their existing code and only the small input conversion necessary to supply the same detections, real scores, and frames. Run one sequence first. Record implementation revision, thresholds, appearance model, and exact command in `results.md`. No generic adapter classes or comparison runner are needed.

Evaluate outputs with TrackEval. Include appearance extraction in comparable supplied-box timing; a system running its own detector is a separate comparison. Time-box dependency trouble and explain any failed or omitted baseline rather than presenting missing values as zero.

## Final delivery

In `outputs/<run_name>/`, keep the checkpoint, settings used, MOT files, TrackEval output, demo video, and one **`results.md`** with actual commands, data splits/detection source, training duration, metrics, speed/memory, method differences, and known failures. A manually written result table is sufficient. Keep large outputs out of ordinary source commits.

Use `tracking.make_evaluation_video(sequence, split="eval")` to render saved evaluation results with boxes and IDs after tracking. See the [video instructions](06-assignment-and-tracker.md#make-a-video-from-saved-results). Turn video off for inference timing. Restart Python, reload the checkpoint, and rerun a clip before delivery. Add the actual working Conda/run instructions to the root README. Skip optional algorithms if any of this remains unfinished.
