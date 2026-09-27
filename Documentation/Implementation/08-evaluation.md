# evaluate.py, comparisons, and final outputs

[Back to TODO](README.md)

## evaluate.py

- **`run_trackeval(config)`** calls the existing TrackEval implementation using the selected dataset, GT, sequence list, and MOT tracker output. Save its command/version, logs, and raw metric output. Keep MOT preprocessing enabled as appropriate.
- **`main()`** reads settings, checks required paths, calls the evaluator, and prints where HOTA, IDF1, MOTA, and identity-switch results were written. Use existing evaluator summaries; no custom metric formulas or report parser is required initially.

Prefer whole held-out sequences to simplify file layout. If using frame ranges, align GT and tracker ranges/indexing in a temporary evaluation copy without changing original data. Do not evaluate a short clip against full-sequence GT. Report MOT17/MOT20 separately and use TrackEval's aggregation. No local scores can be produced for test data without labels.

**Quick check:** the evaluator reads a short output without format errors; inspect a few frame/ID/box rows against the source images. Then evaluate the fixed held-out selection and keep the actual evaluator files.

## ByteTrack and BoT-SORT

Start setup while training runs. Use their existing code and only the small input conversion necessary to supply the same detections, real scores, and frames. Run one sequence first. Record implementation revision, thresholds, appearance model, and exact command in `results.md`. No generic adapter classes or comparison runner are needed.

Evaluate outputs with TrackEval. Include appearance extraction in comparable supplied-box timing; a system running its own detector is a separate comparison. Time-box dependency trouble and explain any failed or omitted baseline rather than presenting missing values as zero.

## Final delivery

In `outputs/<run_name>/`, keep the checkpoint, settings used, MOT files, TrackEval output, demo video, and one **`results.md`** with actual commands, data splits/detection source, training duration, metrics, speed/memory, method differences, and known failures. A manually written result table is sufficient. Keep large outputs out of ordinary source commits.

Use `track.py::draw_tracks` and video writing for the demo. Turn video off for timing. Restart Python, reload the checkpoint, and rerun a clip before delivery. Add the actual working Conda/run instructions to the root README. Skip optional algorithms if any of this remains unfinished.
