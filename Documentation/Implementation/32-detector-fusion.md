# Tracktor-first detector fusion

Run `python detector_fusion.py` in the dragontrack environment from the project
root. Default input is outputs/detector_suite_v1. Existing files are read only;
each run creates a timestamped fusion directory.

Default sweep: RT-DETR confidence 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7 crossed
with overlap thresholds 0.3, 0.5, 0.7 (21 settings). All Tracktor boxes remain.
Additional proposals are considered in descending confidence, rejecting those
overlapping baseline or previously accepted boxes. Confidence is never averaged.
GT is used only to evaluate constructed detections.

Options: `--source yolo26_s`, `--source rfdetr_s`, `--scores .25 .4 .6`,
`--overlaps .3 .5`, `--max-frames 30`, `--suite-dir PATH`, `--output-dir NEW_PATH`.

Read report.md and results.csv. Compare extra TP and FP against baseline as well
as F1 and recall. summary.json records inputs and settings. COMPLETE.json marks
a finished sweep. Each setting exports validation-only MOT-format detection
files, with one-based x/y and width/height. These are not registered training
caches. Full tracking integration remains subsequent work; do not change config
to train on these validation exports. No HOTA/IDF1 improvement is established by
this diagnostic. Label selected settings as validation-tuned.

Verified using six real validation frames and an isolated preservation/duplicate
suppression/threshold check. Full-video results must decide whether to proceed.
