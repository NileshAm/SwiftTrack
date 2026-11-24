# Tracktor versus YOLOX tracking diagnostics

Run from the project root in the `dragon-track` environment:

```bash
python diagnose-tracking.py
```

The script uses the completed Tracktor and YOLOX validation runs by default. To compare different completed runs, pass `--tracktor-run PATH` and `--yolox-run PATH`. Each run must contain TrackEval MOT17 validation inputs for `MOT17-02-SDP` and `MOT17-09-SDP`. Use `--preview-count 0` to skip image previews or `--output-dir PATH` to choose a fresh output directory.

The new `outputs/tracking_diagnostics/<timestamp>/` folder contains:

- `report.md` and `report.json`: tracking metrics and ground truth assisted identity results.
- `frames.csv`: false positives, missed detections, identity switches, and mean matched IoU for every frame of MOT17-02-SDP.
- `frames/`: the same source frame with ground truth boxes in green, Tracktor boxes in cyan, and YOLOX boxes in orange. Frames with the largest YOLOX excess of false positives and identity switches are selected first.

The **Association on correctly detected boxes** table measures tracker identity consistency with detection errors removed from its denominator. For each sequence, boxes are independently matched to ground truth at IoU 0.5. A global one-to-one renaming of track IDs to person IDs then counts how many matched boxes keep a consistent identity. The table also counts people assigned just one track ID and identity changes after a gap of one or more unmatched frames. This isolates association more directly than overall HOTA or IDF1, but it still depends on which boxes the tracker emitted and which people were detected.

Frame counts reproduce TrackEval's MOT17 preprocessing and CLEAR identity switch rules at IoU 0.5. Box jitter is the change in detected box center error relative to the ground truth center between consecutive matched frames. A large value can reflect unstable box localization or a rapidly changing localization error; it is not a direct measure of camera motion.

For the association diagnostic, the script keeps each tracker's **emitted boxes and frame coverage** and replaces matched boxes' identity labels with their matched ground truth identity. Unmatched boxes receive unique labels. The resulting score estimates how much identity assignment could improve on those fixed boxes. It is an analysis using validation ground truth, cannot be used as a deployed tracker, and is not a formal maximum across every HOTA IoU threshold. CLEAR matching favors identity continuity, so changing IDs can slightly change its TP/FP/FN counts even though the boxes have not moved. The experiment does not measure the ceiling of the raw detector cache before tracker filtering.
