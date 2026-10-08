# Training and evaluation plots

Implemented in `Utils/plots.py`; uses the existing Matplotlib and NumPy packages.
No extra training, detector inference, or TrackEval run is needed to plot saved results.
The interactive box/graph demonstrations remain in `Utils/plot.py`.

## How to run

With `SAVE_PLOTS=True` in `Utils/config.py` (the default):

- `python train.py` refreshes training PNGs after each completed epoch, after saving weights/history.
- `python evaluate.py` saves evaluation PNGs after TrackEval finishes and links them in `results.md`.
- `python main.py` regenerates available plots for `OUTPUT_DIR` and `EVALUATION_OUTPUT_DIR`.

For an older run, without editing config:

```bash
python main.py --run-dir outputs/mot17_attention
```

An optional `--evaluation-dir path/to/evaluation/val` selects a custom evaluation folder.
Missing stages are reported and skipped. Existing chart PNGs are replaced; input CSVs,
checkpoints, tracking output, and metrics are not changed. No window opens; open the PNGs.
`SAVE_PLOTS=False` disables automatic plotting but still allows `python main.py`.

## Where the charts go and what they mean

| Output under the run folder | What to look for |
| --- | --- |
| `plots/training/losses.png` | Training/validation total, BCE, contrastive, and assignment loss. The dashed line marks the earliest lowest validation total loss among completed epochs, matching checkpoint selection. Falling training loss with persistently rising validation loss can suggest overfitting. |
| `plots/training/timing.png` | Minutes per epoch and cumulative training/validation hours. Excludes model setup, checkpoint writing, and plotting. |
| `plots/training/supervision.png` | Used/skipped frame-pair graphs and positive/negative cross-frame links used for BCE. Log-like scale keeps zero and large counts visible. The counts use different units and are not accuracy. |
| `evaluation/val/plots/<benchmark>/overall_metrics.png` | Native combined HOTA, DetA, AssA, IDF1, MOTA, detection recall, and precision. Higher is better. MOTA can be negative. |
| `evaluation/val/plots/<benchmark>/per_sequence_metrics.png` | Find which videos have weak tracking, detection, or association scores. |
| `evaluation/val/plots/<benchmark>/tracking_errors.png` | Missed people (FN), extra detections (FP), and identity switches (IDSW), as counts and errors per 100 GT detections. Lower is better. Normalized errors can exceed 100; zero-GT rates are undefined and left blank. |
| `evaluation/val/plots/<benchmark>/diagnostics.png` | Combined HOTA/DetA/AssA versus localization IoU threshold, plus per-video detection precision/recall operating points. These are not confidence threshold sweeps or a precision-recall curve. |

HOTA/DetA/AssA use TrackEval's mean over localization thresholds (`___AUC`).
Detailed CSV rates are fractions and are converted to percentages once. Combined
values come directly from TrackEval, not an unweighted average across videos.
Tracking errors and precision/recall use TrackEval's CLEAR matching results.

Training samples negatives and historical gaps; validation uses all eligible
identity pairs for deterministically selected gaps in the same 30-frame window.
Older histories used adjacent-frame validation; plots retain the appropriate label.
Their losses are useful trends but are not measured on
identical pair distributions. Lower loss does not guarantee higher HOTA or IDF1.
Tracking metrics are from the evaluated checkpoint only, not from every epoch.
Validation is used for selection/tuning, so these charts do not establish final
test performance. Old runs retain their original evaluation protocol; plotting
them does not turn temporal validation into whole-video validation.

## Functions for developers

- `plot_training(history_path, output_dir=None)`: read completed history rows and return three saved PNG paths. Older histories without assignment loss show an unavailable panel.
- `plot_evaluation(evaluation_dir)`: find each benchmark's native `DragonTrack/pedestrian_detailed.csv` and return four PNG paths per benchmark. No model, images, or GT are loaded.
- `save_run_plots(run_dir, evaluation_dir=None)`: plot whichever completed stages are present; used by `main.py`.
- `auto_plot(stage, path)`: used by training/evaluation after saving primary outputs; reports common chart errors so saved results remain usable.

The helpers use Matplotlib's noninteractive canvas without changing the global
backend. Figures are cleared after saving, avoiding accumulated GUI windows
while training. All operations are small compared with an epoch; no additional
progress bar is needed.
