# Final five-epoch MOT17 experiment

Run from the project root in the `dragon-track` Conda environment. Preview all commands first:

```bash
python final-train.py
```

Start the full progression only when ready:

```bash
python final-train.py --run
```

In another terminal, request a pause during any stage:

```bash
python final-train.py --pause
```

After the runner reports `paused`, continue the same recipe:

```bash
python final-train.py --run --resume
```

The interrupted epoch starts again from the last completed checkpoint. Completed stages are skipped. An interrupted detector export verifies completed videos and generates the remaining ones. An interrupted threshold trial reruns only that trial. The parent progress bar reports stage progress and ETA; each trainer shows its own batch/epoch progress. `run.json` records the active stage and exact command.

Every trainable model gets five training epochs by default. The authors' baseline uses its pinned CUDA/DataParallel implementation. Detector and OSNet tuning begin from official pretrained weights; association models begin with new weights. The five training and two validation MOT17 videos are read from `Utils/config.py`. All generated detection files and checkpoints belong to this folder; no older generated detection file or model checkpoint is used as an input. Tracktor is freshly imported from the verified source archive.

The results are organized as `models/`, `caches/`, and `results/`. The final `comparison.md` has separate sections for appearance encoders, GCN, tracker-only GT boxes with IDs hidden, detector-only precision/recall/F1, a fixed GCN-off tracker across detectors, detector-specific association runs, and the authors' baseline. All scores are local development validation results, not official MOT17 test scores. Detector thresholds are chosen on these same validation videos, so the selected scores should not be presented as independent test performance.
