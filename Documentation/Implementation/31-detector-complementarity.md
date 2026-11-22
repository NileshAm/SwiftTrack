# Cached detector complementarity

Run from the project root in the dragontrack environment:

```powershell
python detector_next_steps.py
```

For a quick check, add `--max-frames 30`. The default reads
`outputs/detector_suite_v1`; change this with `--suite-dir PATH`.
No GPU processing, training, downloads, or new detection caches are needed.

Each invocation creates a timestamped directory under the suite's `next_steps/`.
Read `report.md`, `summary.csv`, `size_visibility.csv`, and
`precision_recall.png`. `frame_counts.csv` and `coverage_patterns.csv` provide
per-frame counts. Source cache hashes are verified and recorded in provenance.

The selected mode uses each suite operating point. Cache-floor mode includes
every saved proposal. Confidence sweeps reuse completed detector evaluations.
Extra hits mean identities found by an alternative's independent matching but
missed by Tracktor. Joint added TP rematches concatenated proposals, which can
give an optimistic bound in crowded scenes, including duplicate boxes matching
different nearby people. Neither figure is a deployable fusion result.

Matching and ignored annotation handling reuse evaluate_detector.py's diagnostic
policy, which differs from official TrackEval preprocessing. These are
validation diagnostics. A source's FP count is not the extra FP after fusion.
Use the results to justify a separate fusion experiment with GT-free duplicate
suppression before tracking it. Existing caches and checkpoints are preserved.
