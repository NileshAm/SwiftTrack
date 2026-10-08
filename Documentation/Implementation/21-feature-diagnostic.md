# Compare appearance before and after graph attention

Run `test.py` with a trained checkpoint to measure whether graph processing helps
distinguish identities. This is a diagnostic script, not a training run or a test
framework. It leaves `Utils/config.py`, weights and tracking outputs unchanged.

## Run

From the project folder, in the `dragon-track` environment:

```bash
# Quick check: first 100 frames of each configured validation video
python test.py --max-frames 100

# Full validation videos for a more representative comparison
python test.py

# Optional explicit checkpoint/device/output location
python test.py --checkpoint outputs/mot17_tuned_full_v1/model.pt --device cpu --max-frames 10
```

Alternatively edit `CHECKPOINT_PATH`, `MAX_FRAMES_PER_SEQUENCE`, `DEVICE`,
`OUTPUT_ROOT` and `SHOW_PROGRESS` near the top of `test.py`.
`None` means all frames; a number includes the initial frame used to seed memory.
The videos and detection source come from the current config. Existing checkpoint
split, encoder, preprocessing and temporal-state compatibility checks must pass;
the script reports mismatches instead of changing config or using random weights.
The first frame seeds history, so N frames produce N−1 comparison graphs.
The overall and per-video `tqdm` bars show progress and ETA after model/data setup.

## What is compared

For every frame, the script reuses the chronological validation graph pipeline:

1. Predict existing track positions using past observations and obtain their
   appearance histories. Extract features for the current detections.
2. Use ground-truth identities to label candidate track/detection pairs as the
   same person or different people. Unknown identities are excluded from measured
   pairs, but their nodes remain in the graph.
3. **Before:** compare normalized appearance/history features, excluding the four
   box coordinates. **After:** compare the final normalized graph embeddings.
   Both comparisons use precisely the same labelled candidates and distance gate.
4. Update history using ground-truth assignments only after scoring this frame.

The graph sees geometry as well as appearance, so this measures the complete
graph transformation. It does not isolate attention from the learned projection
and relationship network. This uses GT-guided history, like validation loss;
it does not reproduce errors accumulated by actual tracking assignments.

## Output and interpretation

Each run creates a timestamped folder under
`<checkpoint folder>/feature_diagnostics/` (or `--output-root`). It contains
`report.md`, `summary.csv`, `summary.json`, and a PNG for each video plus `COMBINED`.
JSON records checkpoint hash, epoch, actual inspected frame ranges, state settings,
counts and histogram bins. Read each video's result as well as the combined result.

| Measurement | Meaning |
| --- | --- |
| Same-person mean | Average cosine similarity of allowed correct pairs. |
| Different-person mean | Average cosine similarity of allowed incorrect labelled pairs. |
| Separation/gap | Same-person mean minus different-person mean. Larger generally indicates better separation. |
| Strict top-1 | Among tracks with both a correct candidate and an incorrect labelled competitor, percentage where the best correct score strictly exceeds every incorrect score. |
| Ties | Best correct and incorrect scores differ by no more than `1e-6`; these never count as strict wins. |
| Mean ranking margin | Average best-correct minus best-incorrect score on those competing queries. |
| Gate exclusion | Known same-ID pairs in the available memory/detections removed by the spatial gate. This is not detector recall. |
| Zero candidate nodes | Zero-length feature vectors among nodes participating in measured pairs; counts repeated observations across frames. |

All eligible pair observations enter the means and histograms; there is no random
negative sampling. Combined results pool counts rather than averaging video
percentages. `N/A` means insufficient eligible observations, not zero accuracy.
Ranking excludes tracks whose correct detection is missing or gated out, which
is why the gate counts and query counts must be read alongside it.

If separation **and** strict ranking consistently deteriorate after the graph on
full videos, test a residual appearance connection as a separate experiment.
High cosine values alone do not establish feature collapse. Many ties or zero
vectors also need investigation. If ranking improves but HOTA/IDF1 remains poor,
investigate detection coverage, gating, assignment and track lifecycle next.
Confirm any architectural change through actual tracking evaluation; this report
does not compute HOTA/IDF1 and does not guarantee that a residual will improve them.
Short prefixes are only preliminary checks and may miss difficult later scenes.

## Functions in test.py

- `load_models`: load compatible checkpoint and encoder in evaluation mode.
- `observe_graph`: measure pre/post similarities, competition, ties and gate misses.
- `new_stats` / `summarize`: bounded accumulators and count-weighted summaries.
- `save_plot` / `save_report`: per-video and pooled plots, CSV, JSON and readable report.
- `main`: parse optional arguments, visit validation videos chronologically under
  inference mode, show progress and save a separate report folder.
