# Parameter search — `search.py`

Edit the settings at the top of the root `search.py`, then run:

```bash
conda activate dragon-track
python search.py
```

No new packages are needed. `Utils/config.py` supplies the usual dataset, device,
encoder and other defaults. Search overrides are temporary Python values; the
file is never edited. Running `python train.py` separately still uses the normal
config settings, not the search settings.

## Fresh training for each combination

```python
MODE = "train"
PARAM_GRID = {
    "LEARNING_RATE": [1e-4, 5e-5],
    "MATCH_THRESHOLD": [0.10],
    "TRACK_CONFIRM_MIN_HITS": [1],
}
EPOCH_COUNTS = [3]
EVALUATE_CHECKPOINT = "last"
```

These example settings produce **two independent 3-epoch training runs**,
followed by full validation tracking and evaluation for each. The lower matching
threshold and single-hit confirmation are experiments, not proven improvements.

`EPOCH_COUNTS = [3, 5]` produces **four runs** with this grid: each learning rate
is trained from scratch once for 3 epochs and once for 5 epochs (16 total training
epochs). `range(2, 6)` means counts 2, 3, 4, 5; Python excludes the stop value.
Each unfinished trial starts at epoch 1; the budget does not skip initial epochs or use `config.EPOCHS`.
All trials use the configured seed unless `SEED` is itself in the grid.

`"last"` evaluates the exact final epoch of each run. `"best"` instead evaluates
the checkpoint with the lowest validation loss within that budget, which may be
an earlier epoch. The CSV records both the budget and the checkpoint's actual
epoch. Each trial retains `model.pt` (best validation loss), `last.pt` (final), and
`history.csv`. Training starts with the configured pretrained appearance encoder
and a fresh association model; it does not load `SOURCE_CHECKPOINT`.

The grid takes every combination of the listed values. Add/remove keys as needed;
use the exact config parameter names. Empty `{}` runs the baseline at each chosen
epoch count. Duplicate trials are removed. For a structured value, wrap each
candidate in the values list, e.g. `"GRAPH_HIDDEN_DIMS": [(512, 128), (256, 128)]`.
Supported controls are listed in `TRAIN_PARAMETERS` inside `search.py`.
`LOOKBACK_FRAMES` also updates `MAX_MISSED_FRAMES` in memory so the temporal window
stays consistent. Do not put `EPOCHS`, output paths, or dataset membership in the
grid; keep the validation videos fixed for comparison.

Training-only and tracking-only settings can share a grid, but training mode
repeats training for **every** combination, even if only a threshold changes.
Use tracking mode below when the trained weights can be reused.

## Faster search with existing weights

```python
MODE = "tracking"
SOURCE_CHECKPOINT = config.CHECKPOINT_PATH  # Or Path("outputs/older_run/model.pt")
PARAM_GRID = {
    "MATCH_THRESHOLD": [0.05, 0.10, 0.20, 0.35],
    "TRACK_CONFIRM_MIN_HITS": [1, 3],
}
```

This evaluates eight tracker combinations with the same checkpoint, without
training. `EPOCH_COUNTS` and `EVALUATE_CHECKPOINT` have no effect in this mode.
Remove training parameters such as `LEARNING_RATE` from the grid; the script
rejects them rather than silently ignoring them. Supported names are in
`TRACKING_PARAMETERS`. Existing checkpoint compatibility checks still apply:
use matching dataset, detection source, feature settings and look-back window.
The script does not bypass those checks or reconstruct missing past checkpoints.

## Reduce training time with a frame limit

For a preliminary training search, set these in `search.py`:

```python
MODE = "train"
EPOCH_COUNTS = [2]
TRAIN_MAX_FRAMES_PER_SEQUENCE = 200
```

This trains on frames **1–200 of every configured training video**, for two
epochs per trial. A shorter video uses all its frames. The value is a frame count,
not a percentage, a total across videos, or the number of detections. At least
two frames are required; 200 frames produce 199 adjacent training anchors per
video. Empty-detection frames remain in the chronology. The first frame seeds
track memory, so the history and motion logic remain chronological.

The default `None` uses all frames. The limit is ignored in `MODE="tracking"`.
Validation loss, validation tracking and TrackEval always use full validation
videos. `python train.py` still uses full training videos, and `config.py` is
unchanged. The optional argument `train.main(max_train_frames_per_sequence=...)`
is how search passes the cap to training.

The cap is saved in `search.json`, each trial's `result.json`, and `results.csv`.
Training checkpoints also save `training_frame_limit` and the actual per-video
`training_frame_ranges`; the full dataset/split identity remains recorded
separately. A changed cap selects a new search instead of mixing results with
an older budget. Older searches without this field mean full-video training.

For a more representative quick comparison, try fewer epochs on full videos
first: repeated short prefixes can miss later crowding, occlusions and motion.
If that is still too slow, a 200-frame cap and 1–2 epochs can screen candidates.
Recheck promising settings on full training videos for a longer budget before
selecting final settings. Full validation still takes time, so a training-frame
cap does not reduce the whole search runtime in direct proportion to the cap.

For matching threshold, confirmation or association-distance comparisons on
existing weights, use `MODE="tracking"` and remove training-only grid keys such
as `LEARNING_RATE`. This skips training altogether. It measures the changed
inference behavior; a later training run can match the chosen graph/state settings.

## Outputs and progress

```text
outputs/searches/association_search_<timestamp>/
    search.json                    # Grid, base settings and data split
    results.csv                    # Completed trials, best first
    best.json                      # Winning parameters and checkpoint/report paths
    trial_001/
        settings.json, result.json
        model.pt, last.pt, history.csv     # Training mode only
        tracking/search/val/<sequence>/...
        evaluation/search/val/results.md
    trial_002/...
```

`RANK_METRIC` can be `"HOTA"`, `"IDF1"`, or `"MOTA"` (higher wins). HOTA is the
default; ties use IDF1, then the first trial. `RANK_BENCHMARK="MOT17"` selects
which benchmark supplies the ranking; scores from different benchmarks are not
averaged. Full native metrics remain in each trial's evaluation directory.

The ranking is refreshed after every completed trial. `best.json` references the
actual saved checkpoint and matching settings; nothing is applied automatically
to `config.py`. Copy the winning tracking parameters and checkpoint path there
yourself when ready. Other search parameters must also match that checkpoint.
The best search result is chosen by tracking accuracy, while each trial's
`model.pt` is still chosen by validation loss.

`tqdm` shows the search count/ETA and the existing training/tracking/evaluation
bars. Search ETA becomes available after the first completed trial and is only
approximate when epoch budgets differ. Validation uses all frames of the fixed
validation videos. Optional test videos and videos are skipped; per-trial plots
are controlled by `SAVE_SEARCH_PLOTS`. The dataset and detection cache are reused.

## Resume an interrupted search

`RESUME = True` is the default. Running `python search.py` again finds the newest
compatible folder with the same `SEARCH_NAME`, grid/order, epoch budgets, training frame limit, mode,
checkpoint selection, ranking and effective dataset/model settings. It reloads
completed `result.json` files and skips those trials. If every trial is complete,
it rebuilds the ranking and exits without training or tracking again.

The first unfinished trial **restarts from epoch 1**. This is trial-level resume,
not continuation of optimizer state within a partially trained trial. Its partial
folder is moved into `interrupted/trial_<number>_<timestamp>/`, preserving any
checkpoints, history, tracking outputs and error logs. Subsequent trials then run
normally. Even if training finished but tracking/evaluation was interrupted, the
unfinished trial is retried in full.

```python
RESUME = True
RESUME_DIR = None  # Automatically find the latest compatible search.
# Or choose an exact folder:
# RESUME_DIR = Path("outputs/searches/association_search2_20261003_161703_821353")
```

For an intentional fresh run, set `RESUME = False` and `RESUME_DIR = None`.
An automatic search with no matching settings starts a new timestamped folder;
an explicitly selected incompatible `RESUME_DIR` raises an error. Changing the
grid, its order, or epoch budgets starts a separate experiment. Config defaults
overridden by the search, such as `config.EPOCHS`, are ignored when comparing.
Do not change model code or dataset contents midway through a search.

Old search folders made before resume support are recognized. Tracking searches
now record a checkpoint SHA-256 to reject replaced weights; for older manifests,
the checkpoint must not have been modified after the manifest was written.
Stop the old process before restarting. A file lock prevents two processes using
the updated script from resuming the same folder simultaneously and is released
automatically on exit/crash (implemented with `fcntl` for the project's Linux environment).

A trial counts as completed only when its result matches the trial settings, its
metrics are finite, and its checkpoint and evaluation report still exist. Missing
or incomplete results are retried. Results are saved atomically; the ranking is
rebuilt from trial results after restart, including an interruption just before
the ranking was saved. Resume works even after a forced stop without `error.txt`.

Keep grids small initially; the script prints completed/remaining trials and the
remaining training epochs. A validation winner is not an independent test-set
result and short-budget performance may not predict a long training run.

## Functions and progress checklist

- [x] `build_trials()` — validate parameter names/mode, expand the Cartesian grid
  and epoch counts, remove exact duplicate trials.
- [x] `temporary_settings()` — set attributes in the imported config module and
  restore them in `finally`, including after failures.
- [x] `trial_settings()` — assign private output paths and synchronize look-back
  settings; keep search work on validation videos.
- [x] `search_signature()` / `find_search_folder()` — locate a compatible prior
  experiment, including older manifests, or select a fresh folder.
- [x] `search_lock()` / `completed_result()` — prevent concurrent resume and check
  which trials can be reused.
- [x] `train.load_frame_pairs(..., max_frames=...)` — limit training prefixes only;
  `train.main(max_train_frames_per_sequence=...)` records actual trained ranges.
- [x] `main()` — call the existing train/tracking/evaluate functions sequentially,
  select the requested checkpoint, record actual epochs/metrics and progress.
- [x] `run_trials()` — skip completed trials and archive/retry unfinished ones.
- [x] `write_json()` — atomically replace manifests/results to survive interruption.
- [x] `save_results()` — write the ranking and best completed result after each
  successful trial.
- [ ] Run the desired full-video search and compare HOTA/IDF1, recall, false
  positives and ID switches before adopting settings. **Owner:** ___ .

Implementation checks use temporary small runs, not a separate test suite. They
do not establish which parameter combination is best on the full dataset.

Verified on CPU with six-frame samples from separate MOT17-04 training and
MOT17-02 validation videos: fresh 1- and 2-epoch trials, actual checkpoint epochs,
tracking and TrackEval, then two tracking-only trials. Additional temporary
checks covered grid expansion, ranking, best-loss checkpoint selection, bad
inputs, and restoring settings after a deliberate failure. The config file's
SHA-256 remained unchanged. No full-dataset search has been run by this change.

Resume was checked with temporary pipeline stand-ins: interrupt trial 2 after
trial 1 succeeds, restart only trials 2/3, preserve trial 1 byte-for-byte and
archive partial files. Checks also covered a completed-search no-op, corrupt
results, ranking-write interruption, incompatible settings, a replaced tracking
checkpoint and concurrent-run rejection. A read-only check recognized the
existing five-trial `association_search2` run without modifying its outputs.

The frame limit was verified in a real CPU training/tracking/TrackEval run with
two tiny training videos (6 and 4 frames) and a separate 6-frame validation video.
A limit of 3 produced ranges 1–3 for both training videos while all 6 validation
frames remained in evaluation. Checks covered shorter-video clamping, invalid
limits, checkpoint range records, CSV output, legacy full-video resume and
changed-limit isolation. The earlier interruption/resume checks also passed again.
