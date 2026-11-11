# Four-day implementation TODO

- [x] **Continue training from the last epoch — `Utils/training_resume.py`, `train.py`, `Utils/config.py`, `search.py`. Owner: Codex.** Restore weights/Adam/history/best loss, save random states and latest checkpoint each epoch, back up run files, protect search isolation. [Usage and functions](23-resume-training.md)
- [ ] **Continue the residual run to epoch 40 and compare validation tracking. Owner: ___ .** Training is configured but has not been launched. [Resume guide](23-resume-training.md)

- [x] **Preserve each node's own features — `Utils/attention.py`, `model.py`. Owner: Codex.** Learned residual projections in both attention layers, versioned checkpoints, separate run folder and architecture-aware search resume. [Implementation and training](22-appearance-residual.md)
- [ ] **Train and evaluate `mot17_appearance_residual_v1`. Owner: ___ .** Compare full diagnostic ties/ranking and tracking HOTA/IDF1 with `mot17_tuned_full_v1`. [Comparison guide](22-appearance-residual.md)

- [x] **Appearance before/after graph diagnostic — `test.py`. Owner: Codex.** Same/different identity similarities, strict competing-candidate ranking, ties, gate exclusions, per-video reports and plots. [Usage and interpretation](21-feature-diagnostic.md)
- [x] **Run the diagnostic on full validation videos. Owner: Nilesh.** Epoch-20 tuned model: strict ranking 84.66%→27.87%, ties 0%→50.35%; residual experiment implemented next. [Guide](21-feature-diagnostic.md)

- [x] **Config-independent parameter search — `search.py`. Owner: Codex.** Editable parameter grid and epoch counts, fresh training or existing-checkpoint tracking, separate trial folders, HOTA/IDF1/MOTA ranking and tqdm progress. Restart skips completed trials and archives/retries unfinished trials. [Settings, functions and usage](20-parameter-search.md)
- [ ] **Run the full validation parameter search and compare the winning trial. Owner: ___ .** The implementation checks do not select production settings. [Search guide](20-parameter-search.md)
- [x] **Optional training frame limit for search — `search.py`, `train.py`. Owner: Codex.** Consecutive prefixes per training video, full validation, checkpoint range metadata and resume compatibility. [Usage and trade-offs](20-parameter-search.md#reduce-training-time-with-a-frame-limit)

- [x] **Fixed matching normalization and unit-embedding contrastive loss — `Utils/sinkhorn.py`, `model.py`, `Utils/losses.py`, `train.py`, `Utils/plots.py`. Owner: Codex.** Bounded matching options, learned unmatched preference, per-pair reporting, new checkpoint/run, short GPU verification. [Functions and checks](19-matching-and-loss-fix.md)

- [x] **Restore frozen pretrained appearance — `Utils/config.py`. Owner: Codex.** New run `mot17_paper_tracktor_frozen`; old fine-tuned checkpoint preserved. Superseded by the matching/loss repair above; old outputs are preserved.

- [x] **Actual image batches for train/validation — `Utils/appearance.py`, `Utils/features.py`, `train.py`. Owner: Codex.** Shared encoder forward/backward, chronological history, partial batches, and measured GPU defaults. [Functions, settings, and benchmark](18-batched-training.md)

- [x] **Paper-aligned graph, fine-tuned appearance, Tracktor inputs, and track confirmation. Owner: Codex.** Implemented and checked through a short GPU training/tracking/evaluation run. [Files, functions, settings, and limitations](17-paper-alignment.md)
- [ ] **Full training/evaluation of `mot17_paper_tracktor`. Owner: ___ .** Run the full dataset, compare accuracy, and tune uncertain defaults on validation. [Run instructions](17-paper-alignment.md#run)

The entries below record earlier development stages; the guide above defines the current architecture and run.

- [x] **Train/validate with motion and appearance history — `Utils/track_state.py`, `Utils/training_state.py`, `train.py`.** Chronological past-only state, shared tracking utilities, GT-guided updates after scoring, mixed ages, new checkpoint recipe and history logging. **Owner:** Codex. [Settings and functions](16-state-training.md)
- [ ] **Train and evaluate `mot17_SDP_state30` on full videos.** Compare tracking HOTA/IDF1/MOTA against the previous run; validation loss alone is insufficient. **Owner:** ___ . [Run guide](16-state-training.md#run)

- [x] **Motion prediction and appearance history — `Utils/motion.py`, `Utils/appearance_history.py`, `tracking.py`.** Independent Kalman state and bounded reliable feature memory per ID; on/off settings and separate comparison folders. **Owner:** Codex. [Settings and functions](15-motion-and-history.md)
- [ ] **Measure motion/history accuracy.** Compare baseline, motion only, and both additions using the same checkpoint and validation videos. **Owner:** ___ . [Comparison steps](15-motion-and-history.md#settings-and-output)

- [x] **Shared 30-frame look-back and SDP restoration.** Random historical pairs for training, deterministic gap coverage for validation, matching tracker expiry, bounded feature cache, and fresh run/checkpoint checks. [Settings and implementation](14-lookback.md)

- [x] **Cached RT-DETRv2-S detections — `detect.py`, `Utils/detections.py`.** Person-only MOT export, overall ETA, cache reuse/integrity checks, and training/tracking integration. Run `python detect.py` before fresh association training. **Owner:** Codex. [Settings and functions](13-detection-cache.md)

- [x] **Training/validation and evaluation plotting — `Utils/plots.py`, `main.py`.** Automatic epoch charts, combined/per-video tracking scores, error breakdowns, and IoU diagnostics; regenerate existing results without training. **Owner:** Codex. [Functions and chart guide](12-plots.md)

**Current data split:** configure separate full-video `TRAIN_SEQUENCES` and `VAL_SEQUENCES`. See the [split guide](11-paper-components.md#whole-video-selection--utilssplitspy); old temporal checkpoints require retraining.

**Current architecture:** paper-equation attention and fine-tuned DETR. Follow the [current guide and checklist](17-paper-alignment.md), which supersedes the original DenseNet/mean-message model notes below. Full retraining is required; old checkpoints are incompatible. Test videos have a separate `TEST_SEQUENCES` list and output folder.

Follow the [simplified implementation plan](../DragonTrack_Implementation_README.md). Use your existing Conda environment. The **eight core Python files** are now implemented, including `evaluate.py`; full training/evaluation and comparisons remain to be run. `main.py` plots the selected run’s training history. Current `TRAIN_SEQUENCES` and `VAL_SEQUENCES` select disjoint complete videos; percentage splitting has been removed. [Demo instructions](05-graph-and-model.md#see-it-working-in-mainpy).

No test framework, `.toml` packaging, YAML configuration system, or custom experiment framework. The requested `test.py` is a runnable feature diagnostic. Check other behavior through small real runs and a few prints/asserts. The linked notes explain the important functions without requiring extra infrastructure.

## File layout

Use the existing capitalized `Utils/` folder for shared helpers. Keep the model and main scripts at the project root:

```text
Utils/
    config.py       # Created
    data.py         # Implemented: read data and prepare labels
    features.py     # Dispatch to DETR appearance (optional DenseNet comparison)
    appearance.py   # Full-image DETR + per-box ROI features
    motion.py       # Per-track constant-velocity Kalman filter
    appearance_history.py # Bounded confidence-weighted feature memory
    track_state.py  # Shared track lifecycle for training and inference
    training_state.py # GT-guided chronological training memory
    track_filter.py # Observation-count confirmation and filtered export
    detections.py   # Checked supplied/Tracktor/RT-DETR input selection
    relationships.py # Learned appearance/geometry/position affinities
    attention.py    # Edge-aware neighbor attention
    sinkhorn.py     # Learnable normalization and unmatched matching
    pairs.py        # Identity targets and hard/random negative sampling
    losses.py       # Weighted BCE and contrastive losses
    outputs.py      # Shared split/sequence output paths
    splits.py       # Whole-video split validation and checkpoint compatibility
    graph.py        # Implemented: geometry and graph construction
import_tracktor.py  # Import authors-linked preprocessed public boxes
model.py            # Implemented: association model, normalization, and losses
train.py            # Implemented: train, validate, save weights/loss history
search.py           # Parameter grid and epoch budgets; isolated training/evaluation trials
test.py             # Compare appearance before/after graph attention on validation
tracking.py         # Implemented: assign IDs, preserve short gaps, save MOT rows/preview
evaluate.py         # Implemented: align temporal clips and run TrackEval
```

Imports from the main scripts use `from Utils import config`. When the helpers exist, use imports such as `from Utils.data import load_sequence`. Run the main scripts from the project root; the helper files are imported rather than run directly.

## Track progress simply

Write a developer/agent name next to **Owner**, add a brief note only if useful, and tick the box when the task works. Put `working` or `blocked: reason` in the note when needed. Developer 1/2 below are suggested roles, not assigned people. Give each agent one file or function; do not edit the same file simultaneously.

## Day 1 — read data and get one training step working

- [x] **1. Settings and environment — `Utils/config.py`.** Plain settings for paths, data splits, output folder, model dimensions, learning rate, epochs, and thresholds. **Owner:** Codex (config/checks) / Developer 1 (environment); **Note:** data paths checked; PyTorch 2.6.0+cu124 in dragon-track detects the RTX 4060 and completed a CUDA tensor calculation on 27 September 2026. [How](01-setup-and-audit.md)
- [x] **2. Read data and training labels — `Utils/data.py`.** Implemented `load_sequence`, `read_image`, `load_ground_truth`, `match_detections_to_gt`, and shared `pairwise_iou`. **Owner:** Codex; **Note:** checked three MOT17 sequences and temporary edge cases in dragon-track; installed NumPy/SciPy/Pillow. MOT20 check pending local data. [How](03-data.md)
- [x] **3. Appearance features — `Utils/features.py`.** Implemented `load_encoder`, `extract_features`; returns CPU features and a validity mask. **Owner:** Codex; **Note:** official pretrained weights cached; CPU/RTX 4060 checks passed on 13 MOT17 detections, including batching/order, frozen gradients, and empty/invalid crops. Real MOT20 check pending data. [How](04-features.md)
- [x] **4. Graph building — `Utils/graph.py`.** Implemented `build_graph`; reused `pairwise_iou`. **Owner:** Codex; **Note:** geometry, edge mapping, empty/masked cases, real MOT17 frame pair, and synthetic dense graph checked. Graph design is a documented partial reproduction; real MOT20 check pending data. [How](05-graph-and-model.md)
- [x] **5. Association model and losses — `model.py`.** Implemented `AssociationModel.forward`, `sinkhorn`, `build_targets`, `compute_loss`. **Owner:** Codex; **Note:** CPU checks, synthetic learning, state reload, and real MOT17 GPU forward/backward passed. Mean-message graph model and fixed unmatched Sinkhorn are documented partial-reproduction choices; dataset training remains task 6. [How](05-graph-and-model.md)

**Order:** agree the [simple arrays/dictionaries](02-contracts.md) first. Data and features feed the graph; the graph feeds the model. Developers can work with small made-up arrays while the other file is being written. Before ending Day 1, try one real frame pair and one crowded MOT20 sample.

## Day 2 — train and track

- [x] **6. Training and saved model — `train.py`.** Implemented frame-pair preparation, `train_one_epoch`, `validate`, `main`, best/latest checkpoints, and CSV loss history. **Owner:** Codex; **Note:** now uses percentage splits of the same videos and saves resolved ranges. A 20-frame CPU pipeline passed; earlier checkpoint/reload checks passed. Full configured training remains to be run. [How](07-training.md)
- [x] **7. Tracking and MOT output — `tracking.py`.** Implemented `match_tracks`, `update_tracks`, `run_sequence`, `draw_tracks`, `main`. **Owner:** Codex; **Note:** assignment/aging checks and a 10-frame CPU validation clip using the trained checkpoint passed. Thresholds remain provisional; full evaluation and optional MP4 verification remain pending. [How](06-assignment-and-tracker.md)
- [x] **8. Evaluation setup — `evaluate.py`.** Implemented input validation/cropping/rebasing, `run_trackeval`, and `main`, using installed TrackEval 1.3.0 with MOT preprocessing. **Owner:** Codex; **Note:** real mini-pipeline passed; native metrics, summaries, detailed CSV, logs and run metadata saved. Full metrics and baseline comparisons remain tasks 9/11. [How](08-evaluation.md)

**Finish point:** a saved model reloads and a short sequence produces sensible tracking rows. Freeze the architecture by midday; keep unresolved paper components in limitations instead of rebuilding the project.

## Day 3 — measure and improve

- [ ] **9. Record baseline accuracy and speed.** Use `train.py`, `tracking.py`, and `evaluate.py` on fixed held-out MOT17/MOT20 inputs. Write metrics, FPS, peak memory, and run settings in `outputs/<run_name>/results.md`. Needs tasks 6–8. **Owner:** both / ___; **Note:** ___. [How](08-evaluation.md)
- [ ] **10. Fix the biggest runtime bottleneck.** Edit the existing `extract_features`, `build_graph`, `AssociationModel.forward`, or `run_sequence` as needed. Start with batching/reusing work, then rerun the same sample and compare outputs, metrics, and speed. Needs task 9. **Owner:** Developer 2 / ___; **Note:** ___. [How](09-performance.md)

**Finish point:** real before/after measurements. No separate profiler or multiprocessing pipeline is required.

## Day 4 — comparisons and delivery

- [ ] **11. Run ByteTrack and BoT-SORT comparisons.** Use existing implementations on the same detections and frames, then evaluate with TrackEval. Record their versions/settings/results or a clear setup blocker in `results.md`. No general adapter framework. Needs task 8; can start earlier. **Owner:** Developer 1 or agent / ___; **Note:** ___. [How](08-evaluation.md)
- [ ] **12. Finish the demo and runnable handover.** Add `draw_tracks` in `tracking.py` for a short video. Restart Python, load the saved model, and rerun a clip. Add actual commands to the root README and finish `results.md` with missing paper components and comparison limitations. Needs tasks 9–11, with any failed attempt explained. **Owner:** both / ___; **Note:** ___. [How](08-evaluation.md)

**Optional only after delivery is ready:** one small accuracy improvement with an on/off comparison. Otherwise leave it for later. [Deferred work](10-improvements.md)

## Detector coverage experiment

- [x] Add separate DDQ inference, MOT cache export and shared detector evaluation/comparison entry points. [How](27-ddq.md)
- [ ] Compare DDQ and RT-DETR on full validation; assess recall at similar precision, F1 and small/occluded people before selecting a detector.

- [x] Add `detector-search.py` and `Utils/detector_selection.py`: report-only threshold ranking, isolated/resumable training grids, and checkpoint selection across thresholds with precision/recall floors. [How](26-detector-search.md)
- [ ] Run the small LR grid, validate the winner, then compare a larger detector if coverage remains insufficient.
- [x] Add `detector-train.py` and `Utils/detector_training.py`: MOT detector fine-tuning, ignored-region loss handling, held-out validation, baseline/best/latest checkpoints, resume, plots and local-weight inference integration. [How](25-detector-training.md)
- [ ] Run full detector fine-tuning and compare the selected export against Tracktor before retraining association.
- [x] Add `evaluate_detector.py` for cached-box confidence sweeps, per-video precision/recall/F1, missed/duplicate counts and size/visibility recall.
- [x] Add `detect.PersonDetector.predict` original-resolution input with padding and dynamic anchors; keep caches separate and validate compatibility.
- [ ] Compare full validation at 640, 960 and original resolution; select a useful precision–recall tradeoff, then regenerate training boxes and train a fresh run. [Commands and implementation details](24-detector-evaluation.md)

## Simple working arrangement

| Person | Main files | Suitable AI help |
| --- | --- | --- |
| Developer 1 | `Utils/config.py`, `Utils/data.py`, `Utils/features.py`, `tracking.py`, `evaluate.py` | One parser, crop function, export function, or baseline setup at a time. |
| Developer 2 | `Utils/graph.py`, `model.py`, `train.py`; runtime changes | One geometry function, loss fix, shape-debugging task, or timing change at a time. |

Use one GPU-heavy job at a time. Coordinate before an agent edits another person's file. A short note such as “`extract_features` now returns `[D,F]`; tried 20 crops successfully” is enough for handover.

## Values to fill in before real runs

- Existing Conda environment: ___
- MOT17/MOT20 locations: ___
- Detection source, score meaning, and box coordinate convention: ___
- Shared `SEQUENCES`; `TRAIN_FRACTION=0.8`, `VAL_FRACTION=0.1`, evaluation gets the remainder.
- Pretrained weights and graph library allowed by the project: ___

Put the actual settings in `Utils/config.py`; no extra decision document is needed.

## How the finished code should run

Run from the repository root in your already activated Conda environment. Each script reads `Utils/config.py`; no command-line framework is needed. Training, tracking, and evaluation are available. `tqdm` and TrackEval are installed in `dragon-track`. Train fresh weights in the new `mot17_temporal_split` run, track/evaluate with `TRACK_SPLIT="val"`, then change it to `"eval"` after tuning.

```bash
python train.py
python tracking.py
python evaluate.py
```

Only enable video writing for the demo; keep it off while measuring speed. If behind schedule, finish the trained tracker, evaluation, and results first; drop optional caching and improvements.
