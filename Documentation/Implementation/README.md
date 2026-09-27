# Four-day implementation TODO

Follow the [simplified implementation plan](../DragonTrack_Implementation_README.md). Use your existing Conda environment. Build **eight Python files**, run them directly, and keep run notes in one `results.md`. File/function names below are proposed; no implementation is claimed yet.

No separate test files, `.toml` packaging, YAML configuration system, or custom experiment framework. Check behavior through small real runs and a few prints/asserts. The linked notes explain the important functions without requiring extra infrastructure.

## Track progress simply

Write a developer/agent name next to **Owner**, add a brief note only if useful, and tick the box when the task works. Put `working` or `blocked: reason` in the note when needed. Developer 1/2 below are suggested roles, not assigned people. Give each agent one file or function; do not edit the same file simultaneously.

## Day 1 — read data and get one training step working

- [ ] **1. Settings and environment — `config.py`.** Plain settings for paths, data splits, output folder, model dimensions, learning rate, epochs, and thresholds. Confirm PyTorch sees the GPU in the existing environment. **Owner:** Developer 1 / ___; **Note:** ___. [How](01-setup-and-audit.md)
- [ ] **2. Read data and training labels — `data.py`.** Implement `load_sequence`, `load_ground_truth`, `match_detections_to_gt`. Keep actual scores/frame IDs and separate training labels from inference. **Owner:** Developer 1 / ___; **Note:** ___. [How](03-data.md)
- [ ] **3. Appearance features — `features.py`.** Implement `load_encoder`, `extract_features`. Use frozen DenseNet and small batches of crops. **Owner:** Developer 1 / ___; **Note:** ___. [How](04-features.md)
- [ ] **4. Graph building — `graph.py`.** Implement `pairwise_iou`, `build_graph`. Check box geometry and node/edge indices with a small example. **Owner:** Developer 2 / ___; **Note:** ___. [How](05-graph-and-model.md)
- [ ] **5. Association model and losses — `model.py`.** Implement `AssociationModel.forward`, `sinkhorn`, `compute_loss`. Return named dictionary values; get one finite forward/backward step working. **Owner:** Developer 2 / ___; **Note:** ___. [How](05-graph-and-model.md)

**Order:** agree the [simple arrays/dictionaries](02-contracts.md) first. Data and features feed the graph; the graph feeds the model. Developers can work with small made-up arrays while the other file is being written. Before ending Day 1, try one real frame pair and one crowded MOT20 sample.

## Day 2 — train and track

- [ ] **6. Training and saved model — `train.py`.** Implement `train_one_epoch`, `validate`, `main`; save/load directly with PyTorch. Make a tiny sample learn, then start a bounded longer run. Needs tasks 2–5. **Owner:** Developer 2 / ___; **Note:** ___. [How](07-training.md)
- [ ] **7. Tracking and MOT output — `track.py`.** Implement `match_tracks`, `update_tracks`, `run_sequence`, `main`. Keep IDs through matches/short gaps; handle empty frames and write MOT rows. Needs tasks 2–5; use task 6's checkpoint for meaningful results. **Owner:** Developer 1 / ___; **Note:** ___. [How](06-assignment-and-tracker.md)
- [ ] **8. Evaluation setup — `evaluate.py`.** Implement `run_trackeval`, `main`; run existing TrackEval on a small saved output. Begin setting up existing ByteTrack/BoT-SORT code while training runs. Needs task 7's output. **Owner:** Developer 1 or agent / ___; **Note:** ___. [How](08-evaluation.md)

**Finish point:** a saved model reloads and a short sequence produces sensible tracking rows. Freeze the architecture by midday; keep unresolved paper components in limitations instead of rebuilding the project.

## Day 3 — measure and improve

- [ ] **9. Record baseline accuracy and speed.** Use `train.py`, `track.py`, and `evaluate.py` on fixed held-out MOT17/MOT20 inputs. Write metrics, FPS, peak memory, and run settings in `outputs/<run_name>/results.md`. Needs tasks 6–8. **Owner:** both / ___; **Note:** ___. [How](08-evaluation.md)
- [ ] **10. Fix the biggest runtime bottleneck.** Edit the existing `extract_features`, `build_graph`, `AssociationModel.forward`, or `run_sequence` as needed. Start with batching/reusing work, then rerun the same sample and compare outputs, metrics, and speed. Needs task 9. **Owner:** Developer 2 / ___; **Note:** ___. [How](09-performance.md)

**Finish point:** real before/after measurements. No separate profiler or multiprocessing pipeline is required.

## Day 4 — comparisons and delivery

- [ ] **11. Run ByteTrack and BoT-SORT comparisons.** Use existing implementations on the same detections and frames, then evaluate with TrackEval. Record their versions/settings/results or a clear setup blocker in `results.md`. No general adapter framework. Needs task 8; can start earlier. **Owner:** Developer 1 or agent / ___; **Note:** ___. [How](08-evaluation.md)
- [ ] **12. Finish the demo and runnable handover.** Add `draw_tracks` in `track.py` for a short video. Restart Python, load the saved model, and rerun a clip. Add actual commands to the root README and finish `results.md` with missing paper components and comparison limitations. Needs tasks 9–11, with any failed attempt explained. **Owner:** both / ___; **Note:** ___. [How](08-evaluation.md)

**Optional only after delivery is ready:** one small accuracy improvement with an on/off comparison. Otherwise leave it for later. [Deferred work](10-improvements.md)

## Simple working arrangement

| Person | Main files | Suitable AI help |
| --- | --- | --- |
| Developer 1 | `config.py`, `data.py`, `features.py`, `track.py`, `evaluate.py` | One parser, crop function, export function, or baseline setup at a time. |
| Developer 2 | `graph.py`, `model.py`, `train.py`; runtime changes | One geometry function, loss fix, shape-debugging task, or timing change at a time. |

Use one GPU-heavy job at a time. Coordinate before an agent edits another person's file. A short note such as “`extract_features` now returns `[D,F]`; tried 20 crops successfully” is enough for handover.

## Values to fill in before real runs

- Existing Conda environment: ___
- MOT17/MOT20 locations: ___
- Detection source, score meaning, and box coordinate convention: ___
- Train / validation / held-out sequences or frame ranges: ___
- Pretrained weights and graph library allowed by the project: ___

Put the actual settings in `config.py`; no extra decision document is needed.

## How the finished code should run

These are target commands for the planned files, **not commands that work yet**. Run from the repository root in your already activated Conda environment. Each script reads `config.py`; no command-line framework is needed.

```bash
python train.py
python track.py
python evaluate.py
```

Only enable video writing for the demo; keep it off while measuring speed. If behind schedule, finish the trained tracker, evaluation, and results first; drop optional caching and improvements.
