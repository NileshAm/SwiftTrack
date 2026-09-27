# DragonTrack: practical four-day implementation plan

**Updated:** 27 September 2026  
**Team:** two developers with AI assistance  
**Environment:** use the existing Conda environment  
**Hardware:** RTX 4060 Laptop GPU, 8 GB VRAM  
**Data:** MOT17 and MOT20 with supplied detection boxes  
**Status:** planned work; no implementation or measured results yet.

Use the [TODO checklist](Implementation/README.md) to track work. This revised plan replaces the earlier, more elaborate project structure.

## 1. What to finish

Build one working, trainable DragonTrack-based tracker, run it on small held-out MOT17/MOT20 subsets, measure its accuracy and speed, and compare it with ByteTrack and BoT-SORT where setup succeeds. Save a trained model, tracking output, a short demo, and a brief results note.

Start with the repaired repository approach using frozen DenseNet crop features. Full paper reproduction and a new accuracy improvement are stretch work. Train one selected baseline; do not build and train several architectures within four days.

Keep implementation to eight ordinary Python files at the repository root. Use the existing Conda environment and run the files directly. No package installation project, `.toml` files, separate test files, configuration framework, or custom experiment framework is required. Use simple settings, dictionaries, tensors, and a few printed checks during actual runs.

## 2. Minimal files to implement

These are proposed files, not existing code. Reuse suitable upstream code and preserve its attribution instead of rewriting everything. If working upstream files are adopted under different names, update the checklist accordingly; do not maintain duplicate versions.

| File | Main work | Guide |
| --- | --- | --- |
| `config.py` | Paths, sequences/splits, thresholds, learning rate, epochs, device, and output folder as plain Python settings. | [Setup](Implementation/01-setup-and-audit.md) |
| `data.py` | Read frames/detections/GT; prepare identity labels for training. | [Data](Implementation/03-data.md) |
| `features.py` | Frozen DenseNet crop features in small batches. | [Features](Implementation/04-features.md) |
| `graph.py` | Correct box geometry, candidate pairs, node/edge tensors. | [Graph/model](Implementation/05-graph-and-model.md) |
| `model.py` | Graph association network, scores, Sinkhorn, and training losses together. | [Graph/model](Implementation/05-graph-and-model.md) |
| `train.py` | Small training/validation loop and model saving. | [Training](Implementation/07-training.md) |
| `track.py` | Assignment, track IDs/history, MOT output, optional video, and simple timing. | [Tracking](Implementation/06-assignment-and-tracker.md) |
| `evaluate.py` | Thin wrapper around TrackEval and a short metric summary. | [Evaluation](Implementation/08-evaluation.md) |

Use `outputs/<run_name>/` for the model, tracking files, evaluator output, and video. Write settings actually used, commands, selected frames, and findings in one `results.md` there. A copied `config.py` or the settings saved inside the checkpoint is enough for the first version. Keep datasets and large output files out of source commits.

## 3. First decisions and environment

The team needs the dataset paths and the source of the supplied boxes. Record them directly in `config.py` and the run's notes. Keep real detection confidence scores. GT boxes are for supervision/evaluation; using them as tracker input is a separately labelled oracle experiment.

Use pretrained DenseNet weights if permitted for the project. Reuse graph libraries if already available and allowed; resolve that choice on Day 1 rather than building two backends. Install only missing packages needed by the chosen implementation. Keep the working Conda setup; do not rebuild it for this plan. Once the code runs, record the useful package versions or save a single Conda export for reference.

Start with FP32, one frame-pair graph at a time, small crop batches (for example 16), and no background loader workers. Run only one GPU-heavy job at a time. Try a crowded MOT20 frame early to catch memory problems.

## 4. Essential corrections to carry over from the earlier audit

The earlier inspection used upstream commit `111e263c650f2f3a6ef39a261582eef87e09e8d8`. These findings guide implementation; they are not proof of fixes in this repository.

| Problem | Minimal action |
| --- | --- |
| Training expects a different number of model outputs | Return one dictionary of scores and embeddings; use the same keys in training/tracking. |
| Graph-building arguments disagree | Use one `build_graph` function in both paths. |
| Image height/width and IoU endpoints are wrong | Use x/width, y/height and `(x,y,x+w,y+h)` conversion. |
| Edge features are used as node features | Keep one row per node; gather edge messages and add them to the correct destination nodes. |
| Fixed local paths, frame padding, and unavailable cache/checkpoint assumptions | Read paths from `config.py`, discover actual image filenames, and show clear missing-file errors. |
| Confidence is replaced by `0.9` | Carry the supplied scores through filtering and track creation. |
| Empty frames/no edges cause failures | Return empty matches and update unmatched tracks without crashing. |
| Loss terms are disconnected or absent | Implement weighted BCE and contrastive terms using the selected documented formulas; print both during training. |
| Validation stays in training mode | Use evaluation mode/no gradients during validation; restore training mode afterward. |
| Learning-rate scheduling is inconsistent | Start with a fixed learning rate; omit the scheduler initially. |

Keep a few lines in `results.md` describing the chosen graph, normalization, and loss formulas and any changes from upstream. Do not create separate audit/decision/template documents.

## 5. Research scope to keep honest

The paper uses transformer features, graph attention, learnable Sinkhorn, and combined losses. The earlier source inspection found DenseNet, an active GCN path, ordinary Sinkhorn use, and incomplete loss wiring. Fix the working path first. Verify equations before calling any replacement paper-aligned; a library attention layer alone does not establish reproduction.

Use `repo_repaired` only if the repaired architecture is retained. If the graph or other method changes, call the result `partial_reproduction` and explain it. Frozen cached features also change the training protocol. If transformer-to-box mapping or another paper detail remains unclear by Day 2, keep it in the limitations and proceed with training/evaluation.

The paper reference settings are 20 epochs, batch size 2, and Adam at `1e-4`; these are context from the earlier plan, not a four-day training promise. Measure time per epoch and choose an affordable run. A shorter run must state its actual training duration.

## 6. Four-day schedule

| Day | Developer 1 | Developer 2 | Practical finish point |
| --- | --- | --- | --- |
| 1 | Settings, data reading, frozen features; start track/output code. | Graph construction and association model; wire losses. | Read a real sample, check shapes/geometry, and complete one forward/backward step. Try dense MOT20 input. |
| 2 | Finish tracking and MOT export; prepare TrackEval and baseline setup while training runs. | Tiny-sample learning, validation, checkpoint saving; start the bounded training run. | Short sequence works; saved trained model reloads. Freeze the chosen method by midday. |
| 3 | Held-out evaluation and baseline runs. | Measure runtime and improve the largest bottleneck with simple batching/reuse. | Actual metrics plus comparable before/after timing. |
| 4 | Finish ByteTrack/BoT-SORT comparisons, demo, and results note. | Check saved-model replay, help resolve remaining bugs; optional improvement only if everything else is complete. | Runnable code, checkpoint, outputs, metrics, commands, and limitations. |

Suggested division only: replace Developer 1/2 with actual names in the checklist. Give each AI agent one function or file at a time. Avoid simultaneous edits to the same file and keep GPU training/inference runs sequential. No reviewer forms, mandatory PR workflow, or handoff log is required.

## 7. Check correctness while running

Use a Python console, a notebook already in use, or a temporary print/assert near the relevant code. Do not create a separate test suite.

- Check identical-box IoU is 1 and disjoint-box IoU is 0; check coordinates on a non-square image.
- Print graph shapes/index limits; confirm edge messages update nodes rather than treating edges as nodes.
- Run a tiny labelled batch: both loss terms and gradients must be finite, and repeated training should learn the simple sample.
- Track a short clip containing matches, new people, an empty frame, and a disappearance. Inspect IDs and output boxes.
- Save, restart Python, load the model, and compare the same clip.
- Run a crowded MOT20 sample and note peak GPU memory.

Fix failures before longer runs. A file existing or a script starting is not the same as a working feature.

## 8. Data and evaluation essentials

Keep train, validation, and final evaluation sequences separate where possible. Put all MOT17 detector variants of the same video in the same split. For temporal splits, keep training pairs/history inside their partition. Store chosen sequences/ranges in `config.py`; a separate manifest system is unnecessary.

For noisy detections, match them one-to-one to valid GT boxes with a stated IoU threshold during training. Same valid identity means a positive pair; different valid identities mean a negative pair. Mask ignored/unmatched examples rather than inventing identities. GT identities must never enter inference decisions.

Use TrackEval for HOTA, IDF1, MOTA, and identity switches. Use its MOT preprocessing; do not write metric formulas. Prefer whole short held-out sequences for easy evaluation. If evaluating a frame range, align GT and tracking frames carefully. Report MOT17 and MOT20 separately using evaluator aggregation. Label subsets and never invent scores for unlabelled benchmark test data.

Attempt ByteTrack and BoT-SORT using existing implementations with the same boxes, scores, and frames. Start setup on Day 2; do not write a general baseline framework. If a comparison cannot run in time, record the blocker. Trackers with their own detector belong in a separately labelled comparison.

## 9. Simple performance work

Time the full supplied-box pipeline first: frame reading, appearance features, graph/model, assignment, and track updates. Exclude external detection and say so. Warm up once, synchronize CUDA at whole-run timing boundaries, and use total frames divided by elapsed seconds. Keep video rendering off during timing. Record peak GPU memory and run the same sample more than once.

Only then locate the slow part. Try reading each image once, batching crops/edge calculations, retaining unchanged historical embeddings, and reducing repeated CPU/GPU transfers. Keep the same checkpoint, detections, thresholds, and frame order when comparing before/after. Recheck output and metrics after changes.

Simple frozen-feature caching is optional if repeated extraction is expensive: save CPU features plus boxes/row order and extractor settings in a `.pt` file; regenerate when inputs or settings change. No cache service or automatic versioning system is needed.

Defer multiprocessing queues, custom profilers, mixed precision, compilation, multi-GPU support, and graph pruning unless the required delivery is already complete. Graph pruning changes the algorithm and must not be described as an exact speed refactor.

## 10. What to submit

Deliver the eight working files, the selected checkpoint, MOT result files, raw TrackEval output, a demo video, and one `results.md` containing:

- Exact run commands and settings, data/box source, split/frame ranges, and package versions that mattered.
- What was repaired, which paper components remain missing, and actual training epochs/time.
- Accuracy and speed results for the trained baseline, retained optimization, and comparison trackers; failed comparisons clearly marked.
- Any optional feature's on/off comparison and a short list of future work.

Use a simple table per dataset; all values below are unmeasured placeholders.

| Version | HOTA | IDF1 | MOTA | ID switches | Supplied-box FPS | Peak GPU memory |
| --- | --- | --- | --- | --- | --- | --- |
| Selected baseline | — | — | — | — | — | — |
| After optimization | — | — | — | — | — | — |
| ByteTrack | — | — | — | — | — | — |
| BoT-SORT | — | — | — | — | — | — |

Keep transformer integration, extra trackers, motion/camera compensation, and new accuracy algorithms for follow-up if the four days run out. Finish a measured partial reproduction rather than leaving evaluation unfinished.

## References

These are the references retained from the earlier plan, not a new source audit.

- [DragonTrack paper](https://openaccess.thecvf.com/content/WACV2025/papers/Galoaa_DragonTrack_Transformer-Enhanced_Graphical_Multi-Person_Tracking_in_Complex_Scenarios_WACV_2025_paper.pdf) and [supplement](https://openaccess.thecvf.com/content/WACV2025/supplemental/Galoaa_DragonTrack_Transformer-Enhanced_Graphical_WACV_2025_supplemental.pdf).
- [Pinned DragonTrack code](https://github.com/ostadabbas/DragonTrack/tree/111e263c650f2f3a6ef39a261582eef87e09e8d8). Preserve upstream license/copyright notices when reusing it.
- [TrackEval](https://github.com/JonathonLuiten/TrackEval), [ByteTrack](https://github.com/FoundationVision/ByteTrack), and [BoT-SORT](https://github.com/NirAharon/BoT-SORT).
