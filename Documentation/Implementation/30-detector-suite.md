# Sequential detector experiments A-E

## Start

Activate the existing working environment in the project root:

```powershell
conda activate dragontrack
python detector_suite.py
```

Default: Tracktor, newly fine-tuned RT-DETRv2-S, COCO-pretrained YOLO26-S and
COCO-pretrained RF-DETR-S. Each source gets its own four-epoch OSNet association
model, full validation tracking, and TrackEval evaluation. Then the highest-HOTA
successful confidence-scored detector is tested with high/low confidence recovery.
This may take many hours; a complete suite is not guaranteed to finish overnight.

To include both small and medium YOLO/RF-DETR variants:

```powershell
python detector_suite.py --sizes sm --run-dir outputs/detector_suite_sm_v1
```

First-use preflight only (installs workers, downloads models and runs real-frame
checks; no detector/association training):

```powershell
python detector_suite.py --check-only
```

Then `python detector_suite.py` reuses preparation and continues. Keep the same
options and run directory when restarting. To alter epoch budgets, use a new run:

```powershell
python detector_suite.py --association-epochs 2 --detector-epochs 3 --run-dir outputs/detector_suite_screen_v1
```

## Prerequisites and automatic preparation

MOT17 images/GT/metadata for config's five training and two validation sequences
and the existing base Python/CUDA environment are required. Paths, image counts,
video splits and core imports are checked first. Test videos are not used.
Require at least 8 GiB free disk; medium models and retained failed attempts may
need more. Internet is required for packages and uncached pretrained models.

The suite downloads/checksums the Tracktor archive and official MSMT17 OSNet
weights, imports suite-owned Tracktor caches, and downloads pretrained detector
weights automatically. No friend's outputs or private detector weights are needed.
It does not download the MOT17 dataset or manufacture absent images/annotations.

Optional libraries are installed in `<run-dir>/worker_env`, a local venv sharing
base torch/torchvision. Base environment packages are not uninstalled or upgraded.
Constraints preserve their versions, Transformers 4.49.0, timm 1.0.15 and NumPy
1.26.4. Adapters are pinned to Ultralytics 8.4.216 and RF-DETR 1.3.0. RF-DETR's
newer releases require a different Transformers API. Installation/download/API
failures are surfaced in preflight logs, not silently replaced by other models.
Worker dependency resolution is network-dependent and can still fail.

Windows automatic idle sleep is inhibited during the command and restored on exit.
This does not override lid-close, manual sleep/shutdown, battery failure or restarts.
Keep the laptop open and plugged in. Only run one suite process per run directory.

## Protocol

A. Import Tracktor detections; train a fresh OSNet association model and evaluate.

B. Fine-tune RT-DETRv2-S from pinned COCO weights: square 640, 40% deterministic
training-frame subset, three epochs,
lr 5e-5/backbone lr 5e-6, weight decay 1e-5, batch 1 x accumulation 4. This effective
batch of four is chosen to reduce OOM risk on the project's 6-GB GPU; it is not
identical to the old batch 2 x 2 run.
Validate every epoch, retain the threshold-aware best export, then generate caches
for all training/validation videos. Train OSNet association on the new source.
This recreates the recipe, not the lost trial-52 weights or guaranteed scores.

C. YOLO26 S (and M with `--sizes sm`): official COCO weights, square letterbox
640 with rect=False, RGB PIL inputs, person class resolved by name, max_det=300.

D. RF-DETR S (and M): official pretrained detection weights at each model's
native resolution, official RGB preprocessing, person class resolved by name.
Native preprocessing differs from YOLO and RT-DETR; this is a practical comparison
of detector configurations, not a controlled architecture-only comparison.

Generated caches retain person scores >=0.05 and record full-video coverage,
settings, model weight hashes, timing and box-file hashes. Detector evaluation
sweeps 0.05-0.70 at IoU 0.5/0.75, with size/visibility breakdowns. Select highest
detector F1 subject to precision >=70% and recall >=52%; if unmet, use the best-F1
fallback and explicitly flag ineligibility. Tracktor naturally falls just below
the recall goal but remains the reference. Detector F1 does not establish tracking
performance; the threshold is not optimized separately for each tracker. The
comparison reports validation detection-cache FPS (including frame reads and
export) separately from tracking FPS; Tracktor's detector FPS is n/a because its
boxes are precomputed.

Every association model uses full training videos, the same frozen OSNet, common
graph/motion/history/assignment settings and four epochs. Training and normal
tracking filter boxes using that source's selected confidence threshold. The
existing NEW_TRACK_MIN_SCORE still applies to track births. Separate models are
necessary because the source and its distribution change. No Tracktor-trained
checkpoint is silently reused on new detections.

E. Choose the successful non-Tracktor baseline with highest validation HOTA,
IDF1 as tie-breaker. Using its same association checkpoint, first match detections
above its selected threshold, then use boxes between 0.05 and that threshold to
recover unmatched tracks. Weak boxes never start IDs. Candidate gating, graph,
Sinkhorn and assignment remain unchanged. This is ByteTrack-inspired recovery,
not official ByteTrack. Checkpoint provenance retains its training threshold;
summary.json separately records the explicit inference read threshold and recovery
policy. Reusing validation for detector thresholds and winner selection introduces
selection bias; these are development results, not an independent final test.

## Files and recovery

All new experiments are under `outputs/detector_suite_v1/` by default:

```text
plan.json, state.json, comparison.md, comparison.csv, comparison.json, comparison.png
logs/                    # Persistent stage output and failures
worker_env/              # Isolated optional packages
weights/, jobs/, caches/ # Downloaded models, worker jobs, full-video caches
rtdetr_training/         # Fresh best/latest detector checkpoints and exports
sources/<source>/
  detector_evaluation/   # P/R/F1 reports and size/visibility curves
  operating_point.json   # Selected threshold and eligibility flag
  association/           # model.pt, last.pt, history.csv and training plots
  tracking/, evaluation/ # MOT outputs, HOTA/IDF1/MOTA/IDSW, plots
  result.json
  recovery/              # E for the chosen source, if any
```

Completed stages are skipped only if their recorded artifact hashes match.
Training restores its latest complete epoch. Complete detection videos are
verified and reused; an incomplete video restarts. Incomplete tracking/evaluation
folders are renamed with `_partial_...` and the stage restarts, preserving files.
Changed plan settings require a new run directory. Failed models/stages are logged;
independent remaining experiments continue. Final exit status is nonzero if there
are failures or blocked stages. Check `comparison.md` and `state.json` before
assuming all experiments completed. Do not edit source/config while a run is active.

No program can guarantee the absence of file/network/GPU errors. This runner
creates its generated inputs, checks prerequisites, and records actionable errors.
It never substitutes dummy weights or reports failed experiments as complete.

## Code responsibilities

- `detector_suite.py`: preparation, isolated dependency setup, subprocess stages,
  RT-DETR fine-tuning, threshold selection, compatible association training,
  automatic A-E ordering, resume hashes, tables and plots.
- `detector_worker.py`: official YOLO/RF-DETR adapters, class lookup, RGB and box
  validation, model hashes, full-video MOT export through the shared cache writer.
- `Utils/detections.py`: strict generic cached-source recipes alongside existing
  Tracktor/RT-DETR backends; no silent detector fallback.
- `evaluate_detector.py`: identical cached-box evaluation for generic sources.
- `tracking.associate_frame`: optional high-first/low-recovery association;
  `tracking.main(recovery_high=..., recovery_low=...)` enables E without changing
  the default tracker behavior.

Official APIs: https://docs.ultralytics.com/modes/predict/ and
https://github.com/roboflow/rf-detr/tree/1.3.0.
