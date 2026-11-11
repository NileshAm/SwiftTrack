# Detector threshold and training grid search

Run from the project root in the `dragon-track` Conda environment. Search settings
live at the top of `detector-search.py`; this script **never edits config.py**.
It is separate from the association/tracking search in `search.py`.

## Why this search is needed

The completed batch-size-2 detector run shows that a fixed confidence threshold can
hide an improvement. Pretrained epoch 0 at score 0.4 has precision 77.33%, recall
44.40%, F1 56.41%. Fine-tuned epoch 2 at score 0.2 has precision 74.90%, recall
50.34%, F1 60.21%. It finds 12,035 rather than 10,614 correct person instances,
but also increases false positives from 3,112 to 4,033. Those are validation
observations, not an independent test result or a guarantee for another run.

At score 0.4, epoch 2 recall falls to 39.50%, despite precision reaching 95.47%.
The apparent stagnation therefore includes a change in confidence calibration.
Training loss keeps falling after the early epochs, while validation loss rises;
this is consistent with overfitting. The batch-size-1 run deteriorates more; it
also changed the effective batch from 8 to 4, so it is not a clean isolated
comparison of minibatch size.

Do not optimize raw detection count. Optimize **correct detections (TP/recall)**
alongside F1 and precision. Extra false boxes can make tracking worse.

## 1. Rank existing reports without inference

```bash
python detector-search.py --mode reports
```

By default, `REPORT_RUNS` finds existing `outputs/detector*/settings.json` runs.
You can specify exact run directories instead:

```bash
python detector-search.py --mode reports --runs outputs/detector_rtdetrv2_s_mot17_v1
```

This searches only confidence thresholds that were actually recorded in each epoch
report. It does not interpolate 0.15/0.25 results when only 0.1/0.2/0.3 were saved.
It does not train, alter old checkpoints, export different weights or regenerate
caches. Runs must share exactly the same validation videos, frame IDs and GT hashes.

The result files are `ranking.csv`, `best.json` and `report.md` in a new directory
under `outputs/detector_search_reports/`. `best.json` distinguishes the best measured
result from the best result whose checkpoint is still present. Existing reports do
not recreate discarded weights: the old batch-size-2 epoch-2 checkpoint was not
retained, because the old fixed-score selection chose a different epoch. A new
short training trial is needed to retain weights selected by the threshold sweep.

## 2. Run the small training grid

```bash
python detector-search.py --mode train
```

Search settings (edit the grid at the top of the script):

- Every combination in `PARAM_GRID` becomes a training trial.
- Backbone LR is 0.1 times the selected LR.
- RT-DETRv2-S; input size, accumulation and other options come from your grid.
- `TRAIN_FRACTION = 0.25`: approximately 25% of each training video's frames,
  evenly spread across its duration. Every training video is retained.
- 3 training epochs per trial, independent of `DETECTOR_TRAIN_EPOCHS`.
- Full validation videos at epoch 0 and after every training epoch.
- Confidence scores `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]`.
- Select highest F1 at IoU 0.5 subject to precision >=0.70 and recall >=0.50.

These floors are experimental goals, not evidence that a model meets them.
Recall 0.50 is above the pretrained detector's 0.444 operating point but still
below Tracktor's approximately 0.5146 recall. Set `MIN_RECALL=0.52` if you require
more correct detections than Tracktor. The precision floor also needs to reflect
your tolerance for tracker false positives; Tracktor's current precision is much
higher at approximately 0.996.

An eligible candidate outranks every ineligible one. If none meet both floors,
the script clearly labels the highest-F1 fallback **ineligible**; it does not
pretend the target was achieved. Higher raw box count is never a ranking metric.

Thresholds are evaluated on the same model outputs, so adding a threshold does
not retrain or rerun the model for that threshold (CPU matching work increases).
The selected score can vary by epoch. Both threshold and eligibility are saved in
history/checkpoints and the best export manifest. Pretrained epoch 0 is included:
a trial that does not beat its baseline may export the baseline.

Edit `PARAM_GRID`, `FIXED`, `EPOCHS`, `TRAIN_FRACTION`, `SCORES`, and the floors at the top of the
search script. Supported training grid keys are `lr`, `backbone_lr`, `image_size`,
`batch_size`, `accumulation`, `weight_decay`, and `frame_stride`. `PARAM_GRID` values
override `FIXED`. If `backbone_lr` is specified, it overrides the LR ratio.
For example, after the LR comparison, try input sizes `[640, 960]` with one LR.
Avoid a large all-parameters-at-once grid.

CLI options include `--epochs 3`, `--scores 0.15 0.2 0.25 0.3`, `--min-precision`,
`--min-recall`, `--device`, `--output-dir` and `--max-frames`. The last option is only
for smoke checks and caps **both** train and validation; it is recorded in the
search/training manifests. Use `--train-fraction` for real searches instead:

```bash
python detector-search.py --mode train --train-fraction 0.25 --output-dir outputs/detector_search/fraction025_v1
```

The fraction must be greater than 0 and at most 1. A video with 1,000 frames
contributes 250 evenly spaced frames at 0.25, covering its beginning, middle and
end. Counts round down, with at least one frame per nonempty video (the middle
frame when only one is selected). This preserves coverage across videos and time;
it cannot guarantee that every rare event appears in a small subset.

The same deterministic subset is used for all trials and epochs. Training batches
are still shuffled. Validation uses all frames unless you explicitly use the
separate `--max-frames` smoke-check option. Leave `frame_stride=1` when you want the
fraction relative to the full videos: a larger stride is applied **before** the
fraction and reduces training frames further. Frame IDs are saved in the manifests;
changing the fraction requires a new search output directory.

After selecting promising settings, confirm them using `--train-fraction 1.0`
in a separate search directory. Direct `detector-train.py` still defaults to all
training frames, and also accepts `--train-fraction`. `config.py` is never edited.
Training becomes shorter, but full validation time stays the same.

## Output and recovery

Training trials are placed under `SEARCH_DIR/trial_001`, etc., or your `--output-dir`.
Each has its own best/latest checkpoints, exported `best/`, history/plots, validation
reports and completion record. The search root contains the recipe and ranked
results. Separate processes release GPU memory between trials. `tqdm` shows trial
progress and each trainer's remaining batches; trial ETA needs a completed trial
to become useful. No full search is automatically launched by editing these files.

Restart the same command to continue. Completed trials are checksum-verified and
skipped. An incomplete trial resumes the last completed training epoch. A failed
trial stops the search and remains available for inspection/resume. A lock prevents
two searches sharing the same output directory. Changing grid, data, code, floors,
thresholds or epoch budget requires a new search directory, preventing accidental
reuse of incompatible results. `config.py` remains unchanged.

Do not fine-tune again inside a completed search trial directory. To extend a winner,
use a separate detector run; use the trial's settings in that new run or make a copy
of its run files first. Keep the archived search output intact for comparison.

## Direct threshold-aware training

Legacy `python detector-train.py` retains fixed-score behavior and old checkpoint
recipe compatibility. A new run can select scores jointly with checkpoints:

```bash
python detector-train.py --output-dir outputs/detector_score_sweep_v1 --epochs 3 --batch-size 2 --selection-mode sweep --scores 0.1 0.15 0.2 0.25 0.3 0.4 --min-precision 0.7 --min-recall 0.5
```

The `best/training_manifest.json` records `selection.score`, `selection.eligible`,
precision, recall and F1. When caching detections, use a cache floor **at or below**
the selected score. When reading that cache for association/tracking, set
`DETECTION_MIN_SCORE` to the selected score in the new experiment's config.
Changing confidence or detector source requires a new association run.

## Should we switch models?

Do the small calibration/LR search first. The current data demonstrate that some
improvement was hidden by threshold choice, and that more epochs alone are not
helping. A larger model is worth a controlled comparison afterward, not a promised
fix for occlusion or box-label mismatch.

A practical next candidate is **RT-DETRv2-L / ResNet50**, available as
`PekingU/rtdetr_v2_r50vd`. The official model table reports COCO AP 48.1 for S,
51.9 for M (ResNet50-m), and 53.4 for L, with parameter counts 20M, 36M and 42M.
These are COCO AP values, not MOT F1 or guaranteed improvements on our two videos.
On 8 GB, begin larger-model experiments at 640 and batch size 1, adjusting
accumulation to keep the effective batch comparable; fit still needs checking.

`MODEL_VARIANTS` keeps model name and revision paired. Add a verified, model-specific
pinned revision there to compare RT-DETRv2 variants. Do not reuse the ResNet18 SHA
for ResNet50. `detector-train.py` also accepts `--model-name` and `--revision` without
changing config.py. The architecture must be supported by `RTDetrV2ForObjectDetection`;
this is not a generic loader for arbitrary DETR families. Larger-model downloads
and memory/performance have not been tested here.

Select using validation; keep a separate labeled test set for the final estimate.
More repeated validation searches can overfit the validation set. Continue checking
IoU-0.75, per-video, size/visibility metrics and the final tracker's HOTA/IDF1.

## Functions

- `Utils.detector_selection.select_operating_point`: fixed/grid selection with floors.
- `selection_key`: eligibility first, then F1; shared by trainer and search.
- `detector-search.collect_reports`: read epoch sweeps and verify comparable validation data.
- `checkpoint_epochs`: verify which reported epochs still have actual weights.
- `train_grid`: Cartesian trials, isolation, progress, interruption recovery and checksums.
- `save_ranking`: all measured operating points, eligible winner and retained-weight winner.

Sources: [official RT-DETRv2 model table](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
and [ResNet50 checkpoint](https://huggingface.co/PekingU/rtdetr_v2_r50vd).
