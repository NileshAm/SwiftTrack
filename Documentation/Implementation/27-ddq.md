# Compare DDQ-DETR with RT-DETRv2

DDQ is a separate **inference and evaluation** path in the same `dragon-track`
Conda environment. It does not change RT-DETR settings, weights, training, or the
active Tracktor source. DDQ fine-tuning is not implemented by this entry point.

## Files and responsibilities

- `detect-ddq.py`: select configured videos, run DDQ, export complete MOT detection
  files, preview boxes, record provenance and show video/frame progress with ETA.
- `Utils/ddq.py`: official checkpoint download, MMDetection architecture and
  preprocessing, strict weight loading, person-class lookup, and original-image boxes.
- `evaluate_detector.py --backend ddq`: the same matching, ignored-person policy,
  confidence sweep and size/visibility metrics used for RT-DETR.
- `compare-detectors.py`: compare two or more evaluator `summary.json` reports;
  reject incompatible video coverage, GT counts or scoring policies. Select highest
  measured F1 meeting minimum precision/recall, or mark the fallback ineligible.

## Run DDQ on validation

```bash
conda activate dragon-track
python detect-ddq.py --smoke-test
python detect-ddq.py --split val --cache-dir detections/ddq_r50_5scale_val_v1
python evaluate_detector.py --backend ddq --cache-dir detections/ddq_r50_5scale_val_v1 --scores 0.1 0.15 0.2 0.225 0.25 0.275 0.3 0.35 0.4 0.5 0.6 0.7 --output-dir outputs/detector_evaluation/ddq_r50_5scale_v1
```

The first inference downloads the official COCO checkpoint into `outputs/ddq_weights`.
Use `--checkpoint` only for trusted local weights matching the selected architecture.
The smoke check processes one real frame, writes a preview/boxes under
`outputs/ddq_smoke`, and creates **no** evaluation cache. Full runs use all selected
frames; detection never reads GT. Choose `--split train` or `--split all` to export
other configured video groups.

Default DDQ uses ResNet50 **five-scale**, aspect-preserving resizing with shorter
side 800 and longer side capped at 1333, FP32, cache score floor 0.10, and maximum
300 all-class outputs before filtering to people. Class IDs are resolved from the
checkpoint/classes, not copied from RT-DETR. Internal DDQ query selection remains
enabled; no extra output NMS is added. RGB images are converted to BGR for MMDetection,
and predictions are rescaled to original image coordinates before MOT export.

This native DDQ preprocessing differs from RT-DETR's square 960. Compare each
pretrained recipe first; record the difference rather than treating resolution as
controlled. For a square-resolution experiment:

```bash
python detect-ddq.py --split val --input-mode square --image-size 960 --cache-dir detections/ddq_r50_5scale_square960_v1
```

Five-scale uses its own official COCO checkpoint and adds a higher-resolution
feature level. It may use more memory; the smoke check above tests a frame first.
The default cache name includes `5scale`, so it cannot reuse four-scale results.
Keep the same resizing and confidence sweep when comparing the two variants.
To reproduce the earlier four-scale detector, explicitly select it:

```bash
python detect-ddq.py --variant 4scale --split val --cache-dir detections/ddq_r50_4scale_val_v1
```

The paper's CrowdHuman AP is not the expected MOT F1 of these COCO checkpoints.

## Export the existing RT-DETR search winner for comparison

```bash
python detect.py --backend rtdetrv2 --split val --weights-dir outputs/detector_search/detector_search_lr_0.4/trial_052/best --input-mode square --image-size 960 --min-score 0.10 --cache-dir detections/rtdetr_trial052_val_v1
python evaluate_detector.py --backend rtdetrv2 --cache-dir detections/rtdetr_trial052_val_v1 --scores 0.1 0.15 0.2 0.225 0.25 0.275 0.3 0.35 0.4 0.5 0.6 0.7 --output-dir outputs/detector_evaluation/rtdetr_trial052_v1
python compare-detectors.py --reports outputs/detector_evaluation/rtdetr_trial052_v1/summary.json outputs/detector_evaluation/ddq_r50_5scale_v1/summary.json --min-precision 0.70 --min-recall 0.55
```

This is a practical comparison of your fine-tuned RT-DETR against pretrained DDQ,
not an architecture-only benchmark. Also compare pretrained RT-DETR if you need
matched training history. Compare recall at similar precision, maximum F1,
IoU-0.75 box accuracy, small/occluded-person recall and per-video performance.
The detector threshold 0.25 must not be confused with association matching thresholds.

## Cache safety and recovery

Generated boxes go to `<cache>/<dataset-split>/<sequence>/det.txt`, never into the
dataset. Rows are MOT one-origin xywh with confidence and identity -1. Each video
gets a preview and completion manifest containing original coverage/dimensions,
weight/config hashes, package versions, resizing, confidence floor and timing.

Rerun the same command to skip verified complete videos. Interrupted videos restart
from their first frame; incomplete data cannot pass evaluation. Use a new cache
directory after changing weights, preprocessing, output cap or confidence floor.
A directory lock prevents concurrent exports overwriting one another. Evaluation
checks the saved source's provenance, rather than your current RT-DETR settings.

Evaluation reports are timestamped unless you give a new `--output-dir`; existing
reports are never overwritten. The comparison report stores paths to every source
report. It does not train, change config.py, or replace the active detector.

## Environment

Use `dragon-track`; do not create another environment or replace the working torch,
torchvision, transformers or NumPy packages. DDQ uses `mmdet==3.3.0`,
`mmengine==0.10.7` and `mmcv==2.1.0` with compiled CUDA operators. `mmcv-lite` is
insufficient. The pinned architecture files come from the MMDetection wheel.

This machine's PyTorch is 2.6.0+cu124, Python 3.12 and GPU is RTX 4060 (8 GB).
MMCV needs a source build matching those versions, CUDA 12.4 compiler/headers and
GPU architecture 8.9. Never install an MMCV wheel built for a different PyTorch
version. The older source build also needs legacy setuptools `pkg_resources`;
temporary build tools can provide that without replacing environment setuptools.

MMCV was built successfully against this existing stack. Four-scale DDQ passed
GPU inference with native aspect-preserving input and square 960 on the two
validation image sizes. The existing fine-tuned RT-DETR export also passed GPU
inference afterward. These are functionality checks, not full validation scores.
Five-scale DDQ also passed a real GPU frame. A two-frame four-scale export was
loaded and evaluated successfully by the shared evaluator; synthetic checks cover
color conversion, clipping, empty output, MOT coordinates and cache rejection.
The full validation comparison has not been run automatically.

Published checkpoints can contain duplicate `decoder.aux_reg_branches` tensors
created by DDQ's training forward pass. The loader verifies that each equals its
`bbox_head` counterpart before removing the alias, then loads all model weights
strictly. Other missing, unexpected or shape-mismatched weights are errors.

Sources: [MMDetection DDQ models](https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/ddq),
[official DDQ implementation](https://github.com/jshilong/DDQ/tree/ddq_detr).
