# Crowded pedestrian detector suite

This folder compares official **pretrained** detectors on the fixed SwiftTrack validation videos. It does not
train or fine-tune a detector. Every completed arm writes a verified MOT-format cache, detector diagnostics,
one selected confidence operating point, frozen-OSNet/no-GCN tracks, TrackEval metrics and provenance under
`outputs/crowded_detector_suite_v1/<detector>/`.

## Reproducibility contract

- Videos: `MOT17-02-SDP` and `MOT17-09-SDP`, in full.
- Detection: IoU 0.50 and 0.75; confidence sweep 0.05–0.70; GT height and visibility recall.
- Selection: maximum detector F1 subject to precision >=70% (fallback: maximum F1). The declared rule is
  saved before tracking, so tracking results are not chosen retrospectively.
- Tracking: frozen MSMT17 OSNet, exact existing Tracktor association checkpoint, GCN bypassed, checkpoint
  matching threshold, new-track score 0.5, otherwise checkpoint/baseline settings.
- Baseline: HOTA 44.206, IDF1 51.182, MOTA 50.310, IDSW 226, FP 48, FN 11605.
- A missing official checkpoint/runtime produces `limitation.json`; another model is never substituted.

## Official availability findings

**CrowdDet:** the official repository pins Python 3.6.8, PyTorch 1.5, torchvision 0.6 and CUDA 10.1. Its
README links an ImageNet ResNet-50 initialization, not a trained CrowdHuman detector. Therefore this suite
records the requested pretrained arm as unavailable unless an official detector checkpoint is supplied.

**Iter-E2EDET:** the official CrowdHuman checkpoint exists, but the official code supports Linux/macOS and
builds a bundled Detectron2 extension. It is not a truthful drop-in for the current native Windows/PyTorch
2.2 environment. Run it under WSL/Linux and export the same MOT cache format if this arm is required.

**DEIMv2-S:** official COCO checkpoint and code are available. Download the official checkpoint:

```powershell
New-Item -ItemType Directory -Force detector_models\official, detector_models\weights
git clone https://github.com/Intellindust-AI-Lab/DEIMv2 detector_models\official\DEIMv2
python -m pip install PyYAML faster-coco-eval calflops huggingface_hub gdown
gdown 1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL -O detector_models\weights\deimv2_dinov3_s_coco.pth
```

Do **not** install DEIMv2's full requirements file into `dragontrack`: it pins PyTorch 2.5.1 and torchvision
0.20.1 and would replace this project's working CUDA build. The targeted packages above preserve PyTorch.
The official project recommends PyTorch 2.0+ but tests a newer pinned pair, so any incompatibility is recorded
as a limitation rather than patched by silently changing the model.

**YOLO26-M:** official Ultralytics COCO weights are named `yolo26m.pt`; the package downloads them on the
first run. Use an Ultralytics release that documents YOLO26.

```powershell
python -m pip install --upgrade ultralytics
```

## Commands

Activate the existing environment and perform the non-inference preflight:

```powershell
conda activate dragontrack
python -m detector_models.preflight
```

Run one arm at a time, in the requested order. A completed stage is reused on restart:

```powershell
python -m detector_models.run_suite --detector crowddet
python -m detector_models.run_suite --detector iter_e2edet
python -m detector_models.run_suite --detector deimv2_s
python -m detector_models.run_suite --detector yolo26_m
```

Or allow the same runner to proceed sequentially and record unavailable arms:

```powershell
python -m detector_models.run_suite --detector all
```

The final files are `outputs/crowded_detector_suite_v1/comparison.md` and `comparison.csv`. If a GPU OOM
occurs, do not silently lower resolution for that arm: its recipe would no longer be identical. Record the
failure, clear other GPU processes, and rerun; only introduce a new resolution as a separately named arm.

