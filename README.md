# DragonTrack MOT17 comparison

This repository compares pedestrian tracking on disjoint MOT17 training and validation videos. The [fixed local readings](Documentation/FINAL_COMPARISON_BASELINES.md) cover GT boxes without IDs, Tracktor boxes, and pretrained CrowdHuman + MOT20 YOLOX-X boxes. The [authors' pinned GitHub implementation](Baselines/DragonTrackGithub/README.md) is provided separately for a baseline reproduction.

Use the existing `dragon-track` Conda environment. Select complete train and validation videos in [Utils/config.py](Utils/config.py); keep the video lists disjoint. Generated detections, checkpoints, and evaluation reports live in `detections/` and `outputs/`.

```bash
conda activate dragon-track
cd /home/nilesh/UOM/CV/Project/ModifiedCode
```

## Current tracker comparison

The comparison script scores frozen OSNet, fine-tuned OSNet, and DenseNet, each with or without graph attention, on Tracktor and YOLOX detections. It can also run the GT-box control. GT identities are hidden from tracking and used only by TrackEval.

For the three current report rows, run one source at a time. Each command creates a new output folder containing `comparison.md` and `comparison.csv`:

```bash
python evaluate-tracker-variants.py --detections gt --encoder finetuned_osnet --graph off --detection-min-score 0.5 --new-track-min-score 0.7
python evaluate-tracker-variants.py --detections tracktor --encoder finetuned_osnet --graph off --detection-min-score 0.5 --new-track-min-score 0.7
python evaluate-tracker-variants.py --detections yolox --encoder finetuned_osnet --graph off --detection-min-score 0.5 --new-track-min-score 0.7
```

The script expects the existing association checkpoint at `outputs/encoder_comparison_reid_v1/model.pt`, fine-tuned OSNet at `outputs/osnet_finetune/lr_1e-05/best.pt`, and the validated Tracktor/YOLOX caches. Use its `--finetuned-osnet-association`, `--finetuned-osnet-weights`, `--tracktor-cache-dir`, and `--yolox-cache-dir` arguments if yours are elsewhere. The fine-tuned OSNet arm currently transfers features into an association model trained with frozen OSNet features.

## Detector quality

These commands measure precision, recall, and F1 against MOT17 validation labels. They do not measure tracking:

```bash
python evaluate_detector.py --backend tracktor --scores 0.5 --ious 0.5
python evaluate_detector.py --backend yolox --cache-dir detections/yolox_x_mot20_val_v1 --scores 0.5 --ious 0.5
```

To regenerate YOLOX detections, run `python detect-yolox.py --split val`. To fine-tune OSNet, run `python osnet-train.py --lr 1e-5 --epochs 5`. The direct OSNet tracker entry points are `track-tracktor-osnet.py` and `track-yolox-osnet.py`; the comparison above uses the learned association checkpoint instead.

## Authors' original baseline

The vendored [DragonTrack GitHub baseline](Baselines/DragonTrackGithub/README.md) has dedicated training, inference, and evaluation scripts. From the project root:

```bash
python Baselines/DragonTrackGithub/train.py --epochs 20 --batch-size 2 --output-dir outputs/github_dragontrack/baseline_train
python Baselines/DragonTrackGithub/infer.py --checkpoint outputs/github_dragontrack/baseline_train/models_v3/epoch_20.pth --detectors gt tracktor yolox --output-dir outputs/github_dragontrack/baseline_val
python Baselines/DragonTrackGithub/evaluate.py outputs/github_dragontrack/baseline_val
```

These commands are prepared but have not been run here. Replace `epoch_20.pth` with the checkpoint you want to compare. See the baseline README for the pinned source version, required runtime corrections, and output interpretation.

The older step-by-step implementation notes are in [Documentation/Implementation](Documentation/Implementation/README.md). They describe development history; some retired experiment commands in those notes were removed from the active repository.

For independent train, inference and evaluation commands for **our implementation**, see [Baselines/OurImplementation/README.md](Baselines/OurImplementation/README.md). This supports the same GT, Tracktor and YOLOX box comparison as the authors' baseline while recording the checkpoint and detector used for every run.
