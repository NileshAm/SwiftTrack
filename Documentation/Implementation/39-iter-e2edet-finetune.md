# Fine-tune the pretrained Iter-E2EDET detector on MOT17

`detector-train-itere2edet.py` starts from the authors' released CrowdHuman model and keeps their Iter Sparse R-CNN architecture, top-500 proposals, matching and losses. It adapts MOT17 marked pedestrian boxes into training targets and sends unmarked or non-person GT boxes to the authors' ignored-region path. Only `Utils/config.py` training videos receive gradients; complete validation videos are used for checkpoint selection. The training script does not change OSNet or the association model.

The [authors' repository](https://github.com/megvii-research/Iter-E2EDET) pairs this checkpoint with `50e.6h.500pro.ignore.yaml`. This project validates at the selected 960/1600 inference size. Training also defaults to 960/1600; lower `--train-short-side` and `--train-max-size` explicitly if GPU memory is insufficient. No automatic resolution fallback occurs.

## Train

```bash
conda activate dragon-track
cd /home/nilesh/UOM/CV/Project/ModifiedCode
python detector-train-itere2edet.py \
  --checkpoint weights/iter_e2edet/model.pth \
  --output-dir outputs/itere2edet_mot17_finetune_v1 \
  --epochs 5 --batch-size 1 --accumulation 4 \
  --train-short-side 960 --train-max-size 1600 \
  --lr 1e-5 --backbone-lr 1e-6 --weight-decay 1e-4 \
  --scores 0.2 0.3 0.4 0.5 --device cuda
```

This uses every frame of each configured training video (`--frame-stride 1`). Lower `--frame-stride` for a quick exploratory run in a **new** output directory. `--amp` enables optional bfloat16 CUDA autocast; it is off by default. The code will report an OOM rather than change your chosen batch or image size.

Each epoch validates all configured validation frames at IoU 0.5 and 0.75. It saves `best.pth` when the highest count-weighted IoU 0.5 F1 improves, `latest.pth` with optimizer state after each epoch, `history.json`, per-epoch validation metrics, and a recipe with source hashes. To continue to a larger total epoch count, rerun the same command with `--epochs 8 --resume`; other settings and training data must stay the same. An existing nonempty output directory is protected unless `--resume` is specified.

## Evaluate the best checkpoint

Export into a **new** cache. The pretrained 960/1600 cache stays untouched.

```bash
python detect-itere2edet.py \
  --checkpoint outputs/itere2edet_mot17_finetune_v1/best.pth \
  --split val --short-side 960 --max-size 1600 \
  --min-score 0.05 --cache-dir detections/iter_e2edet_mot17_finetuned_v1

python evaluate_detector.py \
  --backend iter_e2edet --split val \
  --cache-dir detections/iter_e2edet_mot17_finetuned_v1 \
  --scores 0.1 0.2 0.3 0.4 0.5 0.7 --ious 0.5 0.75 \
  --output-dir outputs/detector_evaluation/iter_e2edet_finetuned_v1

python evaluate-tracker-variants.py \
  --detections iter_e2edet \
  --iter-checkpoint outputs/itere2edet_mot17_finetune_v1/best.pth \
  --iter-cache-dir detections/iter_e2edet_mot17_finetuned_v1 \
  --encoder frozen_osnet --graph off \
  --detection-min-score 0.4 --new-track-min-score 0.7 \
  --output-dir outputs/iter_e2edet_finetuned_tracker_040_070
```

Compare against the pretrained 960/1600 baseline at the **same** 0.40/0.70 tracking thresholds: 38.330 HOTA, 47.627 IDF1, 50.556 MOTA on these validation videos. The newly fine-tuned detector may need a different confidence threshold; select one using detector F1 or a separate tracking search, then confirm on untouched videos before reporting a final result. Detector F1 is the checkpoint selection metric, so it does not guarantee a tracking gain.
