# Our implementation: reproducible train, infer, evaluate

These entry points use the existing project code in `train.py`, `tracking.py`, `model.py` and `Utils/`. They keep each run's checkpoint, detection inputs, tracks and metrics in explicit folders. They do not retrain the detector. Choose GT, Tracktor, or pretrained YOLOX boxes with `--detectors`; **each choice trains a separate association model**. The authors' GitHub release trains on GT boxes, so use `--detectors gt` for a closer training-source comparison.

Run from the project root in the existing Conda environment:

```bash
conda activate dragon-track
cd /home/nilesh/UOM/CV/Project/ModifiedCode
python Baselines/OurImplementation/train.py --detectors gt tracktor yolox --encoder detr --epochs 30 --batch-size 2 --val-batch-size 4 --learning-rate 1e-4 --lookback 30 --detection-min-score 0.5 --gt-match-iou 0.5 --output-dir outputs/our_implementation/association_models
```

This example saves three independent checkpoints at `association_models/<detector>/model.pt`; each detector folder also contains `last.pt`. For a single detector, `--output-dir` is its exact run folder. To continue an interrupted run through a larger **total** epoch target, use the same detector choice and output folder and add `--resume --epochs 40`. Training and validation use complete, disjoint videos from `Utils/config.py` by default; `--train-sequences` and `--val-sequences` can override the lists for a new run. `--encoder` accepts `detr`, `osnet`, `densenet`, or `config` (the current config choice). Also available: `--association-distance-px`, `--loss-alpha`, `--negative-ratio`, and `--match-threshold`. The match threshold is saved for inference; it does not change the training loss. Training computes a teacher-forced validation loss, not tracking HOTA.

Without extra flags, YOLOX association training consumes the **pretrained CrowdHuman + MOT20 detection cache**. GT-box training writes identity-free person-box files inside its run folder. Tracktor uses its validated import cache. Set `--yolox-cache-dir` or `--tracktor-cache-dir` when using another cache location. The script never edits `Utils/config.py`; its choices are saved inside each checkpoint. Use a separate output directory for another encoder or settings combination.

To fine-tune YOLOX, export detections from its best checkpoint, then train our association model on those detections:

```bash
python Baselines/OurImplementation/train.py --detectors yolox --encoder detr \
  --train-yolox --yolox-train-epochs 5 --yolox-train-lr 5e-5 \
  --yolox-train-backbone-lr 5e-6 --yolox-train-batch-size 1 \
  --yolox-train-accumulation 4 --yolox-train-fraction 1 \
  --yolox-width 960 --yolox-height 544 --yolox-cache-min-score 0.1 \
  --detection-min-score 0.5 --epochs 30 \
  --output-dir outputs/our_implementation/yolox_finetuned_run
```

`--train-yolox` also builds its cache. The detector trainer saves `yolox_detector/best.pth`, the exporter writes `yolox_cache/`, and association training saves `model.pt` under that output directory. Detector checkpoint selection uses validation F1; later tracker evaluation uses the same validation split, so report these as development results. To keep pretrained weights and only create a fresh cache, replace `--train-yolox` with `--make-yolox-cache`. To export an already trained detector, add `--yolox-checkpoint /path/to/best.pth --make-yolox-cache`.

After that fine-tuned run, evaluate its detector and association checkpoint together:

```bash
python Baselines/OurImplementation/infer.py \
  --checkpoint outputs/our_implementation/yolox_finetuned_run/model.pt \
  --detectors yolox \
  --yolox-checkpoint outputs/our_implementation/yolox_finetuned_run/yolox_detector/best.pth \
  --yolox-cache-dir outputs/our_implementation/yolox_finetuned_run/yolox_cache \
  --output-dir outputs/our_implementation/yolox_finetuned_val
python Baselines/OurImplementation/evaluate.py outputs/our_implementation/yolox_finetuned_val
```

Run inference on the checkpoint and then score its tracks:

```bash
python Baselines/OurImplementation/infer.py --checkpoint outputs/our_implementation/association_models/tracktor/model.pt --detectors gt tracktor yolox --output-dir outputs/our_implementation/association_val
python Baselines/OurImplementation/evaluate.py outputs/our_implementation/association_val
```

The inference command uses the checkpoint's train/validation video lists and feature settings, and saves `tracking/val/<sequence>/tracks.txt` under each detector folder. Its default inference thresholds are detection 0.5 and new track 0.7; the association threshold defaults to the checkpoint value. Override with `--detection-min-score`, `--new-track-min-score`, `--match-threshold`, or `--confirm-min-hits` to record a separate comparison. `--graph off` bypasses graph attention but still uses the model's learned Sinkhorn matcher. Supply `--yolox-cache-dir` and `--tracktor-cache-dir` if your validated caches live elsewhere. For a cache made from a fine-tuned detector, also pass `--yolox-checkpoint` so its hash can be checked. Every output folder must be new.

For the previously reported **fine-tuned OSNet without GCN** arm, use an OSNet association checkpoint and an OSNet fine-tune export:

```bash
python Baselines/OurImplementation/infer.py --checkpoint outputs/encoder_comparison_reid_v1/model.pt --finetuned-osnet-weights outputs/osnet_finetune/lr_1e-05/best.pt --graph off --detectors gt tracktor yolox --output-dir outputs/our_implementation/osnet_no_gcn_val
python Baselines/OurImplementation/evaluate.py outputs/our_implementation/osnet_no_gcn_val
```

This transfers fine-tuned OSNet features into an association model trained on frozen OSNet features; it does not claim the association model was retrained with those features. GT IDs are excluded from every inference input and used only for evaluation. Results are written to `<run>/evaluation/results.md` and `results.json`, with the precise checkpoint and detection hashes in `<run>/manifest.json`. The two local validation videos have been used repeatedly; these are not official MOT17 test results.

**Status:** scripts prepared only. Training, inference and evaluation have not been run by this change.
