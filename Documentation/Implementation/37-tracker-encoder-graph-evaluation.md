# Encoder and graph comparison on fixed detections

Run from the project root in the `dragon-track` environment:

```bash
python evaluate-tracker-variants.py --smoke-test
python evaluate-tracker-variants.py
python evaluate-tracker-variants.py --detections gt
```

The second command is the real-detection baseline: it runs Tracktor and YOLOX on the full MOT17-02-SDP and MOT17-09-SDP validation videos. Each detector is paired with frozen OSNet, fine-tuned OSNet, and pretrained DenseNet121, with graph attention enabled and bypassed: **12 arms total**. It saves `comparison.md`, `comparison.csv`, `baseline_manifest.json`, and per-arm tracks and metrics under a fresh `outputs/tracker_detector_baselines/<timestamp>/` folder. Progress bars show each arm and the overall ETA. The third command retains the earlier six-arm ground-truth-box control. To run one real-detection arm first:

```bash
python evaluate-tracker-variants.py --detections yolox --encoder frozen_osnet --graph off
```

The default baseline uses the validated Tracktor and pretrained CrowdHuman+MOT20 YOLOX caches. Both use the same validation frames and tracker settings. The default read threshold is 0.5 and the new-track threshold is 0.7, matching the earlier detector comparison. Tracktor's scores are placeholders, so its 0.5 read threshold is not a calibrated confidence cut. Choose another value with `--detection-min-score` or `--new-track-min-score` and keep the resulting table separate from the fixed baseline.

With `--detections gt`, the script instead reads marked class-1 pedestrian GT **boxes** from the MOT validation dataset and writes them as score-1 detection input. It does not pass GT person IDs to any tracker. TrackEval reads IDs only after tracking to calculate metrics. This is an analysis control, not a deployable detector-plus-tracker result.

The `without_gcn` arm calls the same checkpoint with graph attention bypassed. It still uses the checkpoint's learned Sinkhorn unmatched preference and Hungarian matching. This isolates the effect of graph attention better than switching to the separate direct-cosine tracker. To compare with direct cosine, see [the previous GT-box control](36-tracker-only-evaluation.md).

Default association checkpoints are `outputs/encoder_comparison_reid_v1/model.pt` for both OSNet variants and `outputs/encoder_comparison_densenet_v1/model.pt` for DenseNet. The fine-tuned appearance weights default to `outputs/osnet_finetune/lr_1e-05/best.pt`; use `--finetuned-osnet-weights PATH` to select another learning-rate run. **Both fine-tuned OSNet arms are transfer experiments:** the GCN and Sinkhorn matcher were trained on frozen OSNet features, then the appearance encoder was replaced. This project has no separately trained association checkpoint for the fine-tuned OSNet. The manifest records this limitation and the exact weight hashes. An alternative compatible association checkpoint can be supplied with the corresponding `--*-association` option.

The completed real-detection results are recorded as the [fixed Tracktor and YOLOX baseline](38-detector-tracker-baseline.md). The earlier full six-arm GT-box control is at `outputs/tracker_gt_variants/20261124_135704_391539/`. Its frozen OSNet without-GCN arm produced HOTA 81.706, DetA 96.127, AssA 69.453, IDF1 80.038, and 634 switches. Keep these ideal-box control values separate from the real-detection baseline.
