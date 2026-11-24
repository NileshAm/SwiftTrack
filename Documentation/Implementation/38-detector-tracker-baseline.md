# Fixed Tracktor and YOLOX tracker baseline

This is the project's **baseline reading** for comparing later tracker changes. It uses the full MOT17-02-SDP and MOT17-09-SDP validation videos, the validated Tracktor public-preprocessed boxes and pretrained CrowdHuman+MOT20 YOLOX-X cache, detection read score **0.5**, and new-track score **0.7**. The exact detector paths, SHA256 hashes, encoder weights, graph checkpoints, thresholds, tracks, and per-sequence metrics are saved in [the completed run](../../outputs/tracker_detector_baselines/20261124_141629_000779/comparison.md) and its `baseline_manifest.json`. All 12 arms finished and all four source cache hashes were unchanged afterward.

| Detector | Appearance | GCN | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tracktor | Frozen OSNet | On | 43.074 | 47.282 | 39.280 | 49.676 | 49.590 | 396 |
| Tracktor | Frozen OSNet | Off | 44.206 | 47.412 | 41.253 | 51.182 | 50.310 | 226 |
| Tracktor | Fine-tuned OSNet | On | 43.895 | 47.392 | 40.692 | 50.928 | 50.096 | 275 |
| **Tracktor** | **Fine-tuned OSNet** | **Off** | **44.229** | **47.390** | **41.314** | **51.232** | **50.280** | **233** |
| Tracktor | DenseNet121 | On | 38.285 | 47.035 | 31.206 | 43.415 | 48.578 | 638 |
| Tracktor | DenseNet121 | Off | 42.816 | 47.200 | 38.884 | 49.930 | 50.209 | 250 |
| YOLOX | Frozen OSNet | On | 36.500 | 42.552 | 31.482 | 45.502 | 48.649 | 754 |
| YOLOX | Frozen OSNet | Off | 38.072 | 42.905 | 33.950 | 48.265 | 49.946 | 458 |
| YOLOX | Fine-tuned OSNet | On | 37.822 | 43.003 | 33.449 | 47.095 | 49.327 | 607 |
| **YOLOX** | **Fine-tuned OSNet** | **Off** | **39.186** | **42.576** | **36.244** | **49.947** | **50.117** | **425** |
| YOLOX | DenseNet121 | On | 34.159 | 41.527 | 28.266 | 41.416 | 46.143 | 1,266 |
| YOLOX | DenseNet121 | Off | 36.934 | 42.634 | 32.171 | 46.733 | 49.360 | 576 |

Graph bypass improves HOTA for every detector/encoder pair. Fine-tuned OSNet without GCN is the highest HOTA arm for both detectors, although its Tracktor lead over frozen OSNet without GCN is only **0.023 points**. Tracktor remains ahead of YOLOX by **5.043 HOTA points** in their best arms. On ideal GT boxes, the fine-tuned OSNet/no-GCN control scored **85.006 HOTA**; the much lower real-box scores show that detector coverage and localization still constrain the full pipeline.

The older **direct-cosine** frozen-OSNet controls, using matched 0.5/0.7 detection and birth thresholds, scored **44.734 HOTA** on Tracktor and **42.045 HOTA** on YOLOX. They are a different matcher, so they are listed separately from the 12 learned-Sinkhorn arms. The new best arms do not surpass those direct-cosine controls on real boxes, despite the learned matcher doing much better on the ideal GT-box control.

`without_gcn` still uses the checkpoint's learned Sinkhorn matcher and assignment; it is not the separate direct-cosine tracker. Both fine-tuned OSNet arms use an association checkpoint trained on frozen OSNet features, so they are **transfer comparisons**, not a jointly fine-tuned model. The graph checkpoints were trained for four epochs on Tracktor detections. Tracktor cache confidence is a placeholder, whereas YOLOX confidence is real; the 0.5 threshold has different meaning for the two sources. The same validation sequences have been repeatedly used for tuning, so these numbers are internal baselines rather than independent test-set estimates.

To repeat the exact matrix with current caches and weights:

```bash
python evaluate-tracker-variants.py --smoke-test
python evaluate-tracker-variants.py
```

Use [the evaluation guide](37-tracker-encoder-graph-evaluation.md) for selecting one detector/encoder arm or changing score thresholds. Store altered thresholds or checkpoints in a separate output folder and compare against this fixed baseline.
