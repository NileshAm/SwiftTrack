# Fixed local comparison readings

These are the measured baseline readings for **MOT17-02-SDP and MOT17-09-SDP**, the same validation videos used repeatedly during development. They are not official MOT17 test results. The authors' GitHub baseline has no local score yet; run its prepared scripts and add its row only after evaluation.

The tracker rows below all use the fine-tuned OSNet encoder without GCN, the existing frozen-OSNet association checkpoint, and its learned Sinkhorn matcher. Thus the fine-tuned encoder is a transfer into the frozen association model, not an association model trained specifically for those new features. Detection read score is 0.5; new-track score is 0.7. GT identities are hidden from every tracker and used only for scoring. Tracktor's imported scores are placeholders of 1.0.

| Box source | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GT boxes, no IDs (upper control) | 85.006 | 96.915 | 74.565 | 84.046 | 97.398 | 528 |
| Tracktor | 44.229 | 47.390 | 41.314 | 51.232 | 50.280 | 233 |
| YOLOX-X CrowdHuman + MOT20 pretrained | 39.186 | 42.576 | 36.244 | 49.947 | 50.117 | 425 |

Detector-only quality on the same validation set at IoU 0.5 and score 0.5:

| Source | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GT person boxes (ideal input; by construction) | 100% | 100% | 100% | 23,906 | 0 | 0 |
| Tracktor | 99.61% | 51.46% | 67.86% | 12,301 | 48 | 11,605 |
| YOLOX-X | 92.89% | 56.64% | 70.37% | 13,541 | 1,037 | 10,365 |

The detector F1 and tracker HOTA measure different things. YOLOX finds more people, but its extra false positives and box/identity consistency reduce tracking performance under these tracker settings. The gap between GT boxes and either detector input shows that detection quality is the largest measured constraint in this setup; it does not by itself prove a better association model could not help.

Sources: [GT tracker comparison](../outputs/tracker_gt_variants/20261124_135704_391539/comparison.md), [detector tracker comparison](../outputs/tracker_detector_baselines/20261124_141629_000779/comparison.md), [Tracktor detector report](../outputs/detector_evaluation/20261008_192911_780353_tracktor/report.md), [YOLOX detector report](../outputs/detector_evaluation/20261123_215328_261233_yolox/report.md). Generated outputs may not be version controlled; retain the run folders when making the final report.

To regenerate the three tracker rows with the current implementation, use the comparison script documented in the project README. To produce the authors' baseline for this same validation split, use [the pinned GitHub baseline](../Baselines/DragonTrackGithub/README.md).
