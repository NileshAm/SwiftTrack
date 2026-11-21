# SwiftTrack: implementation, experiments, tests, and benchmark results

Reviewed on **21 November 2026**, from the code, Git history through `aee1197`, and artifacts currently present under `outputs/` and `detections/`.

## 1. Main findings

You built a DragonTrack-derived multi-person tracking pipeline, then investigated detection quality, association matching, temporal memory, graph feature collapse, residual attention, training speed, and parameter selection.

- **Best saved result on held-out complete videos:** residual attention model, **HOTA 41.794%, IDF1 47.971%, MOTA 49.870%, 331 identity switches**.
- Compared with the preceding tuned model, the residual model improved HOTA by **4.329 percentage points**, IDF1 by **6.537 points**, and MOTA by **1.786 points**, while reducing identity switches from **756 to 331** (**56.22% fewer**). These are observed differences between saved runs, not an estimate across repeated random seeds.
- The diagnostic explains the motivation for this change: the earlier graph reduced strict identity ranking from **84.66% to 27.87%**, with **50.35% ties**. Residual attention achieved **88.55% strict ranking and 0% ties**.
- You completed **52 association/tracking search trials** across seven successful searches, plus one recorded failed search attempt; you also completed a **54-trial RT-DETR detector search**.
- Best eligible RT-DETR search point: **85.20% precision, 52.46% recall, 64.93% F1**, at detector threshold **0.30**, 960-pixel square input, trial 052, epoch 2.
- DDQ four-scale and five-scale full-validation experiments achieved **60.57%** and **61.44% F1**, respectively. These are detector results, not tracking scores.

This report summarizes existing runs. No training, inference, or benchmark was rerun during this review. “History” here means repository commits and saved experiment histories; unavailable terminal/chat history is not reconstructed.

## 2. Evaluation protocols and metric meaning

The main later protocol trains on five complete MOT17 videos and validates on two separate complete videos. Detector variants of the same underlying video must remain in the same split.

| Role | Sequences | Frames |
| --- | --- | --- |
| Training | MOT17-04-SDP, 05-SDP, 10-SDP, 11-SDP, 13-SDP | 1,050 + 837 + 654 + 900 + 750 = 4,191 |
| Validation | MOT17-02-SDP and MOT17-09-SDP | 600 + 525 = 1,125 |
| Unlabelled test inference | MOT17-01-SDP, 03-SDP, 06-SDP, 07-SDP | Full videos where recorded; no scored test GT |

The earliest `mot17_attention` experiment used temporal portions of the same seven FRCNN videos: first 80% for training, next 10% for validation. Its **58.275 HOTA** is therefore a different evaluation protocol and must not be ranked above the held-out-video runs as if conditions were equivalent. The first RT-DETR video-split run also used only two training videos.

- **HOTA:** combined detection and association quality; higher is better.
- **IDF1:** identity consistency measured through identity precision/recall; higher is better.
- **MOTA:** aggregate missed detections, false positives, and identity switches; higher is better.
- **IDSW:** identity-switch count; lower is better, but very low counts can accompany poor coverage.
- **Detector precision/recall/F1:** box matching at the stated IoU; these are not COCO AP or tracking metrics.

Tracking evaluation uses **TrackEval 1.3.0 with MOT preprocessing**. Detector diagnostics use marked class-1 pedestrians, maximum-count IoU matching, and ignored-label handling. All detector comparisons below cover 1,125 frames and 23,906 target pedestrian instances. Percentage metrics are displayed on a 0–100 scale. Detector JSON and detailed TrackEval CSV store rates on a 0–1 scale.

The validation videos were reused for checkpoint selection and parameter tuning. These are local validation benchmarks, not official MOTChallenge test-set scores or independent final generalization estimates.

## 3. What you implemented

| Component | Implementation and purpose | Main source files |
| --- | --- | --- |
| MOT data and labels | Sequence/image/GT loading, MOT one-origin xywh to internal xyxy conversion, valid/ignored labels, IoU assignment, and disjoint whole-video splits. | [data.py](<Utils/data.py>); [splits.py](<Utils/splits.py>) |
| Appearance extraction | Initial frozen DenseNet121 crop path; later full-image pretrained DETR-ResNet50 encoder, ROIAlign 3×3 mean pooling and normalized 256-dimensional box features; frozen and trainable modes; real image batching. | [features.py](<Utils/features.py>); [appearance.py](<Utils/appearance.py>) |
| Graph construction | Track and detection nodes combine appearance with normalized box coordinates; spatial candidate gate; two directed edges per allowed pair; displacement, distance, IoU and cosine edge attributes. | [graph.py](<Utils/graph.py>) |
| Learned relationships and attention | Separate appearance, geometry and positional affinities; two attention layers with widths 512 and 128. Current v5 adds a learned self-feature projection in each layer. | [relationships.py](<Utils/relationships.py>); [attention.py](<Utils/attention.py>); [model.py](<model.py>) |
| Association matching | Cosine scores, learned unmatched preference, log-space Sinkhorn with fixed marginals, masked/isolated handling and final feasibility correction; Hungarian assignment with unmatched costs. | [sinkhorn.py](<Utils/sinkhorn.py>); [tracking.py](<tracking.py>) |
| Training objective | Weighted BCE and contrastive learning aligned to normalized cosine embeddings; hard/random negative sampling; per-graph objective plus per-pair diagnostics. | [losses.py](<Utils/losses.py>); [pairs.py](<Utils/pairs.py>); [train.py](<train.py>) |
| Temporal state | Past-only chronological graph scoring; GT-guided updates after scoring in training/diagnostics; per-track constant-velocity Kalman prediction and bounded confidence-weighted appearance memory. | [track_state.py](<Utils/track_state.py>); [training_state.py](<Utils/training_state.py>); [motion.py](<Utils/motion.py>); [appearance_history.py](<Utils/appearance_history.py>) |
| Tracking output | 30-frame observation gap, track creation/expiry, configurable hit confirmation, observed detection boxes in MOT output, previews and optional videos. | [tracking.py](<tracking.py>); [track_filter.py](<Utils/track_filter.py>) |
| Detector sources and caching | Raw supplied detections, authors-linked Tracktor preprocessing import, RT-DETRv2 inference at square or original resolution, person-only export and checked completion/provenance manifests. | [import_tracktor.py](<import_tracktor.py>); [detect.py](<detect.py>); [detections.py](<Utils/detections.py>) |
| Evaluation and plots | Whole-video TrackEval reports, per-video/combined scores and error plots; detector confidence/IoU sweeps and size/visibility recall; training and feature-diagnostic charts. | [evaluate.py](<evaluate.py>); [evaluate_detector.py](<evaluate_detector.py>); [plots.py](<Utils/plots.py>); [test.py](<test.py>) |
| Search and recovery | Independent association training/tracking grids, detector report ranking and training search, best/latest checkpoints, resume state, backups and trial completion verification. | [search.py](<search.py>); [detector-search.py](<detector-search.py>); [training_resume.py](<Utils/training_resume.py>); [detector_selection.py](<Utils/detector_selection.py>) |
| RT-DETR fine-tuning | Held-out validation, ignored-region classification handling, augmentation, accumulation, backbone LR groups, pretrained epoch-zero comparison and portable best-model export. | [detector-train.py](<detector-train.py>); [detector_training.py](<Utils/detector_training.py>) |
| DDQ inference/comparison | Four/five-scale MMDetection DDQ, strict checkpoint loading with verified alias handling, aspect/square input modes, MOT cache export and shared detector evaluation. | [detect-ddq.py](<detect-ddq.py>); [ddq.py](<Utils/ddq.py>); [compare-detectors.py](<compare-detectors.py>) |
| DDQ fine-tuning code | MOT collator and ignored-target adapters, AdamW training, accumulation, activation checkpointing of suitable blocks, validation, resume and best.pth export. No saved training benchmark found. | [detector-train-ddq.py](<detector-train-ddq.py>); [ddq_training.py](<Utils/ddq_training.py>) |

Current association defaults use frozen pretrained DETR, a 25-pixel gate, a 30-frame look-back, history size 30, motion/history enabled, match threshold 0.05, and minimum confirmation hits 1. Current config values describe the present recipe, not every historical run. The residual layer computes a learned self projection plus the rectified neighbor message; it preserves signed self features even for isolated nodes.

The pipeline remains a locally modified, partial DragonTrack reproduction. Teacher-forced training memory does not simulate all assignment mistakes encountered during free-running tracking.

## 4. Development history

Commit dates show when work was recorded in Git; they are not necessarily execution dates. Several output folders have earlier timestamps than the commits that document them.

| Commit | Date | Recorded change |
| --- | --- | --- |
| 121b2f7 | 2026-09-27 | Initial commit |
| de71bd7 | 2026-09-27 | Update README.md to include acknowledgements and licensing information |
| 0bddade | 2026-09-27 | Add implementation documentation and update .gitignore for datasets |
| 2776be3 | 2026-09-27 | Update documentation to reflect the creation of Utils/config.py and adjust references throughout the implementation files |
| 44ca53c | 2026-09-29 | Add MOT data loading and whole-video split configuration |
| 2209403 | 2026-09-30 | Add frozen appearance features for person detections |
| f880efa | 2026-10-01 | Implement graph attention and learnable association matching |
| 2bd3fe5 | 2026-10-04 | Add chronological association training and supporting utilities |
| 73bf10e | 2026-10-04 | Document training checkpoints and temporal look-back |
| cb7cfb0 | 2026-10-07 | Add detection cache generation and paper-alignment guides |
| 2559fd7 | 2026-10-08 | Add full-video tracking evaluation and result plots |
| 2ba231c | 2026-10-08 | Preserve node appearance through graph attention and add diagnostics |
| 2fbaa6f | 2026-10-08 | Add resumable parameter search and association training recovery |
| eb49f3a | 2026-10-08 | Add detector evaluation and configurable inference resolution |
| 34b9d53 | 2026-10-08 | Document the pipeline and update implementation progress |
| aee1197 | 2026-11-11 | ddq implemented |

## 5. Tracking experiments and benchmark results

| Run / result source | Experiment | Last logged epoch | HOTA % | IDF1 % | MOTA % | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| [mot17_attention](<outputs/mot17_attention/evaluation/val/results.md>) | Early attention; temporal FRCNN split | 20 | 58.275 | 63.069 | 48.837 | 63 |
| [mot17_rtdetrv2_s_video_split](<outputs/mot17_rtdetrv2_s_video_split/evaluation/val/results.md>) | RT-DETR; two training videos | 20 | 24.362 | 25.244 | 6.542 | 1156 |
| [mot17_rtdetrv2_s_video_split_full](<outputs/mot17_rtdetrv2_s_video_split_full/evaluation/val/results.md>) | RT-DETR; five training videos | 20 | 26.480 | 28.642 | 10.357 | 954 |
| [mot17_SDP_video_split](<outputs/mot17_SDP_video_split/evaluation/val/results.md>) | Supplied SDP; five training videos | 20 | 32.667 | 37.368 | 43.198 | 1202 |
| [mot17_paper_tracktor](<outputs/mot17_paper_tracktor/evaluation/paper_motion_history/val/results.md>) | Paper-inspired attention + Tracktor; trainable appearance | 20 | 13.956 | 10.353 | 7.488 | 294 |
| [mot17_paper_tracktor_frozen](<outputs/mot17_paper_tracktor_frozen/evaluation/paper_motion_history/val/results.md>) | Restore frozen pretrained appearance | 20 | 16.472 | 15.999 | 11.809 | 781 |
| [mot17_fixed_matching](<outputs/mot17_fixed_matching/evaluation/paper_motion_history/val/results.md>) | Fixed-marginal matching + normalized contrastive loss | 20 | 11.516 | 7.455 | 4.564 | 68 |
| [mot17_tuned_full_v1](<outputs/mot17_tuned_full_v1/evaluation/paper_motion_history/val/results.md>) | Tuned gate/confirmation; full training | 20 | 37.465 | 41.434 | 48.084 | 756 |
| [mot17_appearance_residual_v1](<outputs/mot17_appearance_residual_v1/evaluation/paper_motion_history/val/results.md>) | Projected self-feature residual attention | 30 | 41.794 | 47.971 | 49.870 | 331 |

**Interpretation of the experiments:**

1. Increasing RT-DETR training coverage from two videos to five increased HOTA from **24.362 to 26.480**. Switching the separate saved run to supplied SDP detections gave **32.667 HOTA**. These compare complete run outcomes, with detector/model settings recorded in their artifacts.
2. The paper-inspired Tracktor run and its frozen-appearance variant performed poorly (**13.956** and **16.472 HOTA**). Freezing improved that saved result, but did not resolve association quality.
3. The fixed-marginal repair made matching probabilities mathematically valid, but its untuned tracking result was only **11.516 HOTA**. Correct normalization alone did not establish better tracking.
4. Gate/confirmation tuning was followed by `mot17_tuned_full_v1`, reaching **37.465 HOTA**. Its saved tracker uses a **25-pixel** gate and **one-hit** confirmation, versus **200 pixels** and **three-hit** confirmation in the fixed-matching run. This comparison includes multiple changes, so it does not isolate one parameter's causal effect.
5. The residual model achieved the strongest held-out-video result, **41.794 HOTA / 47.971 IDF1**, alongside much better graph ranking and fewer switches.

**Epoch provenance:** eight run histories end at epoch 20; the residual history now ends at **epoch 30** and contains a saved epoch-20 history backup. The residual diagnostic and threshold-search records explicitly refer to **epoch 20**. The main tracking report does not pin an epoch or checkpoint hash, so its saved 41.794 HOTA result must not be presented as a newly verified epoch-30 evaluation. Current comments mentioning a future epoch-40 target are stale; config currently sets `EPOCHS=30`.

### Best tracking run: per-video results

| Sequence | HOTA % | DetA % | AssA % | IDF1 % | MOTA % | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-SDP | 39.036 | 43.198 | 35.320 | 45.264 | 46.026 | 17 | 9802 | 210 |
| MOT17-09-SDP | 50.183 | 60.392 | 41.725 | 56.319 | 63.286 | 31 | 1803 | 121 |
| COMBINED | 41.794 | 47.043 | 37.168 | 47.971 | 49.870 | 48 | 11605 | 331 |

Source: [pedestrian_detailed.csv](<outputs/mot17_appearance_residual_v1/evaluation/paper_motion_history/val/MOT17/DragonTrack/pedestrian_detailed.csv>). Combined metrics are TrackEval aggregates, not simple averages of the video percentages.

## 6. Association and tracking parameter searches

| Search / full ranking | Completed | Epoch budget | Training frames/video | Varied parameters | Best HOTA % | Best IDF1 % |
| --- | --- | --- | --- | --- | --- | --- |
| [association_search2_20261003_161703_821353](<outputs/searches/association_search2_20261003_161703_821353/results.csv>) | 5 | 5 | full / not recorded | MATCH_THRESHOLD=0.01,0.025,0.05,0.09,0.75 | 21.654 | 20.240 |
| [association_search3_20261003_225614_077222](<outputs/searches/association_search3_20261003_225614_077222/results.csv>) | 27 | 3 | 400 | MATCH_THRESHOLD=0.025,0.05,0.75; TRACK_CONFIRM_MIN_HITS=1,2,3; ASSOCIATION_DISTANCE_PX=100,200,300 | 27.353 | 27.588 |
| [association_search4_20261004_101321_734762](<outputs/searches/association_search4_20261004_101321_734762/results.csv>) | 4 | 3 | 400 | ASSOCIATION_DISTANCE_PX=100,25,50,75 | 36.989 | 40.165 |
| [association_search5_20261004_114439_372694](<outputs/searches/association_search5_20261004_114439_372694/results.csv>) | 4 | 3 | 400 | ASSOCIATION_DISTANCE_PX=10,20,25,5 | 37.582 | 42.367 |
| [association_search_20261003_120918_760750](<outputs/searches/association_search_20261003_120918_760750/results.csv>) | 4 | 5 | full / not recorded | MATCH_THRESHOLD=0.05,0.1,0.2,0.35 | 23.031 | 22.458 |
| [tracking_match_thresholds2_20261007_101827_969084](<outputs/searches/tracking_match_thresholds2_20261007_101827_969084/results.csv>) | 4 | reuse checkpoint | full / not recorded | MATCH_THRESHOLD=0.05,0.3,0.5,0.7 | 41.794 | 47.971 |
| [tracking_match_thresholds_20261007_100244_226092](<outputs/searches/tracking_match_thresholds_20261007_100244_226092/results.csv>) | 4 | reuse checkpoint | full / not recorded | MATCH_THRESHOLD=0.02,0.05,0.1,0.2 | 41.794 | 47.971 |

The first two association searches predate the CSV training-frame-limit field; their search records have no prefix limit. Searches 3–5 explicitly use **the first 400 frames of each training video**, while validation remains complete. Every `mode=train` trial trains independently, even when the varied setting affects only tracking; those rows are not pure inference-only ablations. All successful association searches use learning rate `1e-4`.

The coarse gate grid tested **100/200/300 pixels**; the next grid tested **25/50/75/100**, and the last **5/10/20/25**. The last two best points both used **25 pixels**, with HOTA **36.989** and **37.582**. Their differing outcomes also show that separately trained short runs need not produce identical results.

A failed search at `association_search5_20261004_114401_504542/trial_001` recorded `ValueError: ASSOCIATION_DISTANCE_PX must be positive or None`. It produced no scored trial and is excluded from the completed count. Source: [error.txt](<outputs/searches/association_search5_20261004_114401_504542/trial_001/error.txt>).

The two residual tracking searches reuse an epoch-20 checkpoint and test thresholds **0.02, 0.05, 0.10, 0.20** and **0.05, 0.30, 0.50, 0.70**. Thresholds **0.02 through 0.20 all gave the same 41.794 HOTA / 47.971 IDF1 / 331 switches**. Raising the threshold to **0.30, 0.50 and 0.70** reduced HOTA to **39.668, 36.912 and 26.715**, while switches increased to **1,700, 2,860 and 6,562**. Their full results are reproduced in Appendix A. Search wall time covers the trial workflow and must not be interpreted as inference FPS.

## 7. Appearance-collapse diagnostic and residual experiment

`test.py` compares the same labelled candidate population before and after graph attention, with GT-guided updates after each scored frame. Both saved diagnostics cover the full validation videos and epoch-20 checkpoints.

### mot17_tuned_full_v1

| Video | Stage | Same-person mean | Different-person mean | Gap | Strict top-1 (%) | Ties (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MOT17-02-SDP | before | 0.9842 | 0.9547 | 0.0295 | 84.7059 | 0.0000 |
| MOT17-02-SDP | after | 0.9790 | 0.9499 | 0.0291 | 27.8588 | 50.3059 |
| MOT17-09-SDP | before | 0.9488 | 0.9179 | 0.0309 | 66.6667 | 0.0000 |
| MOT17-09-SDP | after | 0.9573 | 0.9353 | 0.0221 | 33.3333 | 66.6667 |
| COMBINED | before | 0.9742 | 0.9537 | 0.0205 | 84.6551 | 0.0000 |
| COMBINED | after | 0.9729 | 0.9495 | 0.0234 | 27.8742 | 50.3519 |

Source: [report.md](<outputs/mot17_tuned_full_v1/feature_diagnostics/20261004_182420_727040/report.md>).

### mot17_appearance_residual_v1

| Video | Stage | Same-person mean | Different-person mean | Gap | Strict top-1 (%) | Ties (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MOT17-02-SDP | before | 0.9842 | 0.9547 | 0.0295 | 84.7059 | 0.0000 |
| MOT17-02-SDP | after | 0.9720 | 0.5928 | 0.3792 | 88.5176 | 0.0000 |
| MOT17-09-SDP | before | 0.9488 | 0.9179 | 0.0309 | 66.6667 | 0.0000 |
| MOT17-09-SDP | after | 0.9866 | 0.8231 | 0.1635 | 100.0000 | 0.0000 |
| COMBINED | before | 0.9742 | 0.9537 | 0.0205 | 84.6551 | 0.0000 |
| COMBINED | after | 0.9761 | 0.5988 | 0.3773 | 88.5500 | 0.0000 |

Source: [report.md](<outputs/mot17_appearance_residual_v1/feature_diagnostics/20261006_230501_671762/report.md>).

The residual addition increased the combined post-graph same/different-person similarity gap from **0.0234 to 0.3773**, and strict top-1 from **27.8742% to 88.5500%**. Ties fell from **50.3519% to 0%**. The accompanying tracking result supports the practical value of this implementation on these validation videos.

These rankings include only tracks with an allowed correct match and an allowed labelled competitor. They are neither detector recall nor end-to-end identity accuracy. GT-guided memory omits error propagation from wrong past tracking assignments.

## 8. Detector-source and input-resolution experiments

Each row below selects the best **recorded** F1 threshold at IoU 0.5 from that report. This is a finite threshold sweep, not continuous optimization. Two reports reuse the same 640 cache but sweep different thresholds; both are listed to preserve the experiment history.

| Detector / source report | Threshold | Precision % | Recall % | F1 % | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Tracktor-preprocessed public SDP](<outputs/detector_evaluation/20261008_192911_780353_tracktor/summary.json>) | all | 99.61 | 51.46 | 67.86 | 12301 | 48 | 11605 |
| [Pretrained RT-DETRv2-S, square 640, first sweep](<outputs/detector_evaluation/20261008_192911_788120_rtdetrv2/summary.json>) | 0.3 | 61.49 | 49.87 | 55.08 | 11923 | 7467 | 11983 |
| [Raw supplied SDP](<outputs/detector_evaluation/20261008_192911_792492_supplied/summary.json>) | all | 93.60 | 51.98 | 66.84 | 12426 | 850 | 11480 |
| [Pretrained RT-DETRv2-S, original resolution](<outputs/detector_evaluation/20261008_194449_778386_rtdetrv2/summary.json>) | 0.3 | 33.26 | 23.55 | 27.58 | 5630 | 11296 | 18276 |
| [Pretrained RT-DETRv2-S, square 960](<outputs/detector_evaluation/20261008_195407_928650_rtdetrv2/summary.json>) | 0.4 | 74.70 | 47.39 | 57.99 | 11329 | 3837 | 12577 |
| [Same RT-DETR 640 cache, expanded sweep](<outputs/detector_evaluation/20261008_200012_220126_rtdetrv2/summary.json>) | 0.4 | 77.33 | 44.40 | 56.41 | 10614 | 3112 | 13292 |
| [COCO DDQ-DETR-R50, four-scale](<outputs/detector_evaluation/20261009_182612_860670_ddq/summary.json>) | 0.25 | 78.90 | 49.15 | 60.57 | 11749 | 3142 | 12157 |
| [COCO DDQ-DETR-R50, five-scale](<outputs/detector_evaluation/20261009_184855_394294_ddq/summary.json>) | 0.25 | 76.82 | 51.19 | 61.44 | 12237 | 3693 | 11669 |

**Findings:**

- Tracktor has the highest detector F1 among these source reports (**67.86%**), with **99.61% precision** but **51.46% recall**. Missing Tracktor confidence is substituted with 1, so its confidence sweep is not calibrated like a neural detector's scores.
- Square 960 improves pretrained RT-DETR's best recorded F1 to **57.99%**, versus **56.41%** for the expanded 640 sweep.
- Preserving original image resolution did not help this checkpoint: its best recorded F1 was **27.58%**, and cache generation was slower. More pixels alone did not improve the saved detector result.
- DDQ five-scale adds **488 true positives** and **551 false positives** over four-scale at threshold 0.25: recall rises **2.04 points**, F1 rises **0.87 points**, and precision falls **2.08 points**.

DDQ uses aspect-preserving short side 800 / long side cap 1333, while RT-DETR comparisons use their recorded square/original recipes. This is a comparison of usable model recipes, with differing preprocessing and training histories, rather than a controlled architecture-only comparison.

### Small and occluded people at each report's selected F1 threshold

| Source | Threshold | Height <50 px recall % | Height 50–<100 px recall % | Height ≥100 px recall % | Visibility <0.3 recall % |
| --- | --- | --- | --- | --- | --- |
| Tracktor-preprocessed public SDP | all | 46.08 | 31.70 | 61.68 | 5.22 |
| Pretrained RT-DETRv2-S, square 640, first sweep | 0.3 | 0.94 | 28.94 | 64.36 | 18.25 |
| Raw supplied SDP | all | 36.44 | 33.60 | 62.38 | 16.66 |
| Pretrained RT-DETRv2-S, original resolution | 0.3 | 5.33 | 13.18 | 30.22 | 8.57 |
| Pretrained RT-DETRv2-S, square 960 | 0.4 | 2.82 | 26.95 | 61.26 | 17.03 |
| Same RT-DETR 640 cache, expanded sweep | 0.4 | 0.31 | 17.99 | 61.18 | 12.25 |
| COCO DDQ-DETR-R50, four-scale | 0.25 | 0.86 | 27.12 | 64.11 | 17.24 |
| COCO DDQ-DETR-R50, five-scale | 0.25 | 0.94 | 29.48 | 66.16 | 18.42 |

The selected F1 points still miss many small and heavily occluded people. These rows use different precision levels; use the full confidence sweeps for matched-precision comparisons. The two DDQ [comparison analyses](<outputs/detector_evaluation/20261009_184855_394294_ddq/analysis.md>) also discuss the fine-tuned RT-DETR search winner.

## 9. RT-DETR detector fine-tuning

Three standalone training histories are present. The 640 runs select checkpoints at fixed detector threshold **0.4**. The 960 full-data run sweeps operating points; its baseline and best epoch therefore use different selected thresholds.

| Run / history | Resolution; batch×accum. | Trained epochs | Baseline F1 % | Best epoch | Best threshold | Best precision % | Best recall % | Best F1 % | Final F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [detector_rtdetrv2_s_mot17_v1](<outputs/detector_rtdetrv2_s_mot17_v1/history.csv>) | 640; 2×4 | 10 | 56.41 | 7 | 0.4 | 91.58 | 40.93 | 56.57 | 56.06 |
| [detector_rtdetrv2_s_mot17_v1(batch1)](<outputs/detector_rtdetrv2_s_mot17_v1(batch1)/history.csv>) | 640; 1×4 | 10 | 56.41 | 0 | 0.4 | 77.33 | 44.40 | 56.41 | 51.13 |
| [detector_full_winner_5ep](<outputs/detector_full_winner_5ep/history.csv>) | 960; 2×2 | 5 | 57.99 | 1 | 0.225 | 74.68 | 55.22 | 63.49 | 58.83 |

Both 640 runs use LR `5e-5`, backbone LR `5e-6`, weight decay `1e-4`, and all training frames. Batch 2 × accumulation 4 has effective batch 8; batch 1 × accumulation 4 has effective batch 4. This experiment changes effective batch as well as the physical batch.

The batch-2 640 run barely improves its fixed-threshold baseline (**56.41% → 56.57% F1**); the batch-1 run's best point remains pretrained epoch zero. Threshold ranking across the 640 epoch reports finds **60.21% F1**, epoch 2 at threshold 0.2, but the report marks that epoch's weights **not retained**. A report row cannot recreate those weights. Source: [report.md](<outputs/detector_search_reports/1791485580099450401/report.md>).

The full-data 960 run uses batch 2 × accumulation 2, LR `5e-5`, backbone LR `5e-6`, and weight decay `1e-5`. Best F1 is **63.49% at epoch 1 / threshold 0.225**; epoch 5 ends at **58.83%**. More epochs did not improve the saved validation result.

## 10. RT-DETR detector hyperparameter search

The recorded search contains **54 planned and 54 completed trials**, each trained for **3 epochs**, using **40% of each training video, sampled evenly across the video**. Validation uses all 1,125 frames. This sampling differs from the association search's first-400-frame prefixes.

| Parameter | Recorded grid |
| --- | --- |
| Learning rate | 1e-5, 2e-5, 5e-5 |
| Backbone learning rate | 0.1 × main learning rate |
| Image size | 640, 960 |
| Weight decay | 1e-3, 1e-4, 1e-5 |
| Physical batch size | 2 |
| Gradient accumulation | 2, 4, 8 (effective batches 4, 8, 16) |
| Detector thresholds | 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5 |
| Eligibility floors | Precision ≥70%; recall ≥50% |

Best eligible and retained point: **trial 052, epoch 2, threshold 0.3**, precision **85.20%**, recall **52.46%**, F1 **64.93%**; TP **12,540**, FP **2,179**, FN **11,366**. Settings: image size **960**, LR **5e-5**, backbone LR **5e-6**, weight decay **1e-5**, batch **2**, accumulation **2**.

At threshold **0.25**, the same epoch gives **77.27% precision, 55.76% recall, 64.77% F1**. This point meets the stricter 70% precision / 55% recall comparison target. Neither pretrained DDQ sweep has a point meeting both stricter floors.

The winner outperforms the five-epoch full-data run at their best saved validation points (**64.93% vs 63.49% F1**). This does not prove less training data is generally better: sampling, validation selection and training duration differ. No saved association retraining/evaluation with this detector winner was found.

Sources: [search.json](<outputs/detector_search/detector_search_lr_0.4/search.json>), [best.json](<outputs/detector_search/detector_search_lr_0.4/best.json>), [ranking.csv](<outputs/detector_search/detector_search_lr_0.4/ranking.csv>). Appendix B lists the best eligible-or-fallback report point for every completed detector trial.

## 11. Runtime and memory benchmarks

### Actual saved tracking frame-loop throughput

FPS below is recomputed as total validation frames divided by summed recorded seconds, not an average of per-video FPS. These timings use cached detections and exclude detector inference and model/metadata loading. Their recorded scope includes image reads and frame-loop output, and excludes final preview/video work; they are not end-to-end camera-to-video throughput.

| Run / example timing source | Validation frames | Seconds | Combined FPS |
| --- | --- | --- | --- |
| [mot17_attention](<outputs/mot17_attention/tracking/val/MOT17-02-FRCNN/summary.json>) | 531 | 60.59 | 8.76 |
| [mot17_rtdetrv2_s_video_split](<outputs/mot17_rtdetrv2_s_video_split/tracking/val/MOT17-02-SDP/summary.json>) | 1125 | 138.72 | 8.11 |
| [mot17_rtdetrv2_s_video_split_full](<outputs/mot17_rtdetrv2_s_video_split_full/tracking/val/MOT17-02-SDP/summary.json>) | 1125 | 138.54 | 8.12 |
| [mot17_SDP_video_split](<outputs/mot17_SDP_video_split/tracking/val/MOT17-02-SDP/summary.json>) | 1125 | 176.64 | 6.37 |
| [mot17_paper_tracktor](<outputs/mot17_paper_tracktor/tracking/paper_motion_history/val/MOT17-02-SDP/summary.json>) | 1125 | 124.94 | 9.00 |
| [mot17_paper_tracktor_frozen](<outputs/mot17_paper_tracktor_frozen/tracking/paper_motion_history/val/MOT17-02-SDP/summary.json>) | 1125 | 126.18 | 8.92 |
| [mot17_fixed_matching](<outputs/mot17_fixed_matching/tracking/paper_motion_history/val/MOT17-02-SDP/summary.json>) | 1125 | 127.95 | 8.79 |
| [mot17_tuned_full_v1](<outputs/mot17_tuned_full_v1/tracking/paper_motion_history/val/MOT17-02-SDP/summary.json>) | 1125 | 128.87 | 8.73 |
| [mot17_appearance_residual_v1](<outputs/mot17_appearance_residual_v1/tracking/paper_motion_history/val/MOT17-02-SDP/summary.json>) | 1125 | 126.94 | 8.86 |

### Detector cache-generation throughput

| Detector recipe | Seconds / 1,125 frames | Combined FPS |
| --- | --- | --- |
| Pretrained RT-DETRv2-S, square 640, first sweep | 50.45 | 22.30 |
| Pretrained RT-DETRv2-S, original resolution | 121.02 | 9.30 |
| Pretrained RT-DETRv2-S, square 960 | 67.31 | 16.71 |
| Same RT-DETR 640 cache, expanded sweep | 50.45 | 22.30 |
| COCO DDQ-DETR-R50, four-scale | 216.59 | 5.19 |
| COCO DDQ-DETR-R50, five-scale | 633.61 | 1.78 |

Cache timings include image reads, inference, export and preview, and exclude model loading. The two 640 rows reuse the same cache timings. Five-scale DDQ runs at about **1.78 FPS**, versus **5.19 FPS** for four-scale, for a **0.87-point F1** gain. Hardware/package provenance in the manifests records CUDA execution, PyTorch 2.6.0+cu124, Transformers 4.49.0 for RT-DETR and MMDetection 3.3.0 / MMCV 2.1.0 / MMEngine 0.10.7 for DDQ.

### Historical controlled image-batching check

The implementation notes record an RTX 4060 Laptop GPU check with FP32 DETR, short side 800 / maximum 1333, two CPU threads, and 12 current MOT17-04 frames after warm-up. Scope includes image reads, preprocessing, chronological graph/loss computation and optimizer updates.

| Training image batch | Activation checkpointing | Frames/s | Peak PyTorch allocation |
| --- | --- | --- | --- |
| 1 | On | 3.11–3.20 | 1.38 GiB |
| 2 | On | 3.16 | 2.15 GiB |
| 4 | On | 3.13 | 3.76 GiB |
| 1 | Off | 3.87 | 1.56 GiB |
| 2 | Off | 3.96–3.98 | 2.73 GiB |
| 4 | Off | 3.97 | 5.07 GiB |

Disabling checkpointing improved this short-run throughput by roughly **25%**; batch 2 matched batch 4 with less memory. These are historical documented measurements, not newly rerun benchmarks or whole-dataset peak memory. Source: [18-batched-training.md](<Documentation/Implementation/18-batched-training.md>).

## 12. Training-history evidence

Losses below are the final logged values and the minimum logged validation objective, not tracking accuracy. Objectives and supervision counts changed between versions, so losses must not be compared directly across all runs.

| Run / history | Last epoch | Final train objective | Final val objective | Minimum-val epoch | Minimum val objective | Logged train+val hours |
| --- | --- | --- | --- | --- | --- | --- |
| [mot17_attention](<outputs/mot17_attention/history.csv>) | 20 | 0.1774 | 0.1453 | 17 | 0.1423 | 6.04 |
| [mot17_rtdetrv2_s_video_split](<outputs/mot17_rtdetrv2_s_video_split/history.csv>) | 20 | 0.5741 | 0.8337 | 7 | 0.7783 | 3.10 |
| [mot17_rtdetrv2_s_video_split_full](<outputs/mot17_rtdetrv2_s_video_split_full/history.csv>) | 20 | 0.7518 | 0.8189 | 13 | 0.7801 | 5.86 |
| [mot17_SDP_video_split](<outputs/mot17_SDP_video_split/history.csv>) | 20 | 0.3003 | 0.5372 | 5 | 0.4893 | 6.13 |
| [mot17_paper_tracktor](<outputs/mot17_paper_tracktor/history.csv>) | 20 | 109.1144 | 146.2452 | 4 | 128.1858 | 6.09 |
| [mot17_paper_tracktor_frozen](<outputs/mot17_paper_tracktor_frozen/history.csv>) | 20 | 78.0810 | 153.4716 | 15 | 151.2790 | 3.54 |
| [mot17_fixed_matching](<outputs/mot17_fixed_matching/history.csv>) | 20 | 135.1957 | 113.6245 | 15 | 110.8736 | 3.66 |
| [mot17_tuned_full_v1](<outputs/mot17_tuned_full_v1/history.csv>) | 20 | 18.4614 | 23.8569 | 20 | 23.8569 | 3.71 |
| [mot17_appearance_residual_v1](<outputs/mot17_appearance_residual_v1/history.csv>) | 30 | 8.1997 | 10.6987 | 30 | 10.6987 | 5.60 |

The residual run has epochs 1–30 and an epoch-20 backup under `resume_backups/20261007_104128_866092/`. This supports that continuation was performed. Logged hours sum recorded epoch phases; setup, tracking/evaluation and time between sessions are excluded.

## 13. Tests and verification carried out

The items below are **historical checks described in implementation notes**, supplemented by saved run artifacts where indicated. There is no pytest-style test suite in this project; root `test.py` is an appearance diagnostic. Historical “passed” descriptions do not mean these checks were rerun for this report.

| Area | Recorded checks / observations | Evidence |
| --- | --- | --- |
| Environment | CUDA tensor check on RTX 4060; PyTorch 2.6.0+cu124 detected the GPU. | [01-setup-and-audit.md](<Documentation/Implementation/01-setup-and-audit.md>) |
| Data and GT matching | Three real MOT17 sequences and temporary edge cases; coordinate/shape checks. MOT17-02-FRCNN frame 1: 13 detections, 9 matched pedestrian labels. | [03-data.md](<Documentation/Implementation/03-data.md>) |
| Initial appearance features | CPU/GPU extraction on 13 detections; ordering and batching, frozen gradients, empty/invalid crops. | [04-features.md](<Documentation/Implementation/04-features.md>) |
| Graph/model | Geometry and edge mapping; empty/masked and synthetic dense graphs; synthetic learning, save/reload, real MOT17 GPU forward/backward. | [05-graph-and-model.md](<Documentation/Implementation/05-graph-and-model.md>) |
| Tracker lifecycle | Assignment/aging checks, short CPU tracking clip and MOT export. | [06-assignment-and-tracker.md](<Documentation/Implementation/06-assignment-and-tracker.md>) |
| Pipeline integration | 20-frame CPU training pipeline; checkpoint/reload; short tracking, video and native TrackEval output. | [07-training.md](<Documentation/Implementation/07-training.md>); [08-evaluation.md](<Documentation/Implementation/08-evaluation.md>) |
| Image batching | Batched/single features and gradients, no future-image influence in eval mode, chronological state agreement, partial batches/video boundaries, validation weights unchanged. | [18-batched-training.md](<Documentation/Implementation/18-batched-training.md>) |
| Matching repair | Empty, isolated, masked, rectangular and extreme inputs; fixed totals and bounded assignment probabilities; arithmetic reference, finite gradients, learned unmatched preference, known Hungarian result; contrastive rescaling invariance. Short GPU pipeline maximum marginal error about 4e-6. | [19-matching-and-loss-fix.md](<Documentation/Implementation/19-matching-and-loss-fix.md>) |
| Residual attention | Signed self features without neighbors; one-track/two-detection distinct embeddings and finite gradients; forbidden-edge masking, save/reload, old-version rejection; short CPU training updated residual layers while encoder remained frozen. | [22-appearance-residual.md](<Documentation/Implementation/22-appearance-residual.md>) |
| Full appearance diagnostic | Two saved full-validation diagnostics with per-video plots and before/after ranking tables; actual measured results are in section 7. | [report.md](<outputs/mot17_appearance_residual_v1/feature_diagnostics/20261006_230501_671762/report.md>) |
| DDQ installation/inference | Compiled MMCV with existing CUDA/PyTorch stack; four-scale aspect and square-960 GPU checks, five-scale real-frame check; two-frame cache/evaluator integration; color conversion, clipping, empty output, coordinates and cache rejection. Three smoke-test result folders survive. | [27-ddq.md](<Documentation/Implementation/27-ddq.md>); [smoke outputs](<outputs/ddq_smoke>) |
| Full detector coverage | Eight saved full-validation detector reports, 54 completed detector search trials and three standalone fine-tuning histories. These establish execution beyond the older TODO notes. | [detector reports](<outputs/detector_evaluation>); [report.md](<outputs/detector_search/detector_search_lr_0.4/report.md>) |

## 14. Test-video inference and unverified work

Saved tracking summaries exist for unlabelled MOT17 test videos. These show inference/export activity, not scored accuracy. The residual run records:

| Test sequence / summary | Frames | Exported MOT rows | IDs created | Recorded frame-loop FPS | Video flag |
| --- | --- | --- | --- | --- | --- |
| [MOT17-01-SDP](<outputs/mot17_appearance_residual_v1/tracking/paper_motion_history/test/MOT17-01-SDP/summary.json>) | 450 | 3589 | 60 | 8.71 | True |
| [MOT17-03-SDP](<outputs/mot17_appearance_residual_v1/tracking/paper_motion_history/test/MOT17-03-SDP/summary.json>) | 1500 | 82046 | 277 | 8.19 | True |
| [MOT17-06-SDP](<outputs/mot17_appearance_residual_v1/tracking/paper_motion_history/test/MOT17-06-SDP/summary.json>) | 1194 | 7593 | 376 | 12.26 | True |
| [MOT17-07-SDP](<outputs/mot17_appearance_residual_v1/tracking/paper_motion_history/test/MOT17-07-SDP/summary.json>) | 500 | 8382 | 160 | 8.68 | True |

`save_video=True` records the run setting; this review did not visually inspect video playback.

Evidence gaps and remaining work:

- **Epoch-30 residual tracking:** training history exists through epoch 30, but the existing tracking report is not pinned to that checkpoint. A fresh, versioned evaluation is needed to claim an epoch-30 score.
- **DDQ fine-tuning:** code is implemented, but no DDQ training history, validation epochs or trained export were found. No DDQ fine-tuning benchmark can be reported. Its helper references `Documentation/Implementation/28-ddq-training.md`, which is absent.
- **Motion/history ablation:** these features are implemented and used together, but no saved matched baseline / motion-only / history-only comparison was found. Their individual accuracy gains are unmeasured here.
- **Detector-to-tracker transfer:** detector improvements are benchmarked separately; no saved tracking evaluation establishes HOTA/IDF1 gains from the RT-DETR search winner or DDQ detections.
- **External tracker baselines:** no completed ByteTrack or BoT-SORT evaluation was found.
- **MOT20 and official test scoring:** no scored MOT20 experiment or independent official MOT test result was found.
- **Repeatability:** no multi-seed aggregate, confidence interval or repeated-run variance estimate was found.
- **Documentation drift:** TODO items saying residual training, detector search, detector fine-tuning and full DDQ validation are still pending are superseded by saved artifacts. Other historical prose mentions obsolete temporal splits, trainable/frozen appearance defaults, and an epoch-40 target. Use each run's own manifests and reports to reconstruct its recipe.

## Appendix A. Every completed association/tracking search trial

Rows below are copied from each saved `results.csv`, retaining recorded rank. HOTA/IDF1/MOTA are percentages, IDSW is a count, and time is total trial seconds. Blank epoch/prefix fields in tracking mode mean an existing checkpoint was reused.

### association_search2_20261003_161703_821353

Source: [results.csv](<outputs/searches/association_search2_20261003_161703_821353/results.csv>).

| rank | trial | mode | epochs | checkpoint_epoch | MATCH_THRESHOLD | TRACK_CONFIRM_MIN_HITS | HOTA | IDF1 | MOTA | IDSW | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | train | 5 | 4 | 0.05 | 1 | 21.654 | 20.24 | 34.226 | 4069.0 | 3466.36 |
| 2 | 2 | train | 5 | 1 | 0.025 | 1 | 21.471 | 19.379 | 31.419 | 4742.0 | 3508.8 |
| 3 | 1 | train | 5 | 1 | 0.01 | 1 | 21.405 | 19.677 | 31.657 | 4683.0 | 3460.67 |
| 4 | 5 | train | 5 | 1 | 0.09 | 1 | 18.789 | 14.547 | 18.305 | 7875.0 | 3471.54 |
| 5 | 4 | train | 5 | 5 | 0.75 | 1 | 5.0795 | 1.5391 | 1.0123 | 12011.0 | 3461.16 |

### association_search3_20261003_225614_077222

Source: [results.csv](<outputs/searches/association_search3_20261003_225614_077222/results.csv>).

| rank | trial | mode | epochs | train_max_frames_per_sequence | checkpoint_epoch | MATCH_THRESHOLD | TRACK_CONFIRM_MIN_HITS | ASSOCIATION_DISTANCE_PX | HOTA | IDF1 | MOTA | IDSW | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | train | 3 | 400 | 3 | 0.05 | 1 | 100 | 27.353 | 27.588 | 37.614 | 3259.0 | 1307.03 |
| 2 | 16 | train | 3 | 400 | 2 | 0.05 | 3 | 100 | 27.024 | 27.314 | 38.099 | 3143.0 | 1306.54 |
| 3 | 13 | train | 3 | 400 | 1 | 0.05 | 2 | 100 | 26.846 | 28.054 | 38.547 | 3031.0 | 1299.88 |
| 4 | 7 | train | 3 | 400 | 3 | 0.025 | 3 | 100 | 26.501 | 26.409 | 36.987 | 3407.0 | 1304.76 |
| 5 | 1 | train | 3 | 400 | 3 | 0.025 | 1 | 100 | 26.082 | 26.126 | 36.347 | 3562.0 | 1302.23 |
| 6 | 4 | train | 3 | 400 | 3 | 0.025 | 2 | 100 | 26.025 | 25.077 | 37.15 | 3366.0 | 1301.5 |
| 7 | 2 | train | 3 | 400 | 3 | 0.025 | 1 | 200 | 21.297 | 19.578 | 31.31 | 4768.0 | 1301.41 |
| 8 | 11 | train | 3 | 400 | 2 | 0.05 | 1 | 200 | 21.256 | 19.484 | 31.352 | 4754.0 | 1307.59 |
| 9 | 5 | train | 3 | 400 | 2 | 0.025 | 2 | 200 | 20.844 | 19.272 | 30.03 | 5063.0 | 1306.45 |
| 10 | 8 | train | 3 | 400 | 3 | 0.025 | 3 | 200 | 19.152 | 16.752 | 26.437 | 5896.0 | 1307.11 |
| 11 | 18 | train | 3 | 400 | 1 | 0.05 | 3 | 300 | 18.742 | 16.307 | 27.7 | 5627.0 | 1304.36 |
| 12 | 3 | train | 3 | 400 | 2 | 0.025 | 1 | 300 | 18.199 | 16.092 | 27.202 | 5742.0 | 1310.48 |
| 13 | 12 | train | 3 | 400 | 3 | 0.05 | 1 | 300 | 16.642 | 12.964 | 16.067 | 8412.0 | 1308.47 |
| 14 | 14 | train | 3 | 400 | 1 | 0.05 | 2 | 200 | 16.159 | 14.669 | 16.653 | 3227.0 | 1301.63 |
| 15 | 17 | train | 3 | 400 | 3 | 0.05 | 3 | 200 | 15.079 | 13.274 | 17.046 | 2936.0 | 1308.73 |
| 16 | 6 | train | 3 | 400 | 3 | 0.025 | 2 | 300 | 15.069 | 12.475 | 17.439 | 6009.0 | 1304.11 |
| 17 | 9 | train | 3 | 400 | 3 | 0.025 | 3 | 300 | 14.497 | 11.949 | 16.665 | 6089.0 | 1311.14 |
| 18 | 22 | train | 3 | 400 | 3 | 0.75 | 2 | 100 | 12.762 | 9.0032 | 5.1953 | 9.0 | 1307.77 |
| 19 | 19 | train | 3 | 400 | 3 | 0.75 | 1 | 100 | 12.492 | 6.0461 | 4.7352 | 11121.0 | 1309.1 |
| 20 | 25 | train | 3 | 400 | 2 | 0.75 | 3 | 100 | 11.964 | 8.2239 | 4.6055 | 8.0 | 1309.3 |
| 21 | 15 | train | 3 | 400 | 3 | 0.05 | 2 | 300 | 11.452 | 9.3059 | 8.9559 | 2199.0 | 1306.71 |
| 22 | 20 | train | 3 | 400 | 3 | 0.75 | 1 | 200 | 4.9563 | 1.495 | 0.97883 | 12019.0 | 1311.46 |
| 23 | 21 | train | 3 | 400 | 2 | 0.75 | 1 | 300 | 4.7452 | 1.2688 | 0.75295 | 12073.0 | 1312.69 |
| 24 | 26 | train | 3 | 400 | 3 | 0.75 | 3 | 200 | 3.1483 | 1.8125 | 0.99975 | 2.0 | 1308.15 |
| 25 | 23 | train | 3 | 400 | 3 | 0.75 | 2 | 200 | 3.0853 | 1.7714 | 0.98302 | 3.0 | 1305.08 |
| 26 | 24 | train | 3 | 400 | 1 | 0.75 | 2 | 300 | 2.4528 | 1.4199 | 0.67765 | 0.0 | 1309.93 |
| 27 | 27 | train | 3 | 400 | 2 | 0.75 | 3 | 300 | 2.4255 | 1.4117 | 0.67347 | 0.0 | 1309.38 |

### association_search4_20261004_101321_734762

Source: [results.csv](<outputs/searches/association_search4_20261004_101321_734762/results.csv>).

| rank | trial | mode | epochs | train_max_frames_per_sequence | checkpoint_epoch | MATCH_THRESHOLD | TRACK_CONFIRM_MIN_HITS | ASSOCIATION_DISTANCE_PX | HOTA | IDF1 | MOTA | IDSW | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | train | 3 | 400 | 3 | 0.05 | 1 | 25 | 36.989 | 40.165 | 48.059 | 764.0 | 1309.61 |
| 2 | 2 | train | 3 | 400 | 3 | 0.05 | 1 | 50 | 33.479 | 35.239 | 44.65 | 1577.0 | 1298.56 |
| 3 | 3 | train | 3 | 400 | 3 | 0.05 | 1 | 75 | 29.443 | 30.026 | 39.969 | 2698.0 | 1308.73 |
| 4 | 4 | train | 3 | 400 | 3 | 0.05 | 1 | 100 | 27.136 | 27.571 | 39.007 | 2926.0 | 1332.75 |

### association_search5_20261004_114439_372694

Source: [results.csv](<outputs/searches/association_search5_20261004_114439_372694/results.csv>).

| rank | trial | mode | epochs | train_max_frames_per_sequence | checkpoint_epoch | MATCH_THRESHOLD | TRACK_CONFIRM_MIN_HITS | ASSOCIATION_DISTANCE_PX | HOTA | IDF1 | MOTA | IDSW | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | train | 3 | 400 | 1 | 0.05 | 1 | 25 | 37.582 | 42.367 | 48.917 | 559.0 | 1304.78 |
| 2 | 3 | train | 3 | 400 | 3 | 0.05 | 1 | 20 | 37.3 | 41.782 | 48.887 | 566.0 | 1357.49 |
| 3 | 2 | train | 3 | 400 | 3 | 0.05 | 1 | 10 | 34.833 | 39.04 | 47.277 | 949.0 | 1305.79 |
| 4 | 1 | train | 3 | 400 | 1 | 0.05 | 1 | 5 | 30.031 | 32.922 | 42.132 | 2179.0 | 1287.46 |

### association_search_20261003_120918_760750

Source: [results.csv](<outputs/searches/association_search_20261003_120918_760750/results.csv>).

| rank | trial | mode | epochs | checkpoint_epoch | MATCH_THRESHOLD | TRACK_CONFIRM_MIN_HITS | HOTA | IDF1 | MOTA | IDSW | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | train | 5 | 5 | 0.05 | 1 | 23.031 | 22.458 | 35.535 | 3752.0 | 3464.13 |
| 2 | 2 | train | 5 | 5 | 0.1 | 1 | 20.173 | 16.257 | 19.719 | 7539.0 | 3540.8 |
| 3 | 3 | train | 5 | 1 | 0.2 | 1 | 12.407 | 6.035 | 6.8895 | 10606.0 | 3514.84 |
| 4 | 4 | train | 5 | 5 | 0.35 | 1 | 10.116 | 4.7718 | 3.748 | 11357.0 | 3543.28 |

### tracking_match_thresholds2_20261007_101827_969084

Source: [results.csv](<outputs/searches/tracking_match_thresholds2_20261007_101827_969084/results.csv>).

| rank | trial | mode | epochs | train_max_frames_per_sequence | checkpoint_epoch | MATCH_THRESHOLD | HOTA | IDF1 | MOTA | IDSW | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | tracking | — | — | 20 | 0.05 | 41.794 | 47.971 | 49.87 | 331.0 | 131.54 |
| 2 | 2 | tracking | — | — | 20 | 0.3 | 39.668 | 44.104 | 44.144 | 1700.0 | 128.65 |
| 3 | 3 | tracking | — | — | 20 | 0.5 | 36.912 | 39.719 | 39.291 | 2860.0 | 129.65 |
| 4 | 4 | tracking | — | — | 20 | 0.7 | 26.715 | 24.775 | 23.806 | 6562.0 | 129.83 |

### tracking_match_thresholds_20261007_100244_226092

Source: [results.csv](<outputs/searches/tracking_match_thresholds_20261007_100244_226092/results.csv>).

| rank | trial | mode | epochs | train_max_frames_per_sequence | checkpoint_epoch | MATCH_THRESHOLD | HOTA | IDF1 | MOTA | IDSW | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | tracking | — | — | 20 | 0.02 | 41.794 | 47.971 | 49.87 | 331.0 | 136.55 |
| 2 | 2 | tracking | — | — | 20 | 0.05 | 41.794 | 47.971 | 49.87 | 331.0 | 131.19 |
| 3 | 3 | tracking | — | — | 20 | 0.1 | 41.794 | 47.971 | 49.87 | 331.0 | 132.16 |
| 4 | 4 | tracking | — | — | 20 | 0.2 | 41.794 | 47.971 | 49.87 | 331.0 | 130.26 |

## Appendix B. Detector search: best report point for every completed trial

For each trial, select the highest-F1 row meeting the search's precision ≥70% and recall ≥50% floors; if none qualifies, show the highest-F1 fallback and mark it ineligible. Epoch zero can appear when the pretrained baseline is the best point. This table is derived from the full saved ranking; it does not imply every reported epoch's weights were retained. “Retained” follows the saved ranking's non-empty checkpoint field.

| Trial / history | Size | LR | Weight decay | Accum. | Epoch | Threshold | Precision % | Recall % | F1 % | Eligible | Retained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trial_001](<outputs/detector_search/detector_search_lr_0.4/trial_001/history.csv>) | 640 | 1e-05 | 0.001 | 2 | 3 | 0.25 | 77.88 | 48.91 | 60.09 | False | yes |
| [trial_002](<outputs/detector_search/detector_search_lr_0.4/trial_002/history.csv>) | 640 | 1e-05 | 0.001 | 4 | 3 | 0.3 | 84.30 | 46.18 | 59.67 | False | yes |
| [trial_003](<outputs/detector_search/detector_search_lr_0.4/trial_003/history.csv>) | 640 | 1e-05 | 0.001 | 8 | 3 | 0.3 | 80.17 | 47.16 | 59.39 | False | yes |
| [trial_004](<outputs/detector_search/detector_search_lr_0.4/trial_004/history.csv>) | 640 | 1e-05 | 0.0001 | 2 | 3 | 0.25 | 77.82 | 48.77 | 59.96 | False | yes |
| [trial_005](<outputs/detector_search/detector_search_lr_0.4/trial_005/history.csv>) | 640 | 1e-05 | 0.0001 | 4 | 3 | 0.3 | 84.14 | 46.20 | 59.65 | False | yes |
| [trial_006](<outputs/detector_search/detector_search_lr_0.4/trial_006/history.csv>) | 640 | 1e-05 | 0.0001 | 8 | 3 | 0.3 | 80.00 | 47.20 | 59.37 | False | yes |
| [trial_007](<outputs/detector_search/detector_search_lr_0.4/trial_007/history.csv>) | 640 | 1e-05 | 1e-05 | 2 | 3 | 0.25 | 77.94 | 48.77 | 59.99 | False | yes |
| [trial_008](<outputs/detector_search/detector_search_lr_0.4/trial_008/history.csv>) | 640 | 1e-05 | 1e-05 | 4 | 3 | 0.3 | 84.48 | 46.13 | 59.67 | False | yes |
| [trial_009](<outputs/detector_search/detector_search_lr_0.4/trial_009/history.csv>) | 640 | 1e-05 | 1e-05 | 8 | 3 | 0.3 | 80.03 | 47.26 | 59.43 | False | yes |
| [trial_010](<outputs/detector_search/detector_search_lr_0.4/trial_010/history.csv>) | 960 | 1e-05 | 0.001 | 2 | 3 | 0.3 | 78.98 | 52.07 | 62.76 | True | yes |
| [trial_011](<outputs/detector_search/detector_search_lr_0.4/trial_011/history.csv>) | 960 | 1e-05 | 0.001 | 4 | 3 | 0.3 | 77.02 | 52.59 | 62.50 | True | yes |
| [trial_012](<outputs/detector_search/detector_search_lr_0.4/trial_012/history.csv>) | 960 | 1e-05 | 0.001 | 8 | 3 | 0.35 | 81.71 | 50.06 | 62.08 | True | yes |
| [trial_013](<outputs/detector_search/detector_search_lr_0.4/trial_013/history.csv>) | 960 | 1e-05 | 0.0001 | 2 | 3 | 0.3 | 79.06 | 52.13 | 62.83 | True | yes |
| [trial_014](<outputs/detector_search/detector_search_lr_0.4/trial_014/history.csv>) | 960 | 1e-05 | 0.0001 | 4 | 3 | 0.3 | 77.29 | 52.38 | 62.44 | True | yes |
| [trial_015](<outputs/detector_search/detector_search_lr_0.4/trial_015/history.csv>) | 960 | 1e-05 | 0.0001 | 8 | 3 | 0.35 | 81.89 | 50.10 | 62.17 | True | yes |
| [trial_016](<outputs/detector_search/detector_search_lr_0.4/trial_016/history.csv>) | 960 | 1e-05 | 1e-05 | 2 | 3 | 0.3 | 78.80 | 52.14 | 62.75 | True | yes |
| [trial_017](<outputs/detector_search/detector_search_lr_0.4/trial_017/history.csv>) | 960 | 1e-05 | 1e-05 | 4 | 3 | 0.3 | 77.06 | 52.51 | 62.46 | True | yes |
| [trial_018](<outputs/detector_search/detector_search_lr_0.4/trial_018/history.csv>) | 960 | 1e-05 | 1e-05 | 8 | 3 | 0.3 | 73.77 | 53.27 | 61.87 | True | yes |
| [trial_019](<outputs/detector_search/detector_search_lr_0.4/trial_019/history.csv>) | 640 | 2e-05 | 0.001 | 2 | 3 | 0.2 | 72.78 | 51.33 | 60.20 | True | yes |
| [trial_020](<outputs/detector_search/detector_search_lr_0.4/trial_020/history.csv>) | 640 | 2e-05 | 0.001 | 4 | 3 | 0.25 | 79.29 | 48.86 | 60.46 | False | yes |
| [trial_021](<outputs/detector_search/detector_search_lr_0.4/trial_021/history.csv>) | 640 | 2e-05 | 0.001 | 8 | 2 | 0.25 | 71.04 | 50.15 | 58.80 | True | yes |
| [trial_022](<outputs/detector_search/detector_search_lr_0.4/trial_022/history.csv>) | 640 | 2e-05 | 0.0001 | 2 | 3 | 0.2 | 71.24 | 51.76 | 59.96 | True | yes |
| [trial_023](<outputs/detector_search/detector_search_lr_0.4/trial_023/history.csv>) | 640 | 2e-05 | 0.0001 | 4 | 3 | 0.25 | 80.04 | 48.34 | 60.28 | False | yes |
| [trial_024](<outputs/detector_search/detector_search_lr_0.4/trial_024/history.csv>) | 640 | 2e-05 | 0.0001 | 8 | 2 | 0.25 | 70.66 | 50.20 | 58.70 | True | yes |
| [trial_025](<outputs/detector_search/detector_search_lr_0.4/trial_025/history.csv>) | 640 | 2e-05 | 1e-05 | 2 | 3 | 0.2 | 71.27 | 51.51 | 59.80 | True | yes |
| [trial_026](<outputs/detector_search/detector_search_lr_0.4/trial_026/history.csv>) | 640 | 2e-05 | 1e-05 | 4 | 3 | 0.25 | 81.07 | 47.95 | 60.26 | False | yes |
| [trial_027](<outputs/detector_search/detector_search_lr_0.4/trial_027/history.csv>) | 640 | 2e-05 | 1e-05 | 8 | 2 | 0.25 | 70.62 | 50.24 | 58.71 | True | yes |
| [trial_028](<outputs/detector_search/detector_search_lr_0.4/trial_028/history.csv>) | 960 | 2e-05 | 0.001 | 2 | 3 | 0.3 | 82.89 | 51.36 | 63.42 | True | yes |
| [trial_029](<outputs/detector_search/detector_search_lr_0.4/trial_029/history.csv>) | 960 | 2e-05 | 0.001 | 4 | 3 | 0.3 | 80.87 | 51.93 | 63.25 | True | yes |
| [trial_030](<outputs/detector_search/detector_search_lr_0.4/trial_030/history.csv>) | 960 | 2e-05 | 0.001 | 8 | 3 | 0.3 | 79.17 | 51.77 | 62.61 | True | yes |
| [trial_031](<outputs/detector_search/detector_search_lr_0.4/trial_031/history.csv>) | 960 | 2e-05 | 0.0001 | 2 | 3 | 0.3 | 83.82 | 51.34 | 63.68 | True | yes |
| [trial_032](<outputs/detector_search/detector_search_lr_0.4/trial_032/history.csv>) | 960 | 2e-05 | 0.0001 | 4 | 3 | 0.3 | 80.91 | 52.01 | 63.32 | True | yes |
| [trial_033](<outputs/detector_search/detector_search_lr_0.4/trial_033/history.csv>) | 960 | 2e-05 | 0.0001 | 8 | 3 | 0.3 | 79.63 | 51.56 | 62.59 | True | yes |
| [trial_034](<outputs/detector_search/detector_search_lr_0.4/trial_034/history.csv>) | 960 | 2e-05 | 1e-05 | 2 | 3 | 0.3 | 82.63 | 51.93 | 63.78 | True | yes |
| [trial_035](<outputs/detector_search/detector_search_lr_0.4/trial_035/history.csv>) | 960 | 2e-05 | 1e-05 | 4 | 3 | 0.3 | 80.82 | 51.73 | 63.08 | True | yes |
| [trial_036](<outputs/detector_search/detector_search_lr_0.4/trial_036/history.csv>) | 960 | 2e-05 | 1e-05 | 8 | 3 | 0.3 | 79.24 | 51.92 | 62.73 | True | yes |
| [trial_037](<outputs/detector_search/detector_search_lr_0.4/trial_037/history.csv>) | 640 | 5e-05 | 0.001 | 2 | 3 | 0.2 | 73.00 | 51.87 | 60.65 | True | yes |
| [trial_038](<outputs/detector_search/detector_search_lr_0.4/trial_038/history.csv>) | 640 | 5e-05 | 0.001 | 4 | 3 | 0.2 | 72.75 | 51.07 | 60.01 | True | yes |
| [trial_039](<outputs/detector_search/detector_search_lr_0.4/trial_039/history.csv>) | 640 | 5e-05 | 0.001 | 8 | 3 | 0.25 | 80.57 | 48.62 | 60.64 | False | yes |
| [trial_040](<outputs/detector_search/detector_search_lr_0.4/trial_040/history.csv>) | 640 | 5e-05 | 0.0001 | 2 | 3 | 0.2 | 71.05 | 52.67 | 60.50 | True | yes |
| [trial_041](<outputs/detector_search/detector_search_lr_0.4/trial_041/history.csv>) | 640 | 5e-05 | 0.0001 | 4 | 3 | 0.2 | 73.39 | 51.25 | 60.35 | True | yes |
| [trial_042](<outputs/detector_search/detector_search_lr_0.4/trial_042/history.csv>) | 640 | 5e-05 | 0.0001 | 8 | 3 | 0.25 | 80.67 | 48.93 | 60.91 | False | yes |
| [trial_043](<outputs/detector_search/detector_search_lr_0.4/trial_043/history.csv>) | 640 | 5e-05 | 1e-05 | 2 | 2 | 0.25 | 76.84 | 50.40 | 60.87 | True | yes |
| [trial_044](<outputs/detector_search/detector_search_lr_0.4/trial_044/history.csv>) | 640 | 5e-05 | 1e-05 | 4 | 2 | 0.25 | 78.04 | 49.54 | 60.61 | False | yes |
| [trial_045](<outputs/detector_search/detector_search_lr_0.4/trial_045/history.csv>) | 640 | 5e-05 | 1e-05 | 8 | 3 | 0.2 | 72.82 | 51.50 | 60.33 | True | yes |
| [trial_046](<outputs/detector_search/detector_search_lr_0.4/trial_046/history.csv>) | 960 | 5e-05 | 0.001 | 2 | 3 | 0.25 | 79.76 | 53.88 | 64.32 | True | yes |
| [trial_047](<outputs/detector_search/detector_search_lr_0.4/trial_047/history.csv>) | 960 | 5e-05 | 0.001 | 4 | 3 | 0.3 | 83.46 | 52.82 | 64.70 | True | yes |
| [trial_048](<outputs/detector_search/detector_search_lr_0.4/trial_048/history.csv>) | 960 | 5e-05 | 0.001 | 8 | 3 | 0.3 | 83.85 | 51.50 | 63.81 | True | yes |
| [trial_049](<outputs/detector_search/detector_search_lr_0.4/trial_049/history.csv>) | 960 | 5e-05 | 0.0001 | 2 | 3 | 0.25 | 81.54 | 53.64 | 64.71 | True | yes |
| [trial_050](<outputs/detector_search/detector_search_lr_0.4/trial_050/history.csv>) | 960 | 5e-05 | 0.0001 | 4 | 2 | 0.3 | 84.07 | 51.88 | 64.16 | True | yes |
| [trial_051](<outputs/detector_search/detector_search_lr_0.4/trial_051/history.csv>) | 960 | 5e-05 | 0.0001 | 8 | 3 | 0.3 | 84.39 | 52.00 | 64.34 | True | yes |
| [trial_052](<outputs/detector_search/detector_search_lr_0.4/trial_052/history.csv>) | 960 | 5e-05 | 1e-05 | 2 | 2 | 0.3 | 85.20 | 52.46 | 64.93 | True | yes |
| [trial_053](<outputs/detector_search/detector_search_lr_0.4/trial_053/history.csv>) | 960 | 5e-05 | 1e-05 | 4 | 3 | 0.3 | 85.85 | 51.76 | 64.58 | True | yes |
| [trial_054](<outputs/detector_search/detector_search_lr_0.4/trial_054/history.csv>) | 960 | 5e-05 | 1e-05 | 8 | 3 | 0.3 | 83.43 | 51.82 | 63.93 | True | yes |

## Appendix C. How to trace the numbers

- Tracking scores: `outputs/<run>/evaluation/.../results.md`; exact per-video and combined values in `MOT17/DragonTrack/pedestrian_detailed.csv` and `pedestrian_summary.txt`.
- Training duration/objective/epochs: `outputs/<run>/history.csv`; detector training recipes in `settings.json`.
- Detector metrics, size/visibility recall, timing and provenance: `outputs/detector_evaluation/<timestamp_backend>/summary.json`, `summary.csv` and `report.md`.
- Association search grids: `outputs/searches/<search>/search.json`; ranked results in `results.csv`; selected checkpoint in `best.json`.
- Detector grid and completion: `outputs/detector_search/detector_search_lr_0.4/search.json`, `ranking.csv`, `best.json`, and each trial's `search_complete.json`.
- Diagnostic checkpoint hashes, epochs and full frame coverage: `outputs/<run>/feature_diagnostics/<timestamp>/summary.json`.
- Historical functionality/performance checks: `Documentation/Implementation/`; these notes can predate later full runs.

All numerical conclusions above are limited to artifacts present in this workspace at review time.
