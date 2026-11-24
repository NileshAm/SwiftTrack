# Tracker-only validation control

Run from the project root in the `dragon-track` environment:

```bash
python evaluate-tracker-gt-boxes.py
python evaluate-tracker-gt-boxes.py --similarity 0.5
```

The script extracts TrackEval's preprocessed validation pedestrian boxes for MOT17-02-SDP and MOT17-09-SDP. It writes those boxes as detection input with confidence 1, without passing their identity labels to the tracker. The tracker uses the same frozen OSNet features, cosine assignment, motion, appearance history, and confirmation as `track-tracktor-osnet.py`. Each run writes tracking outputs and `report.md`/`report.json` under `outputs/tracker_gt_boxes/<timestamp>/`.

| Cosine threshold | HOTA | DetA | AssA | IDF1 | MOTA | ID switches | Consistent identity on matched boxes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 63.538 | 96.902 | 41.665 | 56.877 | 92.537 | 1,782 | 55.15% |
| 0.5 | 58.956 | 95.425 | 36.433 | 55.768 | 93.667 | 1,428 | 54.22% |

The 0.7 setting has better association accuracy despite more CLEAR identity switches. Lowering the threshold makes more matches, but more of them are wrong; MOTA alone hides that trade-off. With near-perfect box coverage, association remains the major limit. This is an analysis control using validation labels and cannot be reported as a normal detector-plus-tracker result. The same validation sequences have been used repeatedly for tuning, so these numbers are exploratory.

The separate [fixed-output-box diagnostic](35-tracking-diagnostics.md) measures identity consistency on actual Tracktor and YOLOX tracker outputs. Its denominator excludes boxes that never reached the tracker output, whereas this control tests the tracker after supplying all valid person boxes.
