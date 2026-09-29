# Utils/data.py — read MOT data and prepare labels

[Back to TODO](README.md) · [Shared arrays](02-contracts.md)

| Function | What to do |
| --- | --- |
| `load_sequence(sequence_path, detection_path=None, frame_range=None, *, box_origin=1, min_score=None)` | Return a dictionary keyed by original frame ID with image paths, dimensions, FPS, boxes, scores, and detection-file line numbers (`row_ids`). Relative detection paths are resolved within the sequence folder. Include frames with no detections; keep images on disk. |
| `read_image(frame)` | Load one frame as an RGB NumPy array and check its dimensions against sequence metadata. |
| `load_ground_truth(sequence_path)` | Read valid person identities and ignored-object flags for training/evaluation. Keep these separate from normal tracking inputs. |
| `match_detections_to_gt(boxes, gt, iou_threshold=0.5)` | Match detections to valid GT boxes one-to-one. Return `ids`, `valid_mask`, and `ignored_mask` in detection order. Unmatched IDs are `-1` and cannot be used as supervised person identities. |
| `pairwise_iou(boxes_a, boxes_b)` | Shared overlap helper implemented here so the loader works now; `Utils/graph.py` can import it later instead of duplicating it. |

Convert `(x,y,w,h)` to `(x,y,x+w,y+h)`, applying the stated coordinate-origin conversion once. Keep real scores. Find image files from the dataset rather than assuming four-digit names. Report malformed rows or missing images clearly. Make any clipping/filtering explicit and apply it consistently to boxes, scores, features, and labels.

Choose separate `TRAIN_SEQUENCES` and `VAL_SEQUENCES` in `Utils/config.py`. Both lists refer to folders under `DATASET_ROOT / DATASET_SPLIT`, and every frame of each selected video is used. `Utils/splits.py` supplies `full_frame_range()` and rejects training/validation overlap, including different detector versions of the same video. Percentage splitting has been removed. See the [whole-video guide](11-paper-components.md#whole-video-selection--utilssplitspy).

Training labels use the same valid identity as positive and different valid identities as negative; ignored/unmatched labels are masked. Evaluation copies raw GT so TrackEval can apply its own preprocessing.

**Quick check:** print the first frame ID, image dimensions, boxes, and scores. Draw a few boxes to confirm coordinates. Inspect a frame with no detections and a few matched training identities. Repeat on one MOT20 sample. No separate test file or manifest generator.

## Try the implemented loader

NumPy, SciPy, and Pillow are installed in the `dragon-track` environment. Activate it and run the following Python code from the project root (for example, in `main.py`):

```python
from pathlib import Path
from Utils.data import load_sequence, read_image, load_ground_truth, match_detections_to_gt

sequence = Path("data/MOT17/train/MOT17-02-FRCNN")
frames = load_sequence(sequence, frame_range=(1, 3))
frame = frames[1]  # Original frame ID, not a zero-based list index.
image = read_image(frame)
print("Image shape:", image.shape)
print("Boxes:", frame["boxes"])
print("Scores:", frame["scores"])

# These extra lines are for training supervision only.
gt = load_ground_truth(sequence)
labels = match_detections_to_gt(frame["boxes"], gt[1])
print("Person IDs:", labels["ids"])
print("Usable labels:", labels["valid_mask"])
```

For MOT17-02-FRCNN frame 1, the checked local data gives an image shape of `(1080, 1920, 3)`, 13 detections, and 9 matched person labels. Unmatched labels remain `-1`. Frame dictionaries never include ground truth unless you explicitly load it separately.

Boxes are float32 zero-origin xyxy arrays. They remain unclipped here to preserve the original geometry; crop extraction should clip a copy. Scores are preserved apart from float32 representation. Filtering applies to boxes, scores, and row IDs together. No detections gives boxes shaped `(0, 4)` and scores shaped `(0,)`.

The training-label policy accepts marked class-1 pedestrians; all other annotations are conservatively ignored. A valid person match takes priority; an unmatched detection overlapping ignored GT is marked ignored. Matching maximizes the number of above-threshold pairs first, then total IoU. This policy does not implement evaluation preprocessing: use TrackEval for that. Class/mark meanings were checked against [TrackEval's MOT adapter](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py).

**Checked:** three local MOT17 sequences; image dimensions; box conversion and real scores; unsorted detection rows; alternate filename padding; empty frames/GT; duplicate detections; ignored GT; score filtering; malformed boxes. Checks used temporary data, with no permanent test files. MOT20 is pending because it is not present locally.

This loader does not import PyTorch. PyTorch is now installed in `dragon-track`, and CUDA was verified on 27 September 2026. You can supply paths and options from `Utils/config.py` instead of the literal example path.
