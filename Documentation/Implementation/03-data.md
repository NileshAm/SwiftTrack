# data.py — read MOT data and prepare labels

[Back to TODO](README.md) · [Shared arrays](02-contracts.md)

| Function | What to do |
| --- | --- |
| `load_sequence(sequence_path, detection_path, frame_range=None)` | Read sequence information, actual image filenames, and detection rows. Return lightweight frame records indexed by original frame ID, with boxes/scores; load images as needed rather than holding the sequence in memory. Include frames with no detections. |
| `load_ground_truth(sequence_path)` | Read valid person identities and ignored-object flags for training/evaluation. Keep these separate from normal tracking inputs. |
| `match_detections_to_gt(boxes, gt, iou_threshold)` | Match detections to valid GT boxes one-to-one. Return identities and valid/ignored masks in detection order. Use `graph.py`'s IoU helper; unmatched detections must not become invented identities. |

Convert `(x,y,w,h)` to `(x,y,x+w,y+h)`, applying the stated coordinate-origin conversion once. Keep real scores. Find image files from the dataset rather than assuming four-digit names. Report malformed rows or missing images clearly. Make any clipping/filtering explicit and apply it consistently to boxes, scores, features, and labels.

Choose training/validation/evaluation sequences in `config.py`. Keep detector variants of the same MOT17 video together. For temporal splits, never create training pairs or history across split boundaries. Training can start with consecutive frame pairs: same valid identity is positive; different valid identity is negative; ignored/unmatched labels are masked. Do not treat evaluation GT as detections unless explicitly reporting an oracle experiment.

**Quick check:** print the first frame ID, image dimensions, boxes, and scores. Draw a few boxes to confirm coordinates. Inspect a frame with no detections and a few matched training identities. Repeat on one MOT20 sample. No separate test file or manifest generator.
