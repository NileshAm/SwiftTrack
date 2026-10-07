# RT-DETRv2-S detections, then association training

> **Current settings and architecture:** see [paper alignment](17-paper-alignment.md). The notes below describe earlier development stages; frozen-encoder defaults, previous loss/scoring layers, and old run names are superseded. `main.py` now plots saved run metrics.

**Current default:** `DETECTION_BACKEND="supplied"` uses original SDP boxes and `detect.py` skips inference. This guide describes the optional RT-DETR backend; switch explicitly to `"rtdetrv2"` and use a fresh run to use it again.

`detect.py` runs the COCO-pretrained [PekingU/rtdetr_v2_r18vd checkpoint](https://huggingface.co/PekingU/rtdetr_v2_r18vd), the ResNet18 small RT-DETRv2 model. The existing Transformers 4.49.0 installation supports it; PyTorch does not need upgrading. The model revision is pinned in config. First use downloads weights into Hugging Face's normal cache; later use reuses them.

## Run in this order

```bash
conda activate dragon-track
python detect.py
python train.py
python tracking.py
python evaluate.py
```

`detect.py` processes `TRAIN_SEQUENCES`, `VAL_SEQUENCES`, and `TEST_SEQUENCES` once. The first two lists use `DATASET_ROOT / DATASET_SPLIT`; test uses `TEST_DATASET_ROOT / TEST_DATASET_SPLIT`. It reads only images and `seqinfo.ini`, with no GT or supplied detection files. Overall and current-video frame progress bars show ETA.

To process only one group, use `python detect.py --split train`, `--split val`, or `--split test`. Training needs both training and validation caches. Tracking needs caches for all videos it processes. No detector inference runs inside an epoch or in tracking. `bash run.sh` runs the whole pipeline and stops on the first failure.

## Settings in `Utils/config.py`

| Setting | Current value / purpose |
| --- | --- |
| `DETECTION_BACKEND` | `"rtdetrv2"` selects generated caches. `"supplied"` reads the old `DETECTION_FILE`. |
| `DETECTION_CACHE_DIR` | `detections/rtdetrv2_s_coco_640_score010` under the project. Separate from original data and training outputs. |
| `DETECTOR_MODEL_NAME`, `DETECTOR_REVISION` | Fixed COCO-pretrained model and exact weight revision. |
| `DETECTOR_IMAGE_SIZE` | 640×640 input, using the checkpoint's image processor. Boxes are mapped back to the original image size. |
| `DETECTOR_MIN_SCORE` | 0.10 cache floor; keeps lower-confidence boxes for association. A starting value, not a validated optimum. |
| `DETECTION_MIN_SCORE` | Optional extra filtering when reading the cache. Cannot recover detections discarded during caching. |
| `NEW_TRACK_MIN_SCORE` | Still 0.5; unmatched detections below this cannot start new tracks. Tune on validation. |
| `RUN_NAME` | Fresh `mot17_rtdetrv2_s_video_split` run. Earlier SDP results/checkpoints remain intact. |

The `-SDP` suffix now identifies the **image folder only**. With `DETECTION_BACKEND="rtdetrv2"`, the original SDP boxes are not used. Keep the same train/validation assignment for comparisons; do not add DPM/FRCNN copies as independent videos.

## Saved results and restarts

```text
detections/rtdetrv2_s_coco_640_score010/
  train/MOT17-04-SDP/det.txt, summary.json, preview.jpg
  train/MOT17-02-SDP/det.txt, summary.json, preview.jpg
  test/MOT17-01-SDP/det.txt, summary.json, preview.jpg
```

`train/` here is the original MOT directory containing both our training and validation videos. The disjoint config lists still control their use.

Each detection row is `frame,-1,left,top,width,height,confidence,-1,-1,-1`. Coordinates use MOT's one-based origin; the existing loader converts back to zero-origin `xyxy` exactly once. The person class is resolved by name, other classes are removed, boxes are clipped, and degenerate boxes discarded. Official top-query postprocessing is used without added NMS. The first-frame preview shows boxes/confidence, not tracking IDs.

The summary records full-video coverage, dimensions, detector settings, checksum, timing, and library versions. Completed matching caches are skipped. An interrupted video leaves an incomplete file; rerunning restarts that video while reusing completed videos. Changed settings or a damaged completed file raise an error: choose a fresh cache directory to regenerate.

Training/tracking reject missing or incompatible caches and never silently use supplied boxes. Checkpoints and tracking summaries record the recipe. Switching sources, thresholds, input size, or weight revision requires fresh association training with a new run name. Cache checks validate metadata and exported contents, not every source image's contents; preserve the source images.

## Functions and integration

- `detect.PersonDetector`: loads frozen RT-DETRv2-S and returns original-size person boxes/scores from `predict(image)`.
- `detect.run_sequence`: exports every frame, writes preview/completion metadata, and updates progress.
- `detect.main`: selects groups, skips complete caches, and loads the detector once for pending videos.
- `Utils/data.load_image_frames`: discovers frames without requiring detections or GT; also reused by `load_sequence`.
- `Utils/detections.detection_path`: shared cache/supplied-file resolver for training/tracking.
- `Utils/detections.validate_cache`: checks coverage, recipe, source path/dimensions, and checksum.
- `Utils/detections.detection_settings`: records source compatibility in the split manifest, checkpoints, and tracking outputs.

This changes box generation only. The appearance encoder remains frozen DETR ResNet50; the graph, matching, and losses train as before. RT-DETRv2 itself is not fine-tuned on MOT labels. Report these generated boxes as a custom/private-detection experiment, rather than a result using MOT's supplied public detections. Compare held-out validation scores before claiming improvement over SDP.

Reference: [Transformers 4.49 RT-DETRv2 API](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/rt_detr_v2).
