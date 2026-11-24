# Iter-E2EDET + fine-tuned OSNet comparison

The detector is the [authors' Iter-E2EDET (Iter Sparse R-CNN) implementation](https://github.com/megvii-research/Iter-E2EDET), pinned here at `c649a0b29be48e7253048a2c8efd6b28a2e3197b`. Source and its MIT license are in `ThirdParty/IterE2EDET`. The [official CrowdHuman pretrained checkpoint](https://drive.google.com/file/d/1LTP-Qfe6QsnhCOL3e-lxuuXEqfJ55sgj/view?usp=sharing) must be downloaded separately. Put it at `weights/iter_e2edet/model.pth` or use your own path below. This path uses the full authors' model and config, with old NumPy type aliases updated for the project's NumPy 2 environment. It exports detections and evaluates them on MOT17; no Iter-E2EDET fine-tuning is included.

The detector uses the authors' Detectron2 fork, which must be built in the existing `dragon-track` environment before export. This vendored build includes CUDA headers installed with PyTorch's pip packages, including `cusparse.h`. The build may still need a compatible compiler/CUDA toolkit for your installed PyTorch. The commands below have been prepared but **not run**.

```bash
conda activate dragon-track
cd /home/nilesh/UOM/CV/Project/ModifiedCode
MAX_JOBS=4 python -m pip install --no-build-isolation -e ThirdParty/IterE2EDET
mkdir -p weights/iter_e2edet
python -m pip install gdown
gdown 'https://drive.google.com/file/d/1LTP-Qfe6QsnhCOL3e-lxuuXEqfJ55sgj/view' -O weights/iter_e2edet/model.pth
python detect-itere2edet.py --checkpoint weights/iter_e2edet/model.pth --split val --min-score 0.05
python evaluate_detector.py --backend iter_e2edet --cache-dir detections/iter_e2edet_resolution_v2/higher_960_1600 --scores 0.05 0.1 0.2 0.3 0.5 --ious 0.5 0.75 --output-dir outputs/detector_evaluation/iter_e2edet_val
python evaluate-itere2edet-osnet.py --iter-checkpoint weights/iter_e2edet/model.pth --detection-min-score 0.5 --new-track-min-score 0.7
```

The detector's default input is now 960/1600, using the completed `detections/iter_e2edet_resolution_v2/higher_960_1600` cache as the baseline. The evaluation command reads that cache; it does not run the detector again. Add `--smoke-test` to check the cache and tracker on one frame before the full run. The default 0.5/0.7 scores match the earlier Tracktor/YOLOX tracker comparison. For an additional detector-F1 operating point, run a separate output directory with `--detection-min-score 0.3 --new-track-min-score 0.7`; this changes the tracker input and is not directly the same operating point as the earlier 0.5 comparison.

`detect-itere2edet.py` saves full-video MOT `det.txt` files, one checksum and recipe per sequence, without GT identities. The authors' config uses 800/1333; this project's selected inference baseline uses 960/1600 by default. Pass `--short-side` and `--max-size` to reproduce another arm. The exported score floor is 0.05; the tracker score threshold may be raised later without rerunning detection.

The detector-only report is `outputs/detector_evaluation/iter_e2edet_val/report.md`, with combined and per-video precision, recall, F1, TP, FP, FN and small/occluded-person recall. Use a fresh `--output-dir` for another threshold experiment. The cache's `summary.json` records the exact model and configuration hashes; a complete matching cache is reused on rerun.

`evaluate-itere2edet-osnet.py` fixes the tracker to the fine-tuned MSMT17 OSNet features with **graph attention off** and the existing learned Sinkhorn association checkpoint. It records HOTA, DetA, AssA, IDF1, MOTA, and ID switches on the configured complete MOT17 validation videos. It does not fine-tune the association model for these OSNet features. Change `--osnet-weights` or `--association-checkpoint` to use other local weights. Its output folder contains `comparison.md`, `comparison.csv`, per-sequence TrackEval data, and a manifest with detector and checkpoint hashes.

These are development-validation scores; the CrowdHuman AP reported by the authors is on another dataset and cannot be compared directly with MOT17 HOTA. Keep the MOT17 video split, tracking thresholds, and OSNet checkpoint the same when comparing Tracktor, YOLOX, and Iter-E2EDET.
