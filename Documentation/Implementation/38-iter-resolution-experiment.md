# Iter-E2EDET inference resolution comparison

This uses the existing official CrowdHuman checkpoint. It does not train or change detector or tracker weights. The three input sizes use the same `detect-itere2edet.py` preprocessing, original-image box conversion, top-500 proposal output and no NMS. New caches are created under a separate root.

Run from the project root in `dragon-track`:

```bash
conda activate dragon-track
cd /home/nilesh/UOM/CV/Project/ModifiedCode
python experiment-itere2edet-resolution.py \
  --checkpoint weights/iter_e2edet/model.pth \
  --cache-root detections/iter_e2edet_resolution_v2 \
  --output-dir outputs/iter_e2edet_resolution/my_run \
  --device cuda
```

The script first checks one GPU frame at 800/1333, 960/1600 and 1080/1920. It records the exact status and skips a size if it fails, including OOM; it does not reduce the requested size. Successful sizes each get a full cache, detector evaluation at IoU 0.5 and 0.75 with confidence thresholds 0.10, 0.20, 0.30, 0.50 and 0.70, and a frozen-OSNet/no-GCN tracking evaluation at detection score 0.50 and new-track score 0.70. Original Iter and YOLOX-X baseline metrics are included in the consolidated comparison.

Open `outputs/iter_e2edet_resolution/my_run/report.md` after completion. `status.json` and the per-stage logs record failures. The report distinguishes detector FPS from tracker FPS and includes FP/FN and identity metrics. Full detector threshold sweeps and per-video counts are in `detector/<size>/summary.csv`; tracker outputs are in `tracker/<size>/`.

Use a new `--cache-root` and `--output-dir` for another run. The experiment verifies existing cache summaries before reusing completed new-cache files and does not write to `detections/iter_e2edet_crowdhuman_v1` or `detections/yolox_x_mot20_val_v1`.

## Next step: tracking thresholds on 960/1600

The resolution comparison selected 960/1600 as the current Iter-E2EDET baseline. Search the existing cache with the same frozen OSNet checkpoint, GCN disabled, and no training:

```bash
python search-itere2edet-tracking.py \
  --cache-dir detections/iter_e2edet_resolution_v2/higher_960_1600 \
  --detection-scores 0.3 0.4 0.5 \
  --new-track-scores 0.5 0.7 \
  --output-dir outputs/iter_e2edet_960_threshold_search
```

This runs six combinations on the two complete validation videos, ranks them by HOTA then IDF1, and writes `report.md` and `results.csv` to the selected output directory. Rerunning the same command reuses completed trials. It rejects a changed grid in the same output directory. The 0.50/0.70 combination reproduces the current operating point for comparison. Select a promising setting here, then confirm it on untouched sequences before presenting a final score.
