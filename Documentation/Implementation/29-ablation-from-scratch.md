# Run ablations from a code-only checkout

No existing `outputs/`, detection archive, or OSNet weights need to be shared.
Use the existing working Conda environment and place MOT17 under the paths in
`Utils/config.py`. The configured five training and two validation videos need
all images, `seqinfo.ini` and `gt/gt.txt`. Test videos are not required.

## One command

```powershell
conda activate dragontrack
python gcn_ablation.py --encoder all --prepare --epochs 4
```

This runs sequentially:

1. Check dependencies, CUDA, labelled video metadata and image counts.
2. Download and SHA256-check the official Tracktor public-detection archive.
3. Download and SHA256-check the author's MSMT17 OSNet ReID weights.
4. Import detections for training/validation videos only (no test videos).
5. For DenseNet, DETR, then OSNet: load frozen pretrained features, smoke-check
   extraction, and train association weights for four total epochs. DenseNet and
   DETR pretrained weights download through their standard libraries as needed.
6. For each encoder, track/evaluate with the residual graph, then without it.
7. Write combined comparison tables, per-arm plots/logs, and provenance manifests.

The no-graph arm uses raw appearance cosine and the same trained Sinkhorn matcher
as its graph arm. No retraining of a separate no-graph model is performed. Scores
may need separate calibration in a subsequent experiment.

## Files generated

```text
detections/tracktor_public/prepr_det_files.zip
detections/tracktor_public/train/<sequence>/det.txt, summary.json
weights/reid/osnet_x1_0_msmt17.pth
outputs/encoder_comparison_<encoder>_scratch_4ep/
  model.pt, last.pt, history.csv, setup_manifest.json, plots/
outputs/gcn_ablations/<timestamp>/
  comparison.md, comparison.csv
  <encoder>/<with_gcn|without_gcn>/
    manifest.json, execution.log, tracking/, evaluation/
```

Each arm uses the selected best-validation-loss checkpoint for its encoder; the
selected epoch can be earlier than the total training budget. Checkpoint hashes
and selected epochs are recorded. This is a new local experiment: matching seeds
does not guarantee identical weights/scores across hardware or library versions.

## Restarting and existing results

Rerun the same command after interruption. Verified downloads and imported
detections are reused. Compatible training resumes from the last completed epoch;
an interrupted epoch restarts. Completed training is checked and skipped. Existing
incompatible checkpoints are rejected instead of overwritten. Use a fresh
`--run-tag another_name` to isolate a new experiment. To prevent accidental unequal
budgets, a namespace trained beyond the requested epoch target is rejected.

Ablations always use a new timestamped directory; interrupted tracking restarts
in that fresh directory, while older partial/complete results remain untouched.
Keep the computer awake and plugged in. Do not train these checkpoint files in
another process while ablation is running.

Prepare only, without tracking/evaluation:

```powershell
python prepare_ablation.py --encoder all --epochs 4
python gcn_ablation.py --encoder all --run-tag scratch_4ep
```

For OSNet only use `--encoder reid`. Without `--prepare` or `--run-tag`, the
original ablation command still uses `encoder_comparison_<encoder>_v1` checkpoints.
All overrides are in memory; `Utils/config.py` is not rewritten.

## Environment

Setup checks packages but does not install or replace environment packages.
Use a compatible PyTorch/torchvision pair with CUDA. Known project versions are
`transformers==4.49.0`, `timm==1.0.15`, NumPy 1.x, and `trackeval==1.0.0`.
Core imports also require scipy, pillow, tqdm and matplotlib. For an otherwise
prepared PyTorch environment:

```powershell
python -m pip install "numpy<2" scipy pillow tqdm matplotlib transformers==4.49.0 timm==1.0.15 trackeval==1.0.0
```

The official OSNet architecture and license must be in `Utils/vendor/` as part
of the code checkout. Internet is needed for uncached public models. A failed
download is kept as `.download`, never promoted to the expected asset path;
rerunning restarts that download. Existing assets with wrong hashes are rejected.
No detector fine-tuning or private teammate checkpoint is required for this study.
