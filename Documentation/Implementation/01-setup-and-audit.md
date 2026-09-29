# Existing Conda environment and Utils/config.py

[Back to TODO](README.md)

Use the Conda environment you already configured. Check that `import torch` works and that `torch.cuda.is_available()` is true if using the GPU. Install only packages missing from the chosen code path. No environment-check script or packaging setup is needed.

The settings file is now [Utils/config.py](../../Utils/config.py). It contains plain variables. The remaining scripts will read these values rather than repeat them. It includes:

- Dataset/detection paths and which format/coordinate origin the files use.
- Training, validation, and held-out sequences/frame ranges.
- Output folder and checkpoint path.
- Device, crop batch size, feature dimension, graph/model settings.
- Learning rate (initial reference: `1e-4`), epochs, seed, loss weights.
- Detection/match/birth thresholds, maximum missed frames, and video on/off.

Keep settings in this file so you can run `python train.py`, `python tracking.py`, and `python evaluate.py` without building command-line options. Validate essential paths when each script starts and print useful missing-file messages. Use one graph at a time and FP32 initially.

## Using the file

From a Python script at the project root:

```python
from Utils import config

print(config.DATASET_ROOT)
print(config.LEARNING_RATE)
```

`PROJECT_ROOT` is calculated automatically. The initial data choice is the local MOT17 FRCNN detections: sequence 02 for training, 05 for validation, and 09 for evaluation. Training and validation start with 100 frames each; evaluation uses the whole selected sequence. These are editable development choices, not published benchmark splits.

`TRACKEVAL_ROOT` still needs an evaluator location. `MATCH_THRESHOLD=0.1` is now a provisional normalized-score floor for the implemented tracker, with explicit unmatched costs; tune it on validation. MOT20's path is reserved for later; its data was not found locally. No output folders or checkpoints are created by importing this file. `tqdm` is installed in `dragon-track` for progress bars; use `SHOW_PROGRESS=False` to hide them.

In your activated Conda terminal, run this from the project root to check PyTorch and the GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The configuration import and existing sequence paths have been checked. On 27 September 2026, PyTorch 2.6.0+cu124 in `dragon-track` detected the RTX 4060 Laptop GPU and completed a CUDA tensor calculation. NVIDIA driver 595.91.07 was active. The earlier failure was traced to a missing NVIDIA module for the running kernel; the driver is now available. The current config also imports PyTorch for its GPU-check helper.

After the first successful run, save the settings and useful package versions with the output. A short `results.md` and settings in the checkpoint are enough. Keep the upstream code revision and any method differences in those notes; preserve upstream copyright/license notices in reused code.

**Done when:** the existing environment can run a small PyTorch operation, dataset/output paths are set, and both developers know which settings to import. No `.toml`, configuration loader classes, or extra audit documents.
