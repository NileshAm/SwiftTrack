# Existing Conda environment and config.py

[Back to TODO](README.md)

Use the Conda environment you already configured. Check that `import torch` works and that `torch.cuda.is_available()` is true if using the GPU. Install only packages missing from the chosen code path. No environment-check script or packaging setup is needed.

Create **`config.py`** with plain variables or one dictionary. Include:

- Dataset/detection paths and which format/coordinate origin the files use.
- Training, validation, and held-out sequences/frame ranges.
- Output folder and checkpoint path.
- Device, crop batch size, feature dimension, graph/model settings.
- Learning rate (initial reference: `1e-4`), epochs, seed, loss weights.
- Detection/match/birth thresholds, maximum missed frames, and video on/off.

Keep settings in this file so you can run `python train.py`, `python track.py`, and `python evaluate.py` without building command-line options. Validate essential paths when each script starts and print useful missing-file messages. Use one graph at a time and FP32 initially.

After the first successful run, save the settings and useful package versions with the output. A short `results.md` and settings in the checkpoint are enough. Keep the upstream code revision and any method differences in those notes; preserve upstream copyright/license notices in reused code.

**Done when:** the existing environment can run a small PyTorch operation, dataset/output paths are set, and both developers know which settings to import. No `.toml`, configuration loader classes, or extra audit documents.
