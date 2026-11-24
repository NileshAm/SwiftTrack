# Authors' DragonTrack release baseline

This directory vendors the [authors' DragonTrack repository](https://github.com/ostadabbas/DragonTrack) at commit `111e263c650f2f3a6ef39a261582eef87e09e8d8`. `UPSTREAM_README.md` and `LICENSE` came from that repository. The authors' network, graph, and tracking logic is retained; the three entry points here adapt it to this project's MOT17 folders and TrackEval reports.

The released training path required four small corrections before it could be used with the local MOT images: `data_preparation_train.py` calls the active `build_graph` signature and skips empty graphs; `build_graph.py` reads six digit MOT image names; `model_training.py` uses the six values returned by `completeNet` and its existing weighted BCE loss, since the released contrastive branch is commented out; `utils.py` clamps BCE probabilities away from exactly zero and one. These changes can affect numerical results. This is a **reproduction attempt from a pinned source snapshot**, not a claim that the paper's reported score has been reproduced.

The authors' release trains on MOT ground truth person boxes. The wrapper uses the disjoint complete videos listed in `Utils/config.py` for train and validation, then the authors' model and tracker for inference. Inference accepts GT boxes (without IDs), Tracktor, and YOLOX so the same model can be compared under three box sources. All scores are local validation scores on the selected MOT17 videos, not official test set submissions. GT boxes are an upper control; GT identities never enter the inference model.

## Run from the project root

Activate the existing GPU enabled `dragon-track` Conda environment first. This baseline needs PyTorch, PyTorch Geometric, OpenCV, SciPy, Pillow, Matplotlib, tqdm and TrackEval. It uses the same environment as this project.

```bash
conda activate dragon-track
cd /home/nilesh/UOM/CV/Project/ModifiedCode
python Baselines/DragonTrackGithub/train.py --epochs 20 --batch-size 2 --output-dir outputs/github_dragontrack/baseline_train
```

The trainer saves `outputs/github_dragontrack/baseline_train/models_v3/epoch_N.pth`. Substitute the epoch you want to evaluate:

```bash
python Baselines/DragonTrackGithub/infer.py --checkpoint outputs/github_dragontrack/baseline_train/models_v3/epoch_20.pth --detectors gt tracktor yolox --detection-min-score 0.5 --output-dir outputs/github_dragontrack/baseline_val
python Baselines/DragonTrackGithub/evaluate.py outputs/github_dragontrack/baseline_val
```

Read `outputs/github_dragontrack/baseline_val/evaluation/results.md`. The inference command requires the existing Tracktor and YOLOX detection caches; their locations are arguments `--tracktor-cache-dir` and `--yolox-cache-dir` if you use different folders. A fresh output directory is required for each run. A resumed training run accepts `--checkpoint` and a total `--epochs` target, with another fresh output directory.

**Status:** prepared by static review only. The user will run these commands. Training, inference, and scoring have not been executed for this baseline, so no authors' baseline result is reported yet.
