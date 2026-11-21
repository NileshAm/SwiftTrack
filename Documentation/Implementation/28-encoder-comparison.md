# DenseNet, DETR, and pretrained person ReID

The shared Tracktor detections are imported with `python import_tracktor.py`.
The importer validates the pinned archive checksum and all selected videos.
`Utils/detections.py` supports Python 3.10 with streaming SHA256.

## Three appearance encoders

| Launcher name | Backbone / pretrained task | Feature width |
| --- | --- | --- |
| densenet | DenseNet121 / ImageNet classification | 1024 |
| detr | DETR ResNet50 / COCO detection encoder | 256 |
| reid | OSNet x1.0 / MSMT17 person identification, combineall | 512 |

This is an encoder ablation in the current SwiftTrack association pipeline.
The DenseNet arm is not a reproduction of the entire original DragonTrack system.
Feature widths and whole-frame versus crop processing differ, so graph input
parameter counts and computation differ too; report those differences.

OSNet produces appearance features **before** RelationshipNetwork's learned
affinities. Geometry still uses two normalized xywh boxes; position uses dx, dy,
and Euclidean distance. ReID does not estimate calibrated geometry or position.
Better identity features are a hypothesis to evaluate, not a promised improvement.

The official OSNet source is vendored in `Utils/vendor/osnet.py`, revision
`f8cd150fdf77e8d9e1ed143b7f308c2c609ded50` from
https://github.com/KaiyangZhou/deep-person-reid (MIT license preserved alongside).
Weights are from the author's https://huggingface.co/kaiyangzhou/osnet repository,
revision `a5c5cc037c24235cda3b21085b93ad77c9616224`, filename
`osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`.
They reside at `weights/reid/osnet_x1_0_msmt17.pth` (ignored by Git).
All checkpoint tensors must load strictly. The frozen evaluation embedding is
512-dimensional and L2-normalized; RGB person crops use 256x128 bilinear resizing
and ImageNet normalization. Model fingerprints include the weights' SHA256.

## Commands

Use the activated `dragontrack` environment in the project root:

```powershell
python encoder_experiment.py --encoder reid --stage smoke
python encoder_experiment.py --encoder densenet --stage smoke
python encoder_experiment.py --encoder detr --stage smoke
```

Smoke checks use four boxes from a real validation frame, verify finite features,
graph loss/backward, detection alignment and invalid boxes. Synthetic same-frame
targets test integration only; they are not accuracy results.

Run each arm separately with the same total epoch budget:

```powershell
python encoder_experiment.py --encoder densenet --stage all --epochs 30
python encoder_experiment.py --encoder detr --stage all --epochs 30
python encoder_experiment.py --encoder reid --stage all --epochs 30
```

`all` trains, tracks validation, then evaluates. Alternatively choose `train`,
`track`, or `evaluate`. No config file edits are made. Each arm uses its own
`outputs/encoder_comparison_<name>_v1` directory and compatible association weights.
All appearance encoders are frozen. Test videos and videos are disabled for these
comparisons. Other settings come from config: retain them across arms. Resume
follows config.RESUME_TRAINING; epoch budgets are total targets. Existing tracking
outputs remain protected by the tracker. Do not rerun `all` over completed tracks.

DETR downloads its pinned Hugging Face weights on first use. Setup downloaded
those weights and verified all three encoder smoke checks in `dragontrack`.
TrackEval 1.0.0 was installed for evaluation; it supports the existing NumPy 1.x
environment without the upgrades required by 1.3.0. Full training is intentionally
not launched by setup.
Compare HOTA, IDF1, MOTA, ID switches, speed and memory; validation is used for
selection, so it is not an independent final test. Keep tracker thresholds fixed
for the initial comparison. Report normalization differences (OSNet L2, DenseNet
raw pooled features, DETR L2) and source pretraining data alongside results.
