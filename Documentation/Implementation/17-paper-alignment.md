# Paper-aligned training and tracking

> **Current implementation:** frozen pretrained appearance, corrected matching and projected self-feature residuals in both graph layers, `RUN_NAME="mot17_appearance_residual_v1"`. Follow the [residual guide](22-appearance-residual.md) for current training and checkpoint compatibility. The original neighbour-only architecture and run names below describe earlier development. Residual attention is an experimental extension, not an exact paper reproduction.

[Back to TODO](README.md) · [Settings](../../Utils/config.py) · [Paper](https://openaccess.thecvf.com/content/WACV2025/html/Galoaa_DragonTrack_Transformer-Enhanced_Graphical_Multi-Person_Tracking_in_Complex_Scenarios_WACV_2025_paper.html) · [Authors' code](https://github.com/ostadabbas/DragonTrack)

This is the current implementation guide. It supersedes the architecture, loss, and default-run settings in guides 05 and 11. Motion/history and chronological training from guides 15–16 remain active. The paper and released code disagree in several places; this implementation follows the published equations where possible and identifies the remaining choices below. It is not a verified reproduction of the published benchmark scores.

## Run

Use the existing environment from the project root:

```bash
conda activate dragon-track
python import_tracktor.py
python train.py
python tracking.py
python evaluate.py
```

The Tracktor archive and imported detections for the configured 11 videos are already available locally. The importer verifies and reuses them. No new dependency installation is required. `detect.py` skips inference for this backend; the existing `run.sh` works once import is complete.

Start a **fresh training run**: `RUN_NAME="mot17_fixed_matching"`. Earlier graph checkpoints are incompatible and are rejected. The configured five training videos and two validation videos remain separate; each uses all its frames. `TEST_SEQUENCES` selects four additional test videos. Set it to `[]` if you only want validation tracking. A different run name preserves existing results.

```text
outputs/mot17_fixed_matching/
  model.pt, last.pt, history.csv
  plots/training/
  tracking/paper_motion_history/val/<sequence>/
  tracking/paper_motion_history/test/<sequence>/
  evaluation/paper_motion_history/val/results.md
```

## What follows the paper

| Component | Current behavior | Reference |
| --- | --- | --- |
| Position relationship | Top-left displacement and Euclidean distance; three values | Eq. 1 |
| Graph attention | LeakyReLU edge-conditioned score, neighbor softmax, weighted sum divided by degree, ReLU | Eqs. 2–4 |
| Graph widths | Two layers: 512, then 128 | Sec. 4.1 |
| Association | Cosine similarity, fixed-marginal normalization with learned unmatched preference, Hungarian assignment | Fig. 2 ordering; normalization superseded by [repair](19-matching-and-loss-fix.md) |
| Pair supervision | Same-identity positives; different-identity negatives; same-frame and cross-frame contrastive comparisons; hard/random negative mining | Sec. 3.5 |
| Objective | Sample-weighted BCE sum and contrastive sum, combined as `alpha*WBCE + (1-alpha)*contrastive` | Eqs. 8–10 |
| Spatial/temporal limits | 200 original-image pixels; maximum 30-frame observation gap | Sec. 3.7 |
| Optimization | Adam, learning rate `1e-4`, 20 epochs, two images per batch/update | Sec. 4.1 |

There is no extra assignment objective or custom final scoring MLP. The `assignment` CSV column remains zero solely to keep older plotting/history readers usable.

## Files and important functions

| File / function | What it does |
| --- | --- |
| [import_tracktor.py](../../import_tracktor.py): `import_sequence`, `main` | Validate archive checksum; import selected MOT17 videos; discard upstream track IDs; record provenance and output hashes. |
| [Utils/detections.py](../../Utils/detections.py): `detection_path`, `validate_tracktor` | Select the configured box source consistently for training and tracking. Reject missing, modified, or incompatible caches; never silently fall back. |
| [Utils/appearance.py](../../Utils/appearance.py): `DETRAppearance.extract_boxes` | Process a full image with pretrained ResNet50/DETR encoder; ROI Align each supplied box into a 256-dimensional appearance vector. Training gradients reach the backbone, projection, and transformer encoder. |
| [Utils/features.py](../../Utils/features.py): `load_encoder`, `extract_features` | Preserve row order and validity masks. Enable DETR gradients only in training when `FREEZE_ENCODER=False`. The optional DenseNet path remains frozen. |
| [Utils/graph.py](../../Utils/graph.py): `build_graph` | Build bidirectional track/detection edges within the spatial gate. Nodes contain appearance and normalized xywh. Raw edge columns are dx, dy, distance, IoU, cosine. For messages from neighbor j into i, displacement is position_i minus position_j. |
| [Utils/relationships.py](../../Utils/relationships.py): `RelationshipNetwork.forward` | Learn scalar appearance, geometry, and position relationships. Fuse appearance/geometry and pass that value plus position to attention. |
| [Utils/attention.py](../../Utils/attention.py): `GraphAttention.forward` | Keep degree-normalized neighbour messages and add a learned own-feature projection. Isolated nodes retain self features but remain unmatched. See [residual extension](22-appearance-residual.md). |
| [model.py](../../model.py): `AssociationModel.forward` | Run 512→128 attention, cosine similarity, and the matcher. Return scores, unmatched options, embeddings, and attention diagnostics. |
| [Utils/sinkhorn.py](../../Utils/sinkhorn.py): `sinkhorn`, `LearnableSinkhorn.forward` | Normalize masked rectangular scores in log space with fixed row/column totals and a learned unmatched preference. |
| [Utils/pairs.py](../../Utils/pairs.py): `build_targets`, `select_pairs`, `within_frame_pairs` | Keep identity labels outside model inputs; exclude unknown identities and self-pairs; select contrastive negatives. |
| [Utils/losses.py](../../Utils/losses.py): `compute_loss` | Apply WBCE to matching scores so the Sinkhorn parameters receive gradients; apply contrastive loss to unit-normalized final graph embeddings. |
| [train.py](../../train.py): `iter_graphs`, `_run_epoch`, `main` | Construct past-only chronological state, batch two images and backpropagate their supervised graph losses together, validate without updates, and save graph plus fine-tuned appearance weights together. |
| [Utils/track_filter.py](../../Utils/track_filter.py): `observe_confirmation`, `keep_tentative`, `export_confirmed` | Require repeated observations before keeping an identity in exported results. Remove all rows belonging to identities that never confirm. |
| [tracking.py](../../tracking.py): `main`, `run_sequence`, `make_evaluation_video` | Reload the compatible graph/appearance checkpoint, assign identities, export confirmed tracks, then render videos from those final results. The overall progress bar counts tracking and video frames. |

## Detection source and missing confidence

The authors' README links [MOT Neural Solver](https://github.com/dvl-tum/mot_neural_solver) for the Tracktor-preprocessed detections used in benchmark evaluation. The importer uses its [downloaded public archive](https://webshare.cvg.cit.tum.de/u/brasoand/mot_neural_solver/prepr_det_files.zip), selecting `tracktor_prepr_det.txt` for each configured SDP video. The exact archive SHA256 is pinned in config.

These are **preprocessed SDP/Tracktor boxes**, not the raw SDP `det/det.txt`. MOT top-left coordinates are converted from origin 1 by the existing data reader. The upstream IDs are discarded: our model must assign identities itself. Source rows with confidence `-1` have no usable confidence estimate, so the importer sets their score to `1` and records that conversion. This makes them eligible for births and gives them equal appearance-history weight; it does not mean the detector is perfectly confident.

The archive is external preprocessing. Our train/validation split prevents association-model training on validation videos; it is not a claim about the upstream detector's training data. Switch `DETECTION_BACKEND` to `supplied` or `rtdetrv2` only for a separate fresh run.

## Explicit choices where the recipe is incomplete

- **Appearance:** the paper does not fully specify how detection features map into graph nodes. We pool transformer encoder features inside the supplied boxes; we do not use object-query embeddings or train a box detector. `FREEZE_ENCODER=False` fine-tunes appearance extraction only. Existing detection files never change during training.
- **Memory and training:** the requested Kalman prediction and confidence-weighted appearance history remain enabled. Both training and tracking use those helpers, but training/validation-loss state updates are GT-guided after scoring. Tracking updates use predicted matches. Past features are detached, were computed with earlier encoder weights, and are not recomputed after every optimizer step. There is no full backpropagation through 30 video frames.
- **Graph wiring and scaling:** edges connect live tracks to current detections, without self-loops or same-type message edges. Same-frame contrastive losses still compare node embeddings. Positions/xywh are normalized for learned features; the spatial gate uses raw pixels. The precise wiring and normalization are implementation choices. Scalar Linear+ReLU affinity modules follow the released code; this is not a reconstruction of unspecified hidden layers.
- **Matching:** fixed row/column totals with a learned unmatched logit replace the earlier learned marginal scales. The solver uses 50 iterations and a feasibility correction for remaining error. This intentionally departs from the literal paper equations. See [the repair guide](19-matching-and-loss-fix.md).
- **Loss defaults:** `LOSS_ALPHA=0.5`, class weights 10:1, margin 1, negative ratio 3, and hard-negative fraction 0.5 are configurable. Class weights/margin follow released defaults; the full sampling and weighting recipe is not published. WBCE uses all valid gated cross-frame pairs; contrastive negatives are sampled only during training. Validation uses all eligible contrastive pairs, so its absolute loss is not directly comparable with training loss. GT matching is one-to-one, so duplicate same-frame positive observations are normally absent; no artificial self-positive pairs are created.
- **Batch size:** DETR now processes two images together. Graphs are scored chronologically, their losses share one backward pass, and gradients are averaged by the actual supervised count. Validation batches four images. Activation checkpointing is off by default after GPU speed/memory checks; it remains available to reduce memory. See [batched training](18-batched-training.md).
- **Confirmation:** defaults of three observations within 15 frames follow released tracking settings; exact expiration boundaries are our choice. Confirmed tracks can reconnect after a 30-frame gap; tentative tracks must first meet the confirmation rule. Export retains earlier observations of eventually confirmed IDs, making this an offline output filter. Preview, video, and evaluation use the same filtered identities. FPS excludes final preview/video rendering.

The released repository also contains active CNN/GCN paths, a different optimizer rate, and disabled learnable-Sinkhorn calls. Those paths were not copied over the paper's equations. Published results also use a different evaluation setup from this local five/two video split. Full training and measured validation accuracy remain necessary.

## Completion checklist

- [x] Paper-equation attention, cosine scoring, WBCE/contrastive objective, shared spatial gate.
- [x] Trainable DETR appearance with saved/reloaded encoder weights.
- [x] Tracktor importer and verified caches for all 11 configured videos.
- [x] Confirmation filtering and videos drawn from filtered results.
- [x] Checks of attention/Sinkhorn arithmetic, empty/rectangular graph gradients, 200-pixel boundary, detached history, 30-frame expiry, and confirmation counting.
- [x] Full-resolution GPU fine-tuning check on MOT17-04; about 1.26 GiB peak PyTorch allocation in that check, not a maximum for every video.
- [x] Six-frame train/validation run with gradient accumulation, unchanged validation weights, checkpoint reload, tracking, MP4 decoding, TrackEval, and charts. Small-run metrics are not accuracy evidence.
- [ ] **Owner: ___** Train the full fresh run.
- [ ] **Owner: ___** Evaluate full validation videos; compare HOTA/IDF1/MOTA and identity switches against saved prior runs.
- [ ] **Owner: ___** Tune uncertain defaults using validation; record changes and keep final-test data untouched.
