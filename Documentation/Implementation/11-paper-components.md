# DETR appearance, attention, matching, and test videos

> **Current training:** [chronological motion/history state](16-state-training.md) supersedes the historical frame-pair protocol below. Use the fresh `mot17_SDP_state30` run.

This replaces the DenseNet/mean-message baseline. Use a new run: `RUN_NAME="mot17_SDP_video_split"`. Old architecture and percentage-split checkpoints cannot be used for this run; old output folders are preserved. Run in the existing `dragon-track` Conda environment.

```bash
python train.py
python tracking.py
python evaluate.py
```

The official DETR checkpoint is cached. `transformers==4.49.0` and `timm==1.0.15` are installed; PyTorch/CUDA were not replaced. If recreating the environment, install those two versions with pip in that environment. No new packaging or test framework is required.

## Implementation checklist

- [x] `Utils/appearance.py`: full-image pretrained DETR transformer features.
- [x] `Utils/relationships.py`: independent appearance, geometry, and position networks.
- [x] `Utils/attention.py`: learned, per-neighbor graph attention.
- [x] `Utils/sinkhorn.py`: trainable normalization scales, temperature, unmatched score.
- [x] `Utils/pairs.py`: identity targets and hard/random negative selection.
- [x] `Utils/losses.py`: weighted BCE, contrastive, and direct assignment supervision.
- [x] `model.py` and `train.py`: integration, temporal-gap sampling, new checkpoint version.
- [x] `Utils/outputs.py`, `tracking.py`, `evaluate.py`: separate validation/test sequence folders and videos.
- [x] `Utils/splits.py`: complete-video lists, detector-variant overlap guard, saved split manifest.
- [x] Tiny real-MOT17 GPU run: training, validation, checkpoint reload, tracking, MP4 decoding, TrackEval.
- [ ] Train the full new run and inspect validation losses.
- [ ] Tune tracking thresholds on validation; reserve a separate untouched dataset for final test claims.
- [ ] Measure accuracy before claiming improvement or published-paper reproduction.

## Appearance — `Utils/appearance.py`

`DETRAppearance.__init__(device)` loads the official `facebook/detr-resnet-50` checkpoint at the exact revision in config. It retains the ResNet backbone, input projection, and six-layer transformer encoder. It does not run the detector decoder. Weights remain frozen; trainable appearance adaptation happens in the graph model.

`extract_boxes(image, boxes)` processes the complete RGB image once, respecting `DETR_SHORT_SIDE=800` and `DETR_MAX_SIZE=1333`. The transformer sees the full scene. Each supplied box is clipped, scaled into the actual feature grid, and pooled with 3×3 ROI Align followed by spatial averaging and L2 normalization. It returns `[D,256]` CPU features and a `[D]` validity mask in original detection order. Invalid/outside boxes remain zero and must be filtered using the mask. Empty frames do not run the encoder.

`Utils/features.py` keeps the public `load_encoder()` / `extract_features()` interface and dispatches to this utility. DenseNet is available only if explicitly configured with its original feature width. `CROP_SIZE` and `CROP_BATCH_SIZE` do not affect DETR.

This is an integration choice: transformer encoder ROI features aligned to supplied detections, rather than paper-equivalent detector-query features. FRCNN boxes still control detection coverage, and DETR itself is not fine-tuned.

## Relationships — `Utils/relationships.py`

`RelationshipNetwork.forward(nodes, edges, attributes)` learns three separate scalar affinities for every directed edge:

| Network | Inputs |
| --- | --- |
| Appearance | Both 256-dimensional appearance vectors |
| Geometry | Both normalized xyxy boxes: eight numbers |
| Position | Normalized dx/dy and log width/height ratios: four numbers |

Each small network ends in a sigmoid. Its three outputs are appended to the six original geometry/appearance measurements, giving nine edge attributes. `Utils/graph.py` continues to build track-to-detection edges in both directions, keeping training labels outside the graph. With DETR, node width is 260: 256 appearance values plus four coordinates.

## Attention — `Utils/attention.py`

`GraphAttention.forward(nodes, edges, attributes)` computes attention logits from projected source/destination nodes and the nine edge attributes, applies LeakyReLU, then softmax over incoming neighbors separately for each head. The four heads learn which neighbors to emphasize. Their weighted messages pass through a residual update and layer normalization. Empty/isolated graphs remain valid.

`model.py` combines two attention layers with the existing pair scorer. Its output includes `attention_weights` (one tensor per layer) and `relationships` for inspection. The last three relationship columns are the learned affinities. Attention sums to one per destination/head when that destination has neighbors.

## Matching — `Utils/sinkhorn.py`

`LearnableSinkhorn.forward(logits, mask)` runs the paper's alternating scaled row/column normalization in log space, extended to rectangular track/detection matrices and unmatched options. It learns row scale, column scale, temperature, and unmatched logit. All four parameters are saved in the association checkpoint and optimized by Adam.

Practical deviations are explicit: normalization scales use `1 + softplus(raw)` to keep real-pair scores probability-safe; epsilon is `1e-4`; dustbin masses account for unequal counts; temperature is bounded to `[0.1,10]`. Global row/column factors can largely cancel between alternating normalizations, so the extra temperature/unmatched parameters allow useful matching adaptation. This is a constrained implementation inspired by equations 5–6, not a claim of identical published code.

`marginal_error` reports deviation from the unscaled target masses. Learned scales intentionally change those masses, so nonzero error is not necessarily failed convergence. Masked edges get zero probability. Completely isolated nodes receive an unmatched score of one. Hungarian assignment still produces the final one-to-one matches.

The standalone `sinkhorn()` function remains available for ordinary, fixed normalization; the actual model uses the trainable class.

## Pairs and losses — `Utils/pairs.py`, `Utils/losses.py`

`build_targets()` labels equal valid identities positive and different valid identities negative. Ignored/unlabelled detections remain excluded. `select_pairs()` keeps every positive and up to `NEGATIVE_RATIO=3` negatives per positive. Half the negatives are the hardest available and the remainder random. With no positives, up to three negatives are retained. Validation uses all pairs, with no random mining.

Training now randomly samples historical observations with gaps 1..`LOOKBACK_FRAMES` (30). Validation cycles deterministically through the same gaps; neither crosses video boundaries or uses future frames. The tracker uses the same maximum observation age. These remain two-frame association examples, not predicted-track rollouts. See the [look-back guide](14-lookback.md).

`within_frame_pairs()` also supplies distinct same-frame pairs to the contrastive loss. In this dataset one-to-one GT labeling normally yields only same-frame negatives; cross-frame observations provide positives. We do not fabricate positives by matching a feature to itself. Additional same-frame augmentations are not implemented.

`compute_loss()` combines:

1. Class-balanced binary cross entropy on raw pair logits.
2. Class-balanced margin contrastive loss on L2-normalized graph embeddings, including within-frame and cross-frame pairs.
3. Assignment negative log likelihood on normalized Sinkhorn scores for every true match and known unmatched arrival/departure. This gives the matching module direct gradients. Unknown IDs never become unmatched targets.

The third term is an explicit engineering addition to the paper's described BCE/contrastive combination. Mining never removes positives from assignment targets. `ASSIGNMENT_WEIGHT`, `BCE_WEIGHT`, and `CONTRASTIVE_WEIGHT` control contributions. `history.csv` logs all three components for both training and validation. Gradients are clipped before each update.

These defaults are implementation choices where the paper does not give a complete reproducible recipe; they are not claimed as exact published hyperparameters.

## Whole-video selection — `Utils/splits.py`

Edit `DATASET_ROOT`, `DATASET_SPLIT`, `TRAIN_SEQUENCES`, and `VAL_SEQUENCES` in `Utils/config.py`. Each entry is a sequence folder name under `DATASET_ROOT / DATASET_SPLIT`. MOT17 stores all labelled sequences under `train`; both our lists refer to that physical folder. No percentage settings are used.

```python
TRAIN_SEQUENCES = [
    "MOT17-04-SDP", "MOT17-05-SDP", "MOT17-10-SDP",
    "MOT17-11-SDP", "MOT17-13-SDP",
]
VAL_SEQUENCES = ["MOT17-02-SDP", "MOT17-09-SDP"]
```

This is an editable example, not a prescribed paper split. All frames in the first list train the association model; all frames in the second validate it. Validation also selects the checkpoint, so its tracking scores are tuning results, not an independent final test. Similar scenes or people can still occur in different videos; this guard checks video membership, not visual content.

- `video_key(sequence)` removes known detector suffixes so FRCNN/SDP/DPM versions of one video belong to the same group.
- `validate_video_splits()` rejects empty lists, duplicate names, and base videos shared across training and validation.
- `full_frame_range(path)` returns `(1, seqLength)` from sequence metadata.
- `split_manifest()` records the protocol, dataset location, video membership and frame ranges in checkpoints and tracking summaries.
- `check_checkpoint_split(checkpoint)` rejects old temporal-split checkpoints or changed video assignments. Start a new `RUN_NAME` and retrain; relabelling an old checkpoint would not remove its previous exposure to validation data.

`train.load_frame_pairs()` uses full videos only from the requested configured list. Temporal-gap sampling stays in the same video. `tracking.py` resets tracker state for every video. `evaluate.py` checks the saved manifest and complete frame coverage before scoring `VAL_SEQUENCES`.

## Tracking selections and output folders

```python
TRACK_SPLIT = "val"             # Full validation videos; no temporal eval remainder.
RUN_SPLIT_TRACKING = True       # False: run only TEST_SEQUENCES.
TEST_SEQUENCES = ["MOT17-01-SDP", "MOT17-03-SDP"]  # [] to skip test tracking.
TEST_DATASET_ROOT = MOT17_ROOT
TEST_DATASET_SPLIT = "test"
SAVE_TEST_VIDEO = True
```

Test videos require sequence metadata, images, and detections; no GT is read. They never enter training or TrackEval. `Utils.outputs.tracking_context()` resolves full validation/test videos and their per-sequence folders consistently.

```text
outputs/mot17_SDP_video_split/
    model.pt
    last.pt
    history.csv
    tracking/
        val/MOT17-02-SDP/
            tracks.txt
            summary.json
            preview.jpg
            video.mp4          # SAVE_VIDEO=True, or generated later.
        test/MOT17-01-SDP/
            tracks.txt
            summary.json
            preview.jpg
            video.mp4
    evaluation/val/
        results.md
        MOT17/DragonTrack/     # Native combined and per-sequence metrics.
```

Existing files are protected. To run new test videos after validation, use `RUN_SPLIT_TRACKING=False` and select unprocessed test sequences or a fresh `TEST_OUTPUT_DIR`. `make_evaluation_video(sequence, split="val")` makes a video from saved validation predictions; `split="test"` uses saved test predictions. It refuses to overwrite existing videos. `split="eval"` is no longer used.

## What was verified

The full-video update was checked using separate short MOT17-04 training and MOT17-02 validation fixtures, plus an unlabelled MOT17-01 test fixture. Checks cover overlap rejection across detector variants, old/changed checkpoint rejection, complete frame coverage, train/reload/track/TrackEval, and video export. Full training on the new split remains to be run.

Earlier architecture checks (using the previous temporal protocol):

On the RTX 4060, a temporary real-data fixture completed one training epoch on frames 1–6 of MOT17-02, validation/tracking on 7–8, and full tracking of a two-frame MOT17-01 test fixture without GT. Checkpoint reload, native TrackEval, automatic test MP4, and regenerated validation MP4 passed. Both videos decoded as two 1920×1080 frames. This checks integration, not tracking accuracy.

Additional in-memory checks verified attention normalization, finite gradients through every model parameter (including all four matching parameters), empty/rectangular/fully masked graphs, and Hungarian compatibility. Full new-model accuracy has not yet been measured.

Sources: [DragonTrack paper, sections 3.1–3.6](https://openaccess.thecvf.com/content/WACV2025/papers/Galoaa_DragonTrack_Transformer-Enhanced_Graphical_Multi-Person_Tracking_in_Complex_Scenarios_WACV_2025_paper.pdf), [DETR model documentation](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/detr).
