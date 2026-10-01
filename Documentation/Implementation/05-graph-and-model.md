# Utils/graph.py and model.py — the association core

> **Updated implementation:** use [DETR, attention, matching, and test videos](11-paper-components.md) for the current code, settings, and per-sequence output paths. The notes below record the earlier baseline and its checks; DenseNet defaults, fixed Sinkhorn, percentage splits, and old flat tracking paths are superseded. Current training and validation use separate complete video lists.

[Back to TODO](README.md) · [Shared arrays](02-contracts.md)

## Utils/graph.py

[Implementation](../../Utils/graph.py) is now available. A node represents a track or current detection; an edge lets the later model exchange information between them. A candidate pair represents one possible identity match.

For a step-by-step demonstration, run `python main.py` from the project root in `dragon-track`. It prints node/edge shapes and sample attributes, shows labelled frames, draws candidate connections, and displays a cosine-similarity heatmap. It also saves `graph_demo.png` under `config.OUTPUT_DIR`. Change `FRAME_IDS`, `FOCUS_TRACK`, or `DEMO_DEVICE` near the top of `main.py`; set `SHOW_PLOT=False` to save without opening a window. The plotting helper is `Utils.plot.plot_graph_demo`. The demo was checked end-to-end on CPU with frames 1 and 2.

- **`pairwise_iou(boxes_a, boxes_b)`** is reused from `Utils/data.py` and is also importable from `Utils.graph`.
- **`build_graph(track_boxes, detection_boxes, track_features, detection_features, image_size, candidate_mask=None)`** returns CPU tensors. Pass `image_size=(height, width)`. Use previous-frame detections for an initial training example and actual track history during inference. Box/feature rows must already align. Ground-truth identities never influence edges.

### Exact layout

Let `T` be tracks, `D` detections, `F` appearance width, and `K` allowed pairs. By default every track connects to every detection, so `K=T*D`. There are no same-type edges or self-loops.

| Dictionary key | Shape / meaning |
| --- | --- |
| `node_features` | `[T+D,F+4]`: appearance followed by `(x1/width, y1/height, x2/width, y2/height)`. With DenseNet the width is 1028. |
| `node_type` | `[T+D]`: track=0, detection=1. Tracks occupy nodes `0..T-1`; detection row `d` is node `T+d`. |
| `candidate_pairs` | `[K,2]`: local track/detection row pairs, ordered by track first, then detection. These are not persistent IDs. |
| `edge_index` | `[2,2K]`: source/destination nodes. First K edges go track → detection; next K reverse them. |
| `edge_attr` | `[2K,6]`: attributes in the order described below. |
| `edge_pair_index` | `[2K]`: links both directions to the same candidate; score each candidate once. |
| `candidate_mask` | `[T,D]` boolean; all true by default. |
| `num_tracks`, `num_detections` | Python integers for assembling the rectangular score matrix. |

Edge columns are `dx`, `dy`, `log_width_ratio`, `log_height_ratio`, `iou`, `cosine`:

- `dx/dy`: destination center minus source center, normalized by image width/height.
- Size ratios: natural log of destination width/height divided by source width/height. Reverse edges negate these and `dx/dy`.
- IoU and cosine are unchanged in reverse. Zero appearance vectors give cosine 0. Raw node appearance features are not L2-normalized; normalization is only used for cosine.

Partially outside boxes retain original geometry, so normalized coordinates may be outside `[0,1]`. Invalid dimensions, nonfinite inputs, and mismatched feature rows raise useful errors. Empty sides and fully masked graphs return correctly shaped empty edges. Keep the feature width for empty inputs, e.g. `torch.empty((0,1024))`.

This complete bipartite graph and its six attributes are our documented **partial-reproduction development choice**, not a verified paper graph. The optional boolean mask supports explicit candidate rules, but no pruning is applied by default. IoU and similarity still use full pairwise matrices even when a mask is supplied.

### Use two frames

```python
from Utils import config
from Utils.data import load_sequence, read_image
from Utils.features import load_encoder, extract_features
from Utils.graph import build_graph

sequence = config.DATASET_ROOT / config.DATASET_SPLIT / config.SEQUENCES[0]
frames = load_sequence(sequence, frame_range=(1, 2))
encoder = load_encoder()
features1, valid1 = extract_features(read_image(frames[1]), frames[1]["boxes"], encoder, 2)
features2, valid2 = extract_features(read_image(frames[2]), frames[2]["boxes"], encoder, 2)
graph = build_graph(
    frames[1]["boxes"][valid1.numpy()],
    frames[2]["boxes"][valid2.numpy()],
    features1[valid1], features2[valid2],
    image_size=(frames[2]["height"], frames[2]["width"]),
)
print(graph["node_features"].shape)   # torch.Size([26, 1028])
print(graph["candidate_pairs"].shape) # torch.Size([169, 2])
print(graph["edge_index"].shape)      # torch.Size([2, 338])
```

Apply the same validity masks to scores, row IDs, and training labels outside this example. Candidate rows refer to these filtered inputs; retain their original row-ID mapping. Do not mix sequences or different image coordinate systems. Graph preparation detaches frozen inputs; learning happens in the later association network. Move graph tensors to its device when needed, leaving the integer counts unchanged.

**Checked in dragon-track:** known IoUs/non-square normalization; hand-computed forward/reverse attributes; candidate ordering and edge mapping; row permutations; optional/empty masks; empty sides; invalid inputs; a trainable layer's backward pass; actual pretrained MOT17 frame 1→2 features; and a synthetic graph with 200 nodes/19,200 directed edges. No permanent test files. Real MOT20 verification remains pending local data.

## model.py

[Implementation](../../model.py) now provides `AssociationModel`, `sinkhorn`, `build_targets`, and `compute_loss`. It uses plain PyTorch; no additional graph library is required.

### See it working in main.py

Run `python main.py` in your activated `dragon-track` environment. After the graph explanation, it prints model output shapes, raw logits and normalized scores for the highlighted track, unmatched scores, normalization error, and loss values. It saves `model_demo.png` beside `graph_demo.png` in `config.OUTPUT_DIR` and opens both figures.

The model figure compares random initial scores with scores after **50 updates on the same two frames**, alongside ground-truth targets and the total/BCE/contrastive loss curves. The last row and column show unmatched options; the grey bottom-right cell is hidden bookkeeping mass. Grey target cells are excluded from the loss. Ground truth is matched against the already-filtered boxes and never enters the model's forward pass.

Change `LEARNING_STEPS` near the top of [main.py](../../main.py) to `0` for inference and initial loss only. `DEMO_LEARNING_RATE` controls this tiny example; `FOCUS_TRACK` selects the row printed in detail. Use `DEMO_DEVICE="cpu"` if needed and `SHOW_PLOT=False` to save without windows. Missing ground truth or no valid supervision skips learning. The helpers are `main.print_model_output`, `main.run_model_demo`, and `Utils.plot.plot_model_demo`.

Each run starts with fresh association weights. This fits one example for explanation; it does not load/save a checkpoint or evaluate tracking accuracy. Use [train.py](07-training.md) for dataset training and checkpoint saving. Scores remain soft possibilities, not final track identities.

Checked end-to-end on CPU with MOT17-02 frames 1→2: 13×13 scores, 81 supervised pairs (9 positive, 72 negative), and total loss about 0.79849 → 0.000111 after 50 updates. Both saved figures rendered correctly. With the configured 20 Sinkhorn iterations, the final maximum mass error was about 0.13245 (including the aggregate unmatched row/column); normalization is approximate even when the raw-logit loss is small.

### AssociationModel

Create once with `model = AssociationModel().to(config.DEVICE)`, then call `output = model(graph)`. It automatically copies needed graph tensors onto the model's device without changing the input dictionary. Constructor overrides `feature_dim`, `hidden_dim`, and `num_layers` allow small debugging examples; defaults come from config.

The implemented computation is:

1. Project each node's appearance/geometry vector into the hidden width (default 128).
2. For every directed edge, compute a message from source features, destination features, and the six edge attributes.
3. Sum messages into the **destination node**, divide by its incoming-edge count, and update that node with a residual connection and layer normalization. Repeat for `GRAPH_LAYERS` (default 2). Isolated nodes use their own features plus a zero aggregate.
4. Score each unique candidate using both node representations, their absolute difference, their elementwise product, and forward edge attributes.
5. Assemble rectangular logits and run the masked normalization described below. Expose unit-length learned embeddings for the contrastive loss.

This is **edge-aware mean message passing**, not a verified paper attention layer or the exact upstream GCN. Keep the method labelled `partial_reproduction`. No ground-truth labels enter `forward`.

| Output | Meaning |
| --- | --- |
| `logits [T,D]` | Raw scores for BCE. Invalid entries are zero placeholders; always respect `mask`. |
| `mask [T,D]` | Allowed pairs copied from the graph. |
| `pair_logits [K]`, `candidate_pairs [K,2]` | Raw candidate scores and their local row mapping. |
| `track_embeddings [T,H]`, `detection_embeddings [D,H]` | Unit-length learned vectors; H is hidden width, not the 1024 input feature width. |
| `scores [T,D]` | Soft normalized matching scores; forbidden pairs are zero. |
| `unmatched_track_scores [T]`, `unmatched_detection_scores [D]` | Scores for leaving each item unmatched. |
| `transport [T+1,D+1]` | Full normalized matrix including the unmatched row/column. |
| `marginal_error` | Maximum discrepancy from intended row/column totals after the finite normalization iterations. |

These outputs are **not discrete track IDs**. Randomly initialized scores are only useful for checking that code runs. Train before drawing accuracy conclusions.

### sinkhorn

`sinkhorn(logits, mask, iterations=None, unmatched_score=0.0)` uses stable log-space FP32 updates and a fixed unmatched score. It is differentiable, but has no learnable normalization parameters. The model uses unmatched score 0.0.

Add one unmatched row and column. Real rows/columns target mass 1; the extra row targets mass D and the extra column mass T. Both sides therefore total T+D, even when the real matrix is rectangular. Masked real pairs stay zero. Fully isolated rows/columns are assigned unmatched score 1 directly; empty sides and all-masked inputs work without NaNs.

The default is `SINKHORN_ITERATIONS` from config (currently 20). Normalization is approximate: inspect `marginal_error`, especially for extreme logits. A numerical check with logits ±1000 still had about 0.01 mass error after 100 iterations, despite finite values/gradients. Do not assume exact row sums in that case. Scores are not independent sigmoid probabilities; matching thresholds remain a later validation decision.

### Training targets and losses

`build_targets(track_labels, detection_labels)` accepts two dictionaries returned by `match_detections_to_gt` and returns a binary target matrix and supervision mask. Same valid positive identity → 1; different valid identities → 0. Ignored/unmatched identities are excluded. Apply exactly the same crop filtering/order as the graph, or match GT against the already-filtered boxes.

`compute_loss(output, targets, valid_mask)` returns `total`, `bce`, `contrastive`, `valid_pairs`, `positive_pairs`, and `negative_pairs`:

- BCE is computed stably from **raw logits**, without an extra sigmoid or Sinkhorn probability conversion.
- For unit-length learned embeddings with Euclidean distance `distance`, contrastive loss is `distance**2` for positives and `max(margin-distance, 0)**2` for negatives. Default margin is 1.
- Each term averages positive and negative examples separately, then equally weights whichever classes are present. This prevents numerous negatives from overwhelming the positives.
- Total = `BCE_WEIGHT * bce + CONTRASTIVE_WEIGHT * contrastive`. Defaults are 1 and 1; optional function arguments override these and the margin.
- Supervision intersects the valid-label mask and candidate mask. No supervised pairs returns a differentiable zero with `valid_pairs=0`; skip that batch in training.

These explicit loss/normalization choices are development decisions, not verified paper equations. The current losses train the raw scorer and embeddings; they do not supervise the Sinkhorn output directly. There is no learnable unmatched parameter left disconnected from the loss.

### Minimal usage

```python
from Utils import config
from model import AssociationModel, build_targets, compute_loss

model = AssociationModel().to(config.DEVICE)
output = model(graph)  # graph from the two-frame example above
print(output["scores"].shape)  # [13, 13] for that example
print(output["unmatched_track_scores"])

# For training: obtain labels for the SAME filtered box rows first.
# track_labels = match_detections_to_gt(filtered_track_boxes, gt[1])
# detection_labels = match_detections_to_gt(filtered_detection_boxes, gt[2])
# targets, valid_pairs = build_targets(track_labels, detection_labels)
# losses = compute_loss(output, targets, valid_pairs)
# losses["total"].backward()
```

Use `model.eval()` with `torch.no_grad()` for inference; use `model.train()` and optimizer updates for training. [train.py](07-training.md) now supplies the epoch loop, validation, and dataset checkpoint saving.

**Checked in dragon-track:** rectangular/empty/masked Sinkhorn, finite gradients and known marginals, extreme-logit reporting, correct node/scorer gradients, row permutations, hand-calculated loss values, single-class/empty supervision, state save/reload, and tiny synthetic learning (loss about 0.737 → 0.000192; all 3 identities selected). Actual MOT17 frames 1→2 passed CPU/GPU comparison and an RTX 4060 forward/backward step with 81 valid pairs (9 positive, 72 negative). Empty GPU graphs also passed. This is implementation verification, not a trained dataset result; no dataset checkpoint or permanent test files were created.
