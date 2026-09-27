# graph.py and model.py — the association core

[Back to TODO](README.md) · [Shared arrays](02-contracts.md)

## graph.py

- **`pairwise_iou(boxes_a, boxes_b)`** returns `[A,B]` overlaps using full xyxy endpoints. Avoid division-by-zero/NaNs for degenerate boxes. Empty inputs return correctly shaped empty arrays.
- **`build_graph(track_boxes, detection_boxes, track_features, detection_features, image_size)`** creates the node features, edge indices/attributes, candidate pairs, and row mappings in one dictionary. Use it for training frame pairs and inference tracks. Normalize x by image width and y by height. Begin with the documented baseline candidate rule; do not add top-k pruning as an invisible speed change.

Node features have one row per person/track, not per edge. Combine appearance and geometry according to the selected upstream/verified method. When message edges are bidirectional, keep an explicit mapping to unique candidate pairs. Start with one frame-pair graph at a time.

**Quick check:** identical boxes give IoU 1; disjoint boxes give 0. Print a tiny graph's edges and confirm every endpoint is a real node and each candidate maps to the expected people.

## model.py

- **`AssociationModel.forward(graph)`** runs the selected graph layers and pair scorer. Return a dictionary with raw rectangular logits, the allowed-pair mask, and embeddings for training. Aggregate edge messages into their destination nodes. Fix the upstream node/edge confusion before tuning architecture.
- **`sinkhorn(logits, mask, ...)`** performs the selected numerically stable normalization in FP32, with explicit unmatched/slack handling if the formulation uses it. Reuse and repair the appropriate upstream function. Specify compatible row/column totals for rectangular matrices; do not demand every row and column sum to one when their counts differ. Handle fully masked/empty cases without NaNs. Keep learnable parameters in the model only if their formula is verified.
- **`compute_loss(output, labels, valid_mask)`** computes weighted BCE plus contrastive loss and returns total/BCE/contrastive values. Record the formula, weights, margin, and score domain in the run notes. Do not apply sigmoid twice or treat ignored identities as negatives. No valid pairs means a reported skipped batch or safe zero, not NaN.

Use a correct repaired graph path first. Add paper attention only when its equation is understood and time permits; an arbitrary attention layer is not a verified reproduction. Clearly label any method differences. Avoid creating separate backends, loss modules, or normalization modules.

**Quick check:** run a small graph through `forward`, print output shapes and both loss terms, then call backward. Confirm finite, nonzero gradients reach the intended trainable parameters. Check an empty/no-edge case. Train the tiny labelled sample repeatedly to see whether it learns.
