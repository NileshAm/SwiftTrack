# Agree a few arrays and dictionary keys

[Back to TODO](README.md)

Spend a short discussion agreeing these values before writing connected functions. Use ordinary dictionaries and arrays/tensors; do not create a contracts file, dataclasses, or validators framework.

| Value | Simple representation |
| --- | --- |
| Frame | Dictionary with `frame_id`, image/path, `height`, `width`, `boxes`, and `scores`. |
| Boxes | `[D,4]` in internal zero-origin `(x1,y1,x2,y2)` pixels. Convert input origin once in `Utils/data.py`, and reverse it during MOT export. |
| Scores | `[D]`, preserving the supplied detection confidence. |
| Features | `extract_features` returns `(features, valid_mask)`: CPU tensors `[D,1024]` and boolean `[D]`, in box row order. Filter invalid rows from features, boxes, scores, row IDs, and training labels together before graph construction. |
| Training labels | Valid person ID per detection plus a boolean valid/ignored mask; used only in training. |
| Graph | CPU tensors: `node_features [T+D,F+4]`, `edge_index [2,2K]`, `edge_attr [2K,6]`, `candidate_pairs [K,2]`, `edge_pair_index [2K]`, `candidate_mask [T,D]`, `node_type [T+D]`, plus integer `num_tracks`/`num_detections`. [Exact layout](05-graph-and-model.md#exact-layout). |
| Model output | Dictionary containing `logits`, `mask`, normalized `scores` `[T,D]`; unit-length track/detection embeddings `[T,H]` / `[D,H]`; unmatched scores `[T]` / `[D]`; augmented `transport [T+1,D+1]`; candidate mapping and `marginal_error`. [Details](05-graph-and-model.md#modelpy). |
| Track | Dictionary with ID, last box, score, last-seen frame, missed count, and latest feature. |

`T` is the number of tracks, `D` detections, `N=T+D` nodes. Store track nodes first, then detection nodes. A pair `(t,d)` uses graph nodes `t` and `T+d`. If message edges are bidirectional, score each track–detection pair only once. Start with one graph per batch; no graph-collation system is needed.

Read image shape as height then width. Preserve original frame IDs, including frames with no detections. Keep empty arrays correctly shaped. Ground-truth identities must never be used in tracking decisions.

**Quick check:** print shapes and a couple of row IDs through data → features → graph → model to confirm nothing is reordered. Change these keys only after telling the other developer.
