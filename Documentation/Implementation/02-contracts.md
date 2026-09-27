# Agree a few arrays and dictionary keys

[Back to TODO](README.md)

Spend a short discussion agreeing these values before writing connected functions. Use ordinary dictionaries and arrays/tensors; do not create a contracts file, dataclasses, or validators framework.

| Value | Simple representation |
| --- | --- |
| Frame | Dictionary with `frame_id`, image/path, `height`, `width`, `boxes`, and `scores`. |
| Boxes | `[D,4]` in internal zero-origin `(x1,y1,x2,y2)` pixels. Convert input origin once in `data.py`, and reverse it during MOT export. |
| Scores | `[D]`, preserving the supplied detection confidence. |
| Features | `[D,F]`, exactly the same row order as boxes. |
| Training labels | Valid person ID per detection plus a boolean valid/ignored mask; used only in training. |
| Graph | Dictionary with node features `[N,F_node]`, edge indices `[2,E]`, edge attributes `[E,F_edge]`, candidate pairs, and track/detection counts. |
| Model output | Dictionary containing `logits` `[T,D]`, `mask` `[T,D]`, and track/detection embeddings needed by the loss. |
| Track | Dictionary with ID, last box, score, last-seen frame, missed count, and latest feature. |

`T` is the number of tracks, `D` detections, `N=T+D` nodes. Store track nodes first, then detection nodes. A pair `(t,d)` uses graph nodes `t` and `T+d`. If message edges are bidirectional, score each track–detection pair only once. Start with one graph per batch; no graph-collation system is needed.

Read image shape as height then width. Preserve original frame IDs, including frames with no detections. Keep empty arrays correctly shaped. Ground-truth identities must never be used in tracking decisions.

**Quick check:** print shapes and a couple of row IDs through data → features → graph → model to confirm nothing is reordered. Change these keys only after telling the other developer.
