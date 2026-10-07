# Fixed matching normalization and cosine-aligned training

[Back to TODO](README.md) · [Config](../../Utils/config.py)

The old learned row/column scales could make real or unmatched scores exceed 1. Also, matching compared feature directions while contrastive training used raw feature distances. This change corrects those two issues. It deliberately departs from the earlier literal paper-equation normalization; it is not a claim that published accuracy has been reproduced.

## Run and checkpoint compatibility

`RUN_NAME="mot17_fixed_matching"` uses a new output folder. Pretrained DETR stays frozen, image batching stays enabled, and the video/detection selection is unchanged.

```bash
conda activate dragon-track
python train.py
python tracking.py
python evaluate.py
```

The matcher parameters and embedding contract changed, so old checkpoints are rejected using model version `dragontrack_fixed_marginals_unit_embeddings_v4`. Previous runs remain intact. `model.pt` still saves the lowest-validation-objective checkpoint; `last.pt` still saves each epoch. Selecting by tracking metrics is separate work.

## Matching — `Utils/sinkhorn.py`

`LearnableSinkhorn` now has one learned `unmatched_logit`, initialized from `SINKHORN_UNMATCHED_LOGIT=-1`. The graph supplies learnable cosine scores multiplied by the fixed `SINKHORN_SCORE_SCALE=5`. There are no learned row/column mass multipliers.

`sinkhorn(logits, mask, iterations, unmatched_score)` uses 50 alternating log-space normalizations by default. For T tracks and D detections:

- Every real row and column has total mass 1.
- The extra unmatched row has mass D; the extra unmatched column has mass T.
- Forbidden edges have exactly zero score. Isolated items are fully unmatched.
- Real and unmatched matching options lie in `[0,1]`, up to floating-point rounding. The bottom-right cell is bookkeeping mass and may exceed 1; it is never an assignment option.

Finite iterations can leave residual error. A final feasibility correction only shrinks overflowing real row/column sums. It then sets each item's unmatched score to its remaining mass and the bottom-right cell to the sum of real assignments. This restores the required totals without introducing forbidden edges. For converged inputs, it changes the solution only by rounding/residual error; it does not promise exact optimal transport for unconverged extreme inputs.

`normalization_error` measures the residual **before** that correction. `marginal_error` measures the final totals. Both maxima are saved in training history and tracking summaries. A tiny final error confirms valid normalization, not accurate identities.

The fixed-mass/unmatched design is informed by [SuperGlue's official implementation](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py). Our masked/isolated handling and final feasibility correction are local implementation choices.

## Embeddings and losses — `model.py`, `Utils/losses.py`

`AssociationModel.forward` exposes the same L2-normalized embeddings it uses for cosine matching. `compute_loss` also normalizes defensively so both same-frame and cross-frame contrastive pairs use feature direction consistently. For nonzero unit vectors, squared Euclidean distance is `2 - 2*cosine_similarity`. Multiplying an embedding by a positive scalar can no longer improve contrastive separation without changing matching similarity. Isolated zero embeddings stay zero; this change does not redesign graph attention.

WBCE rejects nonfinite or out-of-range matching probabilities instead of silently hiding a normalization failure through clipping. A small boundary clamp remains solely for numerical stability at log(0).

The optimized objective remains `0.5*WBCE_sum + 0.5*contrastive_sum`, averaged over supervised graphs within a batch. Class weights, margin, and negative mining are unchanged. This separates the requested feature-distance repair from a change in objective weighting.

## Reading the new loss diagnostics

The existing `total`, `bce`, and `contrastive` history columns still report mean graph sums. They can remain large and are not comparable with a per-pair average.

New diagnostics appear in progress bars, epoch summaries, CSV/checkpoint metrics, and `plots/training/losses_per_pair.png`:

| Column | Calculation across the entire epoch |
| --- | --- |
| `bce_per_weight` | Total weighted BCE / total sample weight |
| `contrastive_per_pair` | Total contrastive loss / total selected contrastive pair count |
| `bce_weight_sum` | Denominator of weighted BCE, including positive class weight 10 |
| `contrastive_pairs` | Same-frame plus cross-frame selected pair count |
| `contrastive_positive_pairs`, `contrastive_negative_pairs` | Breakdown of that count |

Zero eligible pairs give a zero diagnostic with a zero count. Ratios are computed from epoch sums, not averages of per-frame ratios. Training still samples negatives while validation uses all eligible negatives, so per-pair reporting removes count scaling but does not make their pair populations identical. Reporting a smaller number is not evidence of better tracking.

## Verification and remaining work

- [x] Fixed totals and bounded real/unmatched scores for empty, isolated, masked, rectangular, and extreme-score inputs.
- [x] Agreement with a converged direct-arithmetic reference; finite gradients and a trainable unmatched preference.
- [x] Known high-confidence matching example resolves correctly through Hungarian assignment.
- [x] Contrastive loss is invariant to positive rescaling of individual embeddings; same-frame and cross-frame terms are covered.
- [x] Short GPU train/validate, frozen-encoder preservation, checkpoint reload, tracking, video, TrackEval, and per-pair plots. Maximum final marginal error in that check was about `4e-6`.
- [ ] Train and evaluate full videos before claiming accuracy improvement.
- [ ] Separately assess graph feature collapse and tune matching/confirmation on validation. Thresholds and confirmation were not changed in this repair.
