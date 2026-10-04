"""Weighted BCE and contrastive loss with matching-aligned unit embeddings.

WBCE supervises normalized matching scores, so Sinkhorn receives gradients
without an added assignment loss. Alpha, class weights and mining settings
are configurable because the paper does not provide their complete recipe.
"""
import math
import torch
from torch.nn import functional as F
from Utils import config
from Utils.pairs import select_pairs, within_frame_pairs


def compute_loss(output, targets, valid_mask, *, identity_labels=None, training=False,
                 margin=None):
    """Sum WBCE and unit-embedding contrastive terms; also return reporting counts.

    Unknown identities and spatially excluded edges never become false labels.
    Numerical clipping protects log(0), not invalid normalization. Pair/weight
    totals allow epoch-wide mean diagnostics without changing the summed objective.
    """
    alpha = config.LOSS_ALPHA
    margin = config.CONTRASTIVE_MARGIN if margin is None else margin
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("LOSS_ALPHA must be between 0 and 1")
    if not all(math.isfinite(x) and x > 0 for x in
               (margin, config.WBCE_POSITIVE_WEIGHT, config.WBCE_NEGATIVE_WEIGHT)):
        raise ValueError("Contrastive margin and BCE weights must be finite and positive")
    scores = output["scores"].float()
    if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1+1e-6).any():
        raise ValueError("Matching scores must be finite probabilities in [0,1]")
    targets = torch.as_tensor(targets, dtype=torch.float32, device=scores.device)
    valid_mask = torch.as_tensor(valid_mask, device=scores.device)
    if targets.shape != scores.shape or valid_mask.shape != scores.shape or valid_mask.dtype != torch.bool:
        raise ValueError("Targets and boolean validity mask must match scores")
    eligible = valid_mask & output["mask"]
    truth = targets[eligible]
    if not ((truth == 0) | (truth == 1)).all():
        raise ValueError("Targets must be binary")
    weights = torch.where(truth.bool(), config.WBCE_POSITIVE_WEIGHT, config.WBCE_NEGATIVE_WEIGHT)
    bce = (weights * F.binary_cross_entropy(scores[eligible].clamp(1e-6, 1-1e-6),
                                           truth, reduction="none")).sum()
    rows, cols = eligible.nonzero(as_tuple=True)
    # Also normalize here for external callers. Both cross-frame and same-frame
    # distances must use the same directions that cosine matching compares.
    track_embeddings = F.normalize(output["track_embeddings"].float(), dim=1, eps=1e-8)
    detection_embeddings = F.normalize(output["detection_embeddings"].float(), dim=1, eps=1e-8)
    distance = torch.linalg.vector_norm(track_embeddings[rows]-detection_embeddings[cols], dim=1)
    distances, truths = [distance], [truth]
    if identity_labels is not None:
        for embeddings, ids in zip((track_embeddings, detection_embeddings), identity_labels):
            d, t = within_frame_pairs(embeddings, ids)
            distances.append(d)
            truths.append(t)
    distance, pair_truth = torch.cat(distances), torch.cat(truths)
    chosen = select_pairs(pair_truth, -distance, training)
    distance, positive = distance[chosen], pair_truth[chosen].bool()
    contrastive = torch.where(positive, distance.square(), F.relu(margin-distance).square()).sum()
    return {"total": alpha*bce + (1-alpha)*contrastive,
            "bce": bce, "contrastive": contrastive,
            "assignment": scores.sum()*0,  # Legacy log column; not part of the objective.
            "bce_weight_sum": weights.sum().detach(),
            "contrastive_pairs": len(distance),
            "contrastive_positive_pairs": int(positive.sum()),
            "contrastive_negative_pairs": int((~positive).sum()),
            "valid_pairs": int(eligible.sum()), "positive_pairs": int(truth.sum()),
            "negative_pairs": int(len(truth)-truth.sum())}
