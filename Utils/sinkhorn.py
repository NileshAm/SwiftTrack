"""Fixed-marginal, masked log-space Sinkhorn with a learned unmatched preference.

Real rows/columns have mass 1; the unmatched row/column have masses D/T.
Learning changes the input scores, never these normalization targets. A final
feasibility correction sends finite-iteration residuals to unmatched entries.
"""

import math
import torch
from torch import nn
from Utils import config


def sinkhorn(logits, mask, iterations=None, unmatched_score=0.0):
    """Return a feasible rectangular soft assignment, including unmatched scores.

    Fully isolated items are unmatched. Masked real pairs stay exactly zero.
    `normalization_error` measures Sinkhorn residual BEFORE feasibility correction;
    `marginal_error` measures the final fixed row/column totals. The unmatched
    bottom-right bookkeeping cell may exceed 1; actual matching options cannot.
    """
    iterations = config.SINKHORN_ITERATIONS if iterations is None else iterations
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    logits = logits.float()
    mask = torch.as_tensor(mask, device=logits.device)
    if logits.ndim != 2 or mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("logits and boolean mask must have shape [T,D]")
    unmatched = torch.as_tensor(unmatched_score, dtype=logits.dtype, device=logits.device)
    if unmatched.numel() != 1 or not torch.isfinite(unmatched).all():
        raise ValueError("unmatched_score must be a finite scalar")
    unmatched = unmatched.reshape(())
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite; represent invalid pairs with mask=False")
    t, d = logits.shape
    scores = logits * 0 + unmatched * 0  # Backward remains valid for empty/all-masked graphs.
    rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
    cols = mask.any(dim=0).nonzero(as_tuple=True)[0]
    r, c = len(rows), len(cols)
    normalization_error = logits.new_zeros(())
    if r and c:
        real = logits[rows[:, None], cols[None, :]]
        allowed = mask[rows[:, None], cols[None, :]]
        real = real.masked_fill(~allowed, -torch.inf)
        top = torch.cat((real, unmatched.expand(r, 1)), dim=1)
        log_values = torch.cat((top, unmatched.expand(1, c+1)), dim=0)
        row_mass = torch.cat((real.new_ones(r), real.new_tensor([float(c)])))
        col_mass = torch.cat((real.new_ones(c), real.new_tensor([float(r)])))
        for _ in range(iterations):
            log_values = log_values + row_mass.log()[:, None] - torch.logsumexp(log_values, dim=1, keepdim=True)
            log_values = log_values + col_mass.log()[None, :] - torch.logsumexp(log_values, dim=0, keepdim=True)
        normalized = log_values.exp()
        normalization_error = torch.maximum(
            (normalized.sum(1)-row_mass).abs().max(),
            (normalized.sum(0)-col_mass).abs().max()).detach()
        real_scores = normalized[:r, :c]
        # Finite iterations can leave a row slightly above 1. Shrink only
        # overflowing real rows/columns; place the remaining mass in dustbins.
        # This preserves forbidden zeros and guarantees nonnegative unmatched
        # options instead of clipping malformed probabilities inside the loss.
        real_scores = real_scores / real_scores.sum(1, keepdim=True).clamp_min(1)
        real_scores = real_scores / real_scores.sum(0, keepdim=True).clamp_min(1)
        scores = scores.index_put((rows[:, None], cols[None, :]), real_scores)
    unmatched_tracks = (1-scores.sum(1)).clamp_min(0)
    unmatched_detections = (1-scores.sum(0)).clamp_min(0)
    top = torch.cat((scores, unmatched_tracks[:, None]), dim=1)
    bottom = torch.cat((unmatched_detections, scores.sum().reshape(1)))
    transport = torch.cat((top, bottom[None, :]), dim=0)
    expected_rows = torch.cat((logits.new_ones(t), logits.new_tensor([float(d)])))
    expected_cols = torch.cat((logits.new_ones(d), logits.new_tensor([float(t)])))
    error = torch.maximum((transport.sum(1)-expected_rows).abs().max(),
                          (transport.sum(0)-expected_cols).abs().max()).detach()
    if not torch.isfinite(transport).all():
        raise RuntimeError("Sinkhorn produced nonfinite transport")
    return {"scores": scores, "unmatched_track_scores": unmatched_tracks,
            "unmatched_detection_scores": unmatched_detections, "transport": transport,
            "marginal_error": error, "normalization_error": normalization_error}


class LearnableSinkhorn(nn.Module):
    """Learn the unmatched logit; enforce fixed masses throughout normalization."""
    def __init__(self):
        super().__init__()
        if not math.isfinite(config.SINKHORN_UNMATCHED_LOGIT):
            raise ValueError("SINKHORN_UNMATCHED_LOGIT must be finite")
        self.unmatched_logit = nn.Parameter(torch.tensor(float(config.SINKHORN_UNMATCHED_LOGIT)))

    def forward(self, cosine, mask):
        if not math.isfinite(config.SINKHORN_SCORE_SCALE) or config.SINKHORN_SCORE_SCALE <= 0:
            raise ValueError("SINKHORN_SCORE_SCALE must be finite and positive")
        return sinkhorn(cosine * config.SINKHORN_SCORE_SCALE, mask,
                        unmatched_score=self.unmatched_logit)
