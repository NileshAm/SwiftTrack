"""Masked log-space Sinkhorn with learnable scaling and unmatched assignment.

Paper equations 5/6 are extended to rectangular matrices with dustbins. Scaling
changes marginal sums: marginal_error measures deviation, not exact convergence.
The paper does not specify rectangular unmatched handling. We retain compatible
dustbin masses from the released implementation, explicitly documented as an
extension to Eqs. 5/6. Scale factors are positive learned scalars initialized to 1.
"""

import math
import torch
from torch import nn
from Utils import config


def sinkhorn(logits, mask, iterations=None, unmatched_score=0.0, *,
             row_scale=None, col_scale=None, epsilon=1e-6, col_epsilon=None):
    """Normalize rectangular scores with one extra unmatched row and column.

    Real rows/columns have target mass 1; the extra row has mass D and the
    extra column mass T. Thus rectangular T-by-D inputs have compatible totals.
    Work in log space and FP32. The last row/column are unmatched options, not
    real people. Results are soft assignments, not discrete one-to-one matches.

    Returns scores [T,D], unmatched_track_scores [T], unmatched_detection_scores
    [D], transport [T+1,D+1], and marginal_error (maximum row/column mass error).
    Normalization is approximate after the configured number of iterations.
    Fully isolated rows/columns go directly to unmatched with score 1.
    """
    col_epsilon = epsilon if col_epsilon is None else col_epsilon
    if not all(math.isfinite(x) and x > 0 for x in (epsilon, col_epsilon)):
        raise ValueError("Sinkhorn stabilizers must be finite and positive")
    iterations = config.SINKHORN_ITERATIONS if iterations is None else iterations
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if not math.isfinite(float(torch.as_tensor(unmatched_score).detach())):
        raise ValueError("unmatched_score must be finite")
    logits = logits.float()
    mask = torch.as_tensor(mask, device=logits.device)
    if logits.ndim != 2 or mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("logits and boolean mask must have shape [T,D]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite; represent invalid pairs with mask=False")
    t, d = logits.shape
    # A connected zero keeps backward valid even when every pair is absent.
    transport = logits.new_zeros((t+1, d+1)) + logits.sum() * 0
    active_rows = mask.any(dim=1)
    active_cols = mask.any(dim=0)
    transport[:t, d] = (~active_rows).to(logits.dtype)
    transport[t, :d] = (~active_cols).to(logits.dtype)
    rows = active_rows.nonzero(as_tuple=True)[0]
    cols = active_cols.nonzero(as_tuple=True)[0]
    r, c = len(rows), len(cols)
    if r and c:
        # Solve only active rows/columns; isolated entries were fixed above.
        real = logits[rows[:, None], cols[None, :]]
        allowed = mask[rows[:, None], cols[None, :]]
        real = real.masked_fill(~allowed, -torch.inf)
        top = torch.cat((real, torch.as_tensor(unmatched_score, device=real.device).expand(r, 1)), dim=1)
        log_kernel = torch.cat((top, torch.as_tensor(unmatched_score, device=real.device).expand(1, c+1)), dim=0)
        log_row_mass = torch.cat((real.new_zeros(r), real.new_tensor([math.log(c)])))
        log_col_mass = torch.cat((real.new_zeros(c), real.new_tensor([math.log(r)])))
        # Equations 5/6 in log space, generalized to rectangular dustbin masses.
        # Unit scales recover ordinary dustbin Sinkhorn (up to the stabilizers).
        log_values = log_kernel
        for _ in range(iterations):
            row_sum = torch.logsumexp(log_values, dim=1, keepdim=True)
            if row_scale is not None:
                row_sum = torch.logaddexp(row_sum + row_scale.log(),
                                         row_sum.new_tensor(math.log(epsilon)))
            log_values = log_values + log_row_mass[:, None] - row_sum
            col_sum = torch.logsumexp(log_values, dim=0, keepdim=True)
            if col_scale is not None:
                col_sum = torch.logaddexp(col_sum + col_scale.log(),
                                         col_sum.new_tensor(math.log(col_epsilon)))
            log_values = log_values + log_col_mass[None, :] - col_sum
        normalized = log_values.exp()
        transport[rows[:, None], cols[None, :]] = normalized[:r, :c]
        transport[rows, d] = normalized[:r, c]
        transport[t, cols] = normalized[r, :c]
        transport[t, d] = normalized[r, c]

    expected_rows = torch.cat((logits.new_ones(t), logits.new_tensor([float(d)])))
    expected_cols = torch.cat((logits.new_ones(d), logits.new_tensor([float(t)])))
    error = torch.maximum((transport.sum(1) - expected_rows).abs().max(),
                          (transport.sum(0) - expected_cols).abs().max()).detach()
    return {
        "scores": transport[:t, :d],
        "unmatched_track_scores": transport[:t, d],
        "unmatched_detection_scores": transport[t, :d],
        "transport": transport,
        "marginal_error": error,
    }


class LearnableSinkhorn(nn.Module):
    """Paper row/column scalars; no learned temperature or unmatched score.

    Exp parameterization keeps denominators positive without imposing theta>=1.
    Input exp(5*cosine) and slack exp(-1) follow the released scorer convention;
    their constants and the stabilizers are explicit configurable assumptions.
    """
    def __init__(self):
        super().__init__()
        self.log_row_scale = nn.Parameter(torch.tensor(0.0))
        self.log_col_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, cosine, mask):
        return sinkhorn(cosine * config.SINKHORN_SCORE_SCALE, mask,
                        unmatched_score=config.SINKHORN_UNMATCHED_LOGIT,
                        row_scale=self.log_row_scale.exp(), col_scale=self.log_col_scale.exp(),
                        epsilon=config.SINKHORN_ROW_EPSILON,
                        col_epsilon=config.SINKHORN_COL_EPSILON)
