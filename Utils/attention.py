"""DragonTrack equations 2–4: edge-conditioned attention and degree mean."""
import torch
from torch import nn
from torch.nn import functional as F


class GraphAttention(nn.Module):
    """Single-head softmax over incoming neighbors, then ReLU(sum / degree).

    No residual, LayerNorm, extra edge-value projection, or multi-head mixing.
    Isolated nodes return zero; their matching row/column is unmatched.
    """
    def __init__(self, input_dim, output_dim, edge_dim=2):
        super().__init__()
        self.project = nn.Linear(input_dim, output_dim, bias=False)
        self.attend = nn.Linear(2 * output_dim + edge_dim, 1, bias=False)

    def forward(self, nodes, edges, attributes):
        source, destination = edges
        projected = self.project(nodes)
        logits = F.leaky_relu(self.attend(torch.cat(
            (projected[destination], projected[source], attributes), dim=1)).squeeze(1),
            negative_slope=0.2)
        maxima = logits.new_full((len(nodes),), -torch.inf)
        maxima.scatter_reduce_(0, destination, logits.detach(), reduce="amax", include_self=True)
        weights = (logits - maxima[destination]).exp()
        denominator = weights.new_zeros(len(nodes)).index_add(0, destination, weights)
        weights = weights / denominator[destination].clamp_min(1e-12)
        aggregate = projected.new_zeros(projected.shape).index_add(
            0, destination, projected[source] * weights[:, None])
        degree = torch.bincount(destination, minlength=len(nodes)).clamp_min(1)
        return F.relu(aggregate / degree[:, None]), weights
