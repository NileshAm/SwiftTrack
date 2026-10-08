"""Edge-conditioned attention with a projected self-feature residual.

The residual is a local extension motivated by the before/after diagnostic.
See Documentation/Implementation/22-appearance-residual.md.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GraphAttention(nn.Module):
    """Preserve own features alongside ReLU of the mean attention message.

    A separate linear projection handles width changes in the self path. Add
    it AFTER message ReLU so signed self features are not clipped away. There
    are no self-loop candidate edges: isolated nodes retain their own projected
    features but their matching row/column is still forbidden and unmatched.
    """
    def __init__(self, input_dim, output_dim, edge_dim=2):
        super().__init__()
        self.project = nn.Linear(input_dim, output_dim, bias=False)
        self.residual = nn.Linear(input_dim, output_dim, bias=False)
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
        return self.residual(nodes) + F.relu(aggregate / degree[:, None]), weights
