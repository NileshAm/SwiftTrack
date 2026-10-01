"""Separate appearance/geometry/position affinities for DragonTrack edges.

The paper leaves intermediate affinity widths unspecified. Scalar Linear+ReLU
maps and appearance/geometry fusion follow the released affinity modules;
position inputs follow paper Eq. 1 (dx, dy, Euclidean distance).
"""
import torch
from torch import nn


class RelationshipNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.appearance = nn.Sequential(nn.Linear(2 * feature_dim, 1), nn.ReLU())
        self.geometry = nn.Sequential(nn.Linear(8, 1), nn.ReLU())
        self.position = nn.Sequential(nn.Linear(3, 1), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(2, 1), nn.ReLU())

    def forward(self, nodes, edges, attributes):
        source, destination = edges
        appearance = self.appearance(torch.cat((nodes[source, :self.feature_dim],
                                                nodes[destination, :self.feature_dim]), dim=1))
        geometry = self.geometry(torch.cat((nodes[source, -4:], nodes[destination, -4:]), dim=1))
        affinity = self.fusion(torch.cat((appearance, geometry), dim=1))
        position = self.position(attributes[:, :3])
        return torch.cat((affinity, position), dim=1)
