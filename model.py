"""Paper-equation graph attention, cosine association, and learnable Sinkhorn.

See Documentation/Implementation/17-paper-alignment.md for explicit defaults
and remaining unspecified details. Earlier model checkpoints are incompatible.
"""
import torch
from torch import nn
from torch.nn import functional as F
from Utils import config
from Utils.relationships import RelationshipNetwork
from Utils.attention import GraphAttention
from Utils.sinkhorn import LearnableSinkhorn

MODEL_VERSION = "dragontrack_fixed_marginals_unit_embeddings_v4"


class AssociationModel(nn.Module):
    def __init__(self, feature_dim=None, hidden_dims=None):
        super().__init__()
        feature_dim = config.FEATURE_DIM if feature_dim is None else feature_dim
        hidden_dims = config.GRAPH_HIDDEN_DIMS if hidden_dims is None else hidden_dims
        if len(hidden_dims) != 2 or any(not isinstance(n, int) or n <= 0 for n in (feature_dim, *hidden_dims)):
            raise ValueError("Require a positive feature dimension and two graph widths")
        self.input_dim = feature_dim + 4
        self.relationships = RelationshipNetwork(feature_dim)
        self.attention = nn.ModuleList([
            GraphAttention(self.input_dim, hidden_dims[0]),
            GraphAttention(hidden_dims[0], hidden_dims[1])])
        self.matcher = LearnableSinkhorn()

    def forward(self, graph):
        device = next(self.parameters()).device
        nodes = graph["node_features"].to(device)
        edges = graph["edge_index"].to(device)
        attributes = graph["edge_attr"].to(device)
        pairs = graph["candidate_pairs"].to(device)
        mask = graph["candidate_mask"].to(device)
        t, d = graph["num_tracks"], graph["num_detections"]
        if nodes.shape != (t+d, self.input_dim):
            raise ValueError("Node feature shape differs from the model")
        if edges.shape != (2, 2*len(pairs)) or attributes.shape != (2*len(pairs), 5):
            raise ValueError("Expected two directed edges and five raw attributes per candidate")
        if mask.shape != (t, d) or not torch.equal(mask.nonzero(), pairs):
            raise ValueError("Candidate pairs must follow mask order")
        relationships = self.relationships(nodes, edges, attributes)
        embeddings = nodes
        attention_weights = []
        for layer in self.attention:
            embeddings, weights = layer(embeddings, edges, relationships)
            attention_weights.append(weights)
        unit = F.normalize(embeddings, dim=1, eps=1e-8)
        rows, cols = pairs.unbind(dim=1)
        similarity = (unit[rows] * unit[t + cols]).sum(1).clamp(-1, 1)
        cosine = embeddings.new_zeros((t, d)).index_put((rows, cols), similarity)
        output = {"logits": cosine, "pair_logits": similarity, "mask": mask,
                  "candidate_pairs": pairs, "track_embeddings": unit[:t],
                  "detection_embeddings": unit[t:], "attention_weights": attention_weights,
                  "relationships": relationships}
        output.update(self.matcher(cosine, mask))
        return output
