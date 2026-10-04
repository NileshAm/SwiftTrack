"""Build one track/detection graph using ordinary CPU PyTorch tensors.

Track nodes come first, then detection nodes. Every allowed pair has two
directed message edges, but only one candidate entry to score for assignment.
This is an explicit development graph, not a verified paper reproduction.
"""

import torch
from torch.nn import functional as F

from Utils.data import pairwise_iou
from Utils import config


# At receiving node i, displacement is position_i - position_j (neighbor j).
EDGE_FEATURE_NAMES = ("dx", "dy", "distance", "iou", "cosine")


def _matrix(values, name, columns=None, *, detach=True):
    """Convert inputs to float32 on CPU and catch misaligned/invalid arrays."""
    values = torch.as_tensor(values, dtype=torch.float32, device="cpu")
    if detach:
        values = values.detach()
    if columns is not None and values.shape == (0,):
        values = values.reshape(0, columns)
    if values.ndim != 2 or (columns is not None and values.shape[1] != columns):
        shape = f"[N,{columns}]" if columns is not None else "[N,F]"
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")
    return values


def build_graph(track_boxes, detection_boxes, track_features, detection_features,
                image_size, candidate_mask=None):
    """Build a graph dictionary from two sets of boxes and appearance features.

    Boxes: zero-origin xyxy arrays [T,4] and [D,4]. Features: [T,F] and [D,F],
    in the corresponding box order. Pass image_size=(height, width). Filter
    invalid crops using extract_features()'s mask before calling this function.

    candidate_mask: optional boolean [T,D]; None starts with every pair. The configured spatial gate is always applied. A supplied
    mask must come from an explicitly chosen rule, never ground-truth IDs.

    Returns CPU tensors ready to move to the association model's device:
      node_features: [T+D,F+4], appearance followed by normalized xywh.
      edge_index: [2,2K], source/destination node indices.
      edge_attr: [2K,5], columns described by EDGE_FEATURE_NAMES.
      candidate_pairs: [K,2], local (track row, detection row), track-major.
      edge_pair_index: [2K], association candidate for each message edge.
      candidate_mask: [T,D], node_type: [T+D] (0 track, 1 detection).
      num_tracks / num_detections: Python integers.

    K is the number of allowed pairs. Forward edges occupy the first K columns
    and reverse edges the next K. No same-type edges or self-loops are added.
    Empty inputs must keep their feature width, e.g. torch.empty((0, 1024)).
    Ground-truth labels and persistent track IDs stay outside this graph.
    """
    track_boxes = _matrix(track_boxes, "track_boxes", 4)
    detection_boxes = _matrix(detection_boxes, "detection_boxes", 4)
    track_features = _matrix(track_features, "track_features", detach=False)
    detection_features = _matrix(detection_features, "detection_features", detach=False)
    num_tracks, num_detections = len(track_boxes), len(detection_boxes)
    if len(track_features) != num_tracks or len(detection_features) != num_detections:
        raise ValueError("Each box must have exactly one feature row; filter all inputs together")
    if track_features.shape[1] == 0 or track_features.shape[1] != detection_features.shape[1]:
        raise ValueError("Track and detection features must have the same nonzero width")

    size = torch.as_tensor(image_size, dtype=torch.float32, device="cpu")
    if size.shape != (2,) or not torch.isfinite(size).all() or (size <= 0).any():
        raise ValueError("image_size must be positive (height, width)")
    height, width = size.unbind()
    boxes = torch.cat((track_boxes, detection_boxes), dim=0)
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("Boxes must have positive width/height in xyxy order")

    # Do not clamp: partially outside boxes retain their original geometry.
    scale = torch.stack((width, height, width, height))
    normalized_boxes = torch.cat((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]), dim=1) / scale
    appearance = torch.cat((track_features, detection_features), dim=0)
    node_features = torch.cat((appearance, normalized_boxes), dim=1)

    if candidate_mask is None:
        candidate_mask = torch.ones((num_tracks, num_detections), dtype=torch.bool)
    else:
        candidate_mask = torch.as_tensor(candidate_mask, device="cpu")
        if candidate_mask.dtype != torch.bool or candidate_mask.shape != (num_tracks, num_detections):
            raise ValueError("candidate_mask must be boolean with shape [T,D]")
        candidate_mask = candidate_mask.clone()
    limit = config.ASSOCIATION_DISTANCE_PX
    if limit is not None:
        if not isinstance(limit, (float, int)) or not 0 < limit < float("inf"):
            raise ValueError("ASSOCIATION_DISTANCE_PX must be positive or None")
        # Use top-left positions from Eq. 1; the gate uses original pixels.
        distance = torch.linalg.vector_norm(track_boxes[:, None, :2] - detection_boxes[None, :, :2], dim=2)
        candidate_mask &= distance <= limit
    pairs = candidate_mask.nonzero(as_tuple=False)
    track_rows, detection_rows = pairs.unbind(dim=1)
    source = track_rows
    destination = num_tracks + detection_rows
    forward_edges = torch.stack((source, destination))
    edge_index = torch.cat((forward_edges, forward_edges.flip(0)), dim=1)

    displacement = normalized_boxes[destination, :2] - normalized_boxes[source, :2]
    distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
    overlaps = torch.from_numpy(pairwise_iou(track_boxes.numpy(), detection_boxes.numpy()))
    normalized_appearance = F.normalize(appearance, p=2, dim=1, eps=1e-12)
    similarities = (normalized_appearance[:num_tracks]
                    @ normalized_appearance[num_tracks:].T).clamp(-1, 1)
    forward_attr = torch.cat((
        displacement, distance,
        overlaps[track_rows, detection_rows, None],
        similarities[track_rows, detection_rows, None],
    ), dim=1)
    # Reverse displacement changes sign; distance/IoU/cosine are symmetric.
    reverse_attr = forward_attr * torch.tensor([-1, -1, 1, 1, 1])
    edge_attr = torch.cat((forward_attr, reverse_attr), dim=0)
    if not torch.isfinite(node_features).all() or not torch.isfinite(edge_attr).all():
        raise ValueError("Graph features overflowed; check coordinate and feature magnitudes")

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "candidate_pairs": pairs,
        "edge_pair_index": torch.arange(len(pairs)).repeat(2),
        "candidate_mask": candidate_mask,
        "node_type": torch.cat((torch.zeros(num_tracks, dtype=torch.long),
                                torch.ones(num_detections, dtype=torch.long))),
        "num_tracks": num_tracks,
        "num_detections": num_detections,
    }
