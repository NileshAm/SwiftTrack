"""Identity targets and balanced hard/random negative selection for training."""

import torch
from Utils import config


def build_targets(track_labels, detection_labels):
    """Same valid GT identity is positive; different valid identities are negative.

    Unlabelled/ignored detections are excluded, not silently treated as people
    with a new identity. One-to-one detection/GT labeling is done by data.py.
    """
    def ids(labels):
        values = torch.as_tensor(labels["ids"], dtype=torch.long, device="cpu")
        valid = torch.as_tensor(labels["valid_mask"], dtype=torch.bool, device="cpu")
        if values.ndim != 1 or valid.shape != values.shape:
            raise ValueError("Each identity vector needs an aligned validity mask")
        return values, valid & (values > 0)
    a, av = ids(track_labels)
    b, bv = ids(detection_labels)
    valid = av[:, None] & bv[None, :]
    return ((a[:, None] == b[None, :]) & valid).float(), valid


def select_pairs(truth, difficulty, training):
    """Keep all positives and a bounded mix of hardest and random negatives.

    Validation keeps every pair and is deterministic. When no positive exists,
    keep up to NEGATIVE_RATIO negatives so disappearance-only graphs still train.
    Selection uses detached difficulty scores, never changes the inference graph.
    """
    selected = torch.ones_like(truth, dtype=torch.bool)
    if not training:
        return selected
    if config.NEGATIVE_RATIO <= 0 or not 0 <= config.HARD_NEGATIVE_FRACTION <= 1:
        raise ValueError("Require NEGATIVE_RATIO > 0 and HARD_NEGATIVE_FRACTION in [0,1]")
    positive = truth.bool()
    negatives = (~positive).nonzero(as_tuple=True)[0]
    count = min(len(negatives), int(config.NEGATIVE_RATIO * max(1, int(positive.sum()))))
    if count == len(negatives):
        return selected
    selected = positive.clone()
    ranked = negatives[torch.argsort(difficulty.detach()[negatives], descending=True, stable=True)]
    hard_count = int(count * config.HARD_NEGATIVE_FRACTION)
    selected[ranked[:hard_count]] = True
    remaining = ranked[hard_count:]
    random_rows = torch.randperm(len(remaining), device=truth.device)[:count-hard_count]
    selected[remaining[random_rows]] = True
    return selected


def within_frame_pairs(embeddings, identities):
    """Return distinct same-frame embedding pairs and targets; no self-pairs.

    Cross-frame positives come from the association graph. Same-frame positives
    exist only if genuine repeated observations of an ID are supplied; we do not
    manufacture positives by pairing an embedding with itself.
    """
    ids = torch.as_tensor(identities, device=embeddings.device)
    i, j = torch.triu_indices(len(ids), len(ids), offset=1, device=embeddings.device)
    valid = (ids[i] > 0) & (ids[j] > 0)
    i, j = i[valid], j[valid]
    distance = torch.linalg.vector_norm(embeddings[i] - embeddings[j], dim=1)
    return distance, (ids[i] == ids[j]).float()
