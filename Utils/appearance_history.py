"""Bounded, confidence-weighted appearance memory for one tracked person.

Keeps normalized CPU features from reliable matched observations only.
See Documentation/Implementation/15-motion-and-history.md for the contract.
"""

from collections import deque
import math

import torch
from torch.nn import functional as F


class AppearanceHistory:
    """Average up to max_size accepted observations, then normalize again.

    Low-confidence matches can maintain a track without changing its memory.
    The caller retains the initial feature until a reliable observation arrives.
    """

    def __init__(self, max_size=30, min_score=0.5):
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError("Appearance history size must be a positive integer")
        if not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("Appearance history minimum score must be between 0 and 1")
        self.samples = deque(maxlen=max_size)
        self.min_score = min_score

    def update(self, feature, score):
        """Store a detached copy if reliable and nonzero; return whether stored."""
        feature = torch.as_tensor(feature).detach().to(device="cpu", dtype=torch.float32)
        if feature.ndim != 1 or not feature.numel() or not torch.isfinite(feature).all():
            raise ValueError("Appearance must be a finite nonempty feature vector")
        if self.samples and feature.shape != self.samples[0][0].shape:
            raise ValueError("Appearance feature width changed within one track")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Appearance confidence must be finite")
        if score < self.min_score or feature.norm().item() <= 1e-12:
            return False
        # Confidence is a weight, not an assumed calibrated probability.
        self.samples.append((F.normalize(feature, dim=0).clone(), max(1e-6, min(score, 1.0))))
        return True

    def feature(self):
        """Return a normalized CPU prototype, or None before any accepted sample."""
        if not self.samples:
            return None
        features = torch.stack([feature for feature, _ in self.samples])
        weights = features.new_tensor([weight for _, weight in self.samples])
        average = (features * weights[:, None]).sum(0) / weights.sum()
        if average.norm().item() <= 1e-12:
            return self.samples[-1][0].clone()
        return F.normalize(average, dim=0)
