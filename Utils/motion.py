"""Small constant-velocity Kalman filter for tracking boxes (NumPy only).

State: center x/y, width/height, followed by their per-frame velocities.
Predictions supply graph geometry; measured detections still supply MOT output.
See Documentation/Implementation/15-motion-and-history.md for tuning and limits.
"""

import numpy as np


def _measurement(box):
    box = np.asarray(box, dtype=np.float64)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or np.any(box[2:] <= box[:2])):
        raise ValueError("Motion boxes must be finite, positive-size xyxy")
    return np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2]]


class MotionState:
    """One independent filter per ID; advance using original frame numbers."""

    def __init__(self, box, frame_id):
        measurement = _measurement(box)
        self.mean = np.r_[measurement, np.zeros(4)]
        self.frame_id = int(frame_id)
        scale = max(measurement[3], 1.0)
        self.covariance = np.diag(([scale / 10] * 4 + [scale / 16] * 4)) ** 2

    def predict(self, frame_id):
        """Advance once per elapsed frame; repeated calls at one frame are safe.

        Position noise is height/20 and velocity noise height/160 per frame.
        Uncertainty grows through missing frames. Never update with a prediction.
        Boxes can extend outside the image; clamp sizes, not center positions.
        """
        frame_id = int(frame_id)
        if frame_id < self.frame_id:
            raise ValueError("Motion cannot predict backwards in time")
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4)
        for _ in range(frame_id - self.frame_id):
            scale = max(self.mean[3], 1.0)
            noise = np.diag(([scale / 20] * 4 + [scale / 160] * 4)) ** 2
            self.mean = transition @ self.mean
            self.covariance = transition @ self.covariance @ transition.T + noise
            self._limit_size()
        self.frame_id = frame_id
        return self.box()

    def update(self, box, frame_id):
        """Correct the prediction with an associated detection at this frame."""
        measurement = _measurement(box)
        self.predict(frame_id)
        noise = np.eye(4) * (max(self.mean[3], 1.0) / 20) ** 2
        innovation_covariance = self.covariance[:4, :4] + noise
        gain = np.linalg.solve(innovation_covariance, self.covariance[:4, :]).T
        self.mean += gain @ (measurement - self.mean[:4])
        # Joseph form maintains covariance stability over long sequences.
        residual = np.eye(8)
        residual[:, :4] -= gain
        self.covariance = (residual @ self.covariance @ residual.T
                           + gain @ noise @ gain.T)
        self.covariance = (self.covariance + self.covariance.T) / 2
        self._limit_size()

    def _limit_size(self):
        for index in (2, 3):
            if self.mean[index] < 1.0:
                self.mean[index] = 1.0
                self.mean[index + 4] = max(0.0, self.mean[index + 4])

    def box(self):
        """Return a fresh float32 xyxy box, without changing filter state."""
        center, size = self.mean[:2], self.mean[2:4]
        return np.r_[center - size / 2, center + size / 2].astype(np.float32)
