"""Past-only motion/history state for supervised chronological association.

GT identity is used only to assign loss labels and to associate observations
AFTER scoring. Features/boxes always come from detections, never GT boxes.
See Documentation/Implementation/16-state-training.md.
"""

import numpy as np
import torch

from Utils import config
from Utils.track_state import expire_tracks, predict_track_boxes, update_tracks


class TrainingTrackState:
    """One video/epoch's teacher-forced state; never shared across splits."""

    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.tracks = []
        self.next_id = 1
        self.last_update = None

    def inputs(self, frame_id):
        """Predict before seeing current detections; return arrays and labels.

        All tracks retain their own last-seen time and bounded appearance
        memory. Labels are returned separately and must not enter the graph.
        """
        if self.last_update is not None and frame_id <= self.last_update:
            raise ValueError("Training track state requires strictly increasing frames")
        self.tracks = expire_tracks(self.tracks, frame_id)
        boxes = predict_track_boxes(self.tracks, frame_id)
        features = (torch.stack([track["feature"] for track in self.tracks])
                    if self.tracks else torch.empty((0, self.feature_dim)))
        identities = np.array([track["supervision_id"] for track in self.tracks], dtype=np.int64)
        labels = {"ids": identities, "valid_mask": identities > 0}
        ages = np.array([frame_id - track["last_seen"] for track in self.tracks], dtype=np.int64)
        return boxes, features, labels, ages

    def update(self, frame, features, labels):
        """Ingest this frame only after its graph/loss has been consumed.

        Known IDs guide correct history updates (teacher forcing). Unknown
        detections can create tracks at the same birth threshold as inference,
        but unknown IDs are never assumed to match one another or supervised
        as known identities. They still compete in the association graph.
        """
        frame_id = frame["frame_id"]
        if self.last_update is not None and frame_id <= self.last_update:
            raise ValueError("Training track state requires strictly increasing frames")
        self.tracks = expire_tracks(self.tracks, frame_id)
        identities = np.where(labels["valid_mask"], labels["ids"], -1)
        if len(identities) != len(frame["boxes"]) or len(features) != len(identities):
            raise ValueError("Training detections, features and labels must align")
        known = identities[identities > 0]
        if len(np.unique(known)) != len(known):
            raise ValueError("A frame cannot contain multiple labelled detections of one identity")
        rows = {int(identity): row for row, identity in enumerate(identities) if identity > 0}
        matches = [(row, rows[track["supervision_id"]])
                   for row, track in enumerate(self.tracks)
                   if track["supervision_id"] in rows]
        matched_tracks = {row for row, _ in matches}
        matched_detections = {row for _, row in matches}
        unmatched_tracks = [row for row in range(len(self.tracks)) if row not in matched_tracks]
        unmatched_detections = [row for row in range(len(identities)) if row not in matched_detections]
        # New IDs follow the same detection order and confidence rule as inference.
        birth_labels = {}
        next_id = self.next_id
        for row in unmatched_detections:
            if frame["scores"][row] >= config.NEW_TRACK_MIN_SCORE:
                birth_labels[next_id] = int(identities[row])
                next_id += 1
        self.tracks, self.next_id, _ = update_tracks(
            self.tracks, matches, unmatched_tracks, unmatched_detections,
            frame, features, self.next_id)
        for track in self.tracks:
            if track["id"] in birth_labels:
                track["supervision_id"] = birth_labels[track["id"]]
        self.last_update = frame_id
