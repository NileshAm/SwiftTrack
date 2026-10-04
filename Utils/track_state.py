"""Track lifecycle shared by chronological training and inference.

Motion and appearance are updated only from accepted observed detections.
Association decisions are supplied by the caller; this module never reads GT.
"""

import numpy as np
import torch

from Utils import config
from Utils.motion import MotionState
from Utils.appearance_history import AppearanceHistory
from Utils.track_filter import validate_confirmation, observe_confirmation, keep_tentative


def tracker_settings():
    """Validate and record shared track-state settings."""
    if (isinstance(config.MAX_MISSED_FRAMES, bool)
            or not isinstance(config.MAX_MISSED_FRAMES, int) or config.MAX_MISSED_FRAMES < 0):
        raise ValueError("MAX_MISSED_FRAMES must be a nonnegative integer")
    if not np.isfinite(config.NEW_TRACK_MIN_SCORE):
        raise ValueError("NEW_TRACK_MIN_SCORE must be finite")
    validate_confirmation()
    for name in ("USE_MOTION_PREDICTION", "USE_APPEARANCE_HISTORY"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be True or False")
    AppearanceHistory(config.APPEARANCE_HISTORY_SIZE, config.APPEARANCE_HISTORY_MIN_SCORE)
    return {"motion_prediction": config.USE_MOTION_PREDICTION,
            "motion_model": "constant_velocity_xywh_kalman_v1",
            "appearance_history": config.USE_APPEARANCE_HISTORY,
            "appearance_method": "bounded_confidence_weighted_mean_v1",
            "appearance_history_size": config.APPEARANCE_HISTORY_SIZE,
            "appearance_history_min_score": config.APPEARANCE_HISTORY_MIN_SCORE,
            "variant": config.TRACKING_VARIANT,
            "confirmation_min_hits": config.TRACK_CONFIRM_MIN_HITS,
            "confirmation_window": config.TRACK_CONFIRM_WINDOW,
            "association_distance_px": config.ASSOCIATION_DISTANCE_PX,
            "output_boxes": "observed_detections_only"}


def predict_track_boxes(tracks, frame_id):
    """Advance live filters to this frame, including frames with no detections."""
    boxes = []
    for track in tracks:
        if config.USE_MOTION_PREDICTION:
            if "motion" not in track:
                track["motion"] = MotionState(track["box"], track["last_seen"])
            boxes.append(track["motion"].predict(frame_id))
        else:
            boxes.append(track["box"])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def update_tracks(tracks, matches, unmatched_tracks, unmatched_detections,
                  frame, features, next_id):
    """Update tracks in place; return survivors, next unused ID, observations.

    Only matched detections and eligible new births are visible observations.
    Ages use original frame IDs. IDs increase monotonically within a sequence.
    """
    frame_id = frame["frame_id"]
    observations = []

    def observe(track, row):
        box = frame["boxes"][row].copy()
        feature = features[row].detach().cpu().clone()
        score = float(frame["scores"][row])
        if config.USE_MOTION_PREDICTION:
            if "motion" in track:
                track["motion"].update(box, frame_id)
            else:
                track["motion"] = MotionState(box, frame_id)
        if config.USE_APPEARANCE_HISTORY:
            if "appearance_history" not in track:
                track["appearance_history"] = AppearanceHistory(
                    config.APPEARANCE_HISTORY_SIZE, config.APPEARANCE_HISTORY_MIN_SCORE)
            history = track["appearance_history"]
            history.update(feature, score)
            prototype = history.feature()
            feature = prototype if prototype is not None else track.get("feature", feature)
        track.update(box=box, feature=feature, score=score, last_seen=frame_id, missed=0)
        observe_confirmation(track, frame_id)
        observations.append({"id": track["id"], "confirmed": track["confirmed"], "box": track["box"].copy(), "score": track["score"]})

    for track_row, detection_row in matches:
        observe(tracks[track_row], detection_row)
    for row in unmatched_tracks:
        tracks[row]["missed"] = frame_id - tracks[row]["last_seen"]
    survivors = [track for track in tracks if track["missed"] <= config.MAX_MISSED_FRAMES and keep_tentative(track, frame_id)]
    for row in unmatched_detections:
        if frame["scores"][row] >= config.NEW_TRACK_MIN_SCORE:
            track = {"id": next_id}
            observe(track, row)
            survivors.append(track)
            next_id += 1
    return survivors, next_id, observations


def expire_tracks(tracks, frame_id):
    """Allow a match at gap MAX_MISSED_FRAMES+1, expire older observations."""
    return [track for track in tracks
            if frame_id - track["last_seen"] - 1 <= config.MAX_MISSED_FRAMES
            and keep_tentative(track, frame_id)]


def state_settings():
    """State recipe saved with training weights, excluding output folder labels."""
    settings = tracker_settings()
    return {key: value for key, value in settings.items()
            if key not in ("variant", "output_boxes")} | {
                "max_frame_gap": config.MAX_MISSED_FRAMES + 1,
                "new_track_min_score": config.NEW_TRACK_MIN_SCORE}
