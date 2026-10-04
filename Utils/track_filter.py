"""Observation-count confirmation and filtered MOT export.

The paper does not give confirmation counts. Defaults (3 observations in a
15-frame window) follow the released tracker; exact boundary handling is ours.
"""
from collections import deque
import csv
from Utils import config


def validate_confirmation():
    for name in ("TRACK_CONFIRM_MIN_HITS", "TRACK_CONFIRM_WINDOW"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.TRACK_CONFIRM_MIN_HITS > config.TRACK_CONFIRM_WINDOW:
        raise ValueError("Confirmation hit count cannot exceed its frame window")


def observe_confirmation(track, frame_id):
    """A frame counts once. Once confirmed, a track remains confirmed."""
    track.setdefault("first_seen", frame_id)
    hits = track.setdefault("hit_frames", deque())
    while hits and hits[0] < frame_id - config.TRACK_CONFIRM_WINDOW + 1:
        hits.popleft()
    if not hits or hits[-1] != frame_id:
        hits.append(frame_id)
    track["confirmed"] = track.get("confirmed", False) or len(hits) >= config.TRACK_CONFIRM_MIN_HITS


def keep_tentative(track, frame_id):
    """Unconfirmed tracks get one complete window in which to gather evidence."""
    return track.get("confirmed", False) or frame_id - track["first_seen"] < config.TRACK_CONFIRM_WINDOW


def export_confirmed(raw_path, output_path, confirmed_ids):
    """Stream chronological rows, removing all observations of unconfirmed IDs.

    Earlier observations of eventually confirmed IDs are retained. This is an
    offline output filter, not an assertion that confirmation was known then.
    Videos are rendered from this final file so they agree with evaluation.
    """
    kept = 0
    with raw_path.open(newline="") as source, output_path.open("x", newline="") as target:
        writer = csv.writer(target)
        for row in csv.reader(source):
            if int(row[1]) in confirmed_ids:
                writer.writerow(row)
                kept += 1
    return kept
