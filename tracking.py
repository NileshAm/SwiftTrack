"""Run `python tracking.py` to associate detections using a trained checkpoint.

Settings live in Utils/config.py. No ground-truth annotations are read here.
Tracks predict positions through gaps and retain reliable appearance history.
"""

import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch
from tqdm.auto import tqdm

from Utils import config
from Utils.data import load_sequence, read_image
from Utils.detections import detection_path, detection_settings
from Utils.features import load_encoder, extract_features
from Utils.graph import build_graph
from model import AssociationModel, MODEL_VERSION
from Utils.splits import validate_video_splits, split_manifest, check_checkpoint_split
from Utils.outputs import tracking_folder, tracking_context
from Utils.track_filter import export_confirmed
from Utils.track_state import (tracker_settings, state_settings, predict_track_boxes,
                               update_tracks, expire_tracks)


def match_tracks(scores, allowed_mask, unmatched_track_scores,
                 unmatched_detection_scores, match_threshold):
    """Hungarian assignment with an explicit unmatched choice for every item.

    Minimize negative log scores on an augmented (T+D)-square matrix. A real
    match replaces BOTH unmatched costs. Forbidden/below-threshold pairs are
    excluded before solving. Return matches, unmatched track rows, detection rows.
    """
    def array(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    scores = array(scores).astype(np.float64)
    mask = array(allowed_mask)
    ut = array(unmatched_track_scores).astype(np.float64)
    ud = array(unmatched_detection_scores).astype(np.float64)
    if scores.ndim != 2 or mask.shape != scores.shape or mask.dtype != np.bool_:
        raise ValueError("scores and boolean allowed_mask must have shape [T,D]")
    t, d = scores.shape
    if ut.shape != (t,) or ud.shape != (d,):
        raise ValueError("Unmatched score vectors must align with tracks/detections")
    if match_threshold is None or not 0 <= match_threshold <= 1:
        raise ValueError("MATCH_THRESHOLD must be between 0 and 1")
    if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1+1e-6).any()
           for a in (scores, ut, ud)):
        raise ValueError("Association scores must be finite probabilities in [0,1]")
    scores, ut, ud = (np.minimum(a, 1.0) for a in (scores, ut, ud))  # Roundoff only.
    if not t or not d:
        return [], list(range(t)), list(range(d))

    # Clip zeros only for logarithms, never to make forbidden pairs eligible.
    pair_cost = -np.log(np.maximum(scores, 1e-12))
    track_cost = -np.log(np.maximum(ut, 1e-12))
    detection_cost = -np.log(np.maximum(ud, 1e-12))
    allowed = (mask & (scores > 0) & (scores >= match_threshold)
               & (pair_cost < track_cost[:, None] + detection_cost[None, :]))
    costs = np.full((t+d, t+d), np.inf)
    costs[:t, :d] = np.where(allowed, pair_cost, np.inf)
    costs[np.arange(t), d+np.arange(t)] = track_cost
    costs[t+np.arange(d), np.arange(d)] = detection_cost
    costs[t:, d:] = 0  # Remaining dummy rows/columns can complete the assignment.
    rows, cols = linear_sum_assignment(costs)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if r < t and c < d]
    matched_tracks = {r for r, _ in matches}
    matched_detections = {c for _, c in matches}
    return (matches, [r for r in range(t) if r not in matched_tracks],
            [c for c in range(d) if c not in matched_detections])


def draw_tracks(image, observations):
    """Return an RGB image with observed boxes/IDs, using existing Pillow."""
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for observation in observations:
        identity = observation["id"]
        color = tuple(70 + (identity * multiplier) % 186 for multiplier in (37, 79, 113))
        x1, y1, x2, y2 = observation["box"]
        bounds = (max(0, float(x1)), max(0, float(y1)),
                  min(width-1, float(x2)), min(height-1, float(y2)))
        if bounds[2] >= bounds[0] and bounds[3] >= bounds[1]:
            draw.rectangle(bounds, outline=color, width=3)
            point = (bounds[0], max(0, bounds[1]-12))
            label = f"ID {identity}"
            draw.rectangle(draw.textbbox(point, label), fill=color)
            draw.text(point, label, fill="black")
    return np.asarray(canvas)


def make_evaluation_video(sequence=None, split=None, *, overall=None):
    """Make an MP4 from saved tracking results, without rerunning the model.

    Defaults to the first configured sequence and current TRACK_SPLIT. Pass
    split='val' for a full validation video, or 'test' for a full
    test video. Run tracking first. Returns the path and preserves empty frames.
    """
    import cv2

    split = config.TRACK_SPLIT if split is None else split
    choices = config.TEST_SEQUENCES if split == "test" else config.VAL_SEQUENCES
    if sequence is None and not choices:
        raise ValueError("No sequences configured for this split")
    sequence = choices[0] if sequence is None else sequence
    context = tracking_context(sequence, split)
    folder = context["output_dir"]
    predictions, summary_path = folder / "tracks.txt", folder / "summary.json"
    if not predictions.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Track {sequence} ({split}) first; expected {predictions}")
    sequence_path, frame_range = context["sequence_path"], context["frame_range"]
    summary = json.loads(summary_path.read_text())
    if (summary.get("complete") is not True or summary.get("split") != split
            or summary.get("sequence") != sequence or summary.get("frame_range") != list(frame_range)
            or summary.get("frames") != frame_range[1]-frame_range[0]+1
            or summary.get("dataset_root") != str(context["dataset_root"].resolve())
            or summary.get("dataset_split") != context["dataset_split"]
            or summary.get("data_split") != split_manifest()):
        raise ValueError("Saved tracking results do not cover the configured dataset/split")
    # Reuse the MOT reader for boxes/images; preserve prediction IDs separately.
    frames = load_sequence(sequence_path, predictions.resolve(), frame_range, box_origin=config.BOX_ORIGIN)
    identities = {frame_id: [] for frame_id in frames}
    with predictions.open(newline="") as file:
        for row in csv.reader(file):
            if not row:
                continue
            frame_id, identity = float(row[0]), float(row[1])
            if frame_id not in identities or not identity.is_integer() or identity <= 0:
                raise ValueError("Prediction has an out-of-split frame or invalid track ID")
            identities[int(frame_id)].append(int(identity))
    if sum(map(len, identities.values())) != summary.get("mot_rows"):
        raise ValueError("Prediction row count differs from the tracking summary")
    output = folder / "video.mp4"
    if output.exists():
        raise FileExistsError(f"Video already exists: {output}")
    first = next(iter(frames.values()))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             first["fps"], (first["width"], first["height"]))
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Could not create {output}")
        for frame_id, frame in tqdm(frames.items(), desc=f"Video {sequence} ({split})",
                                   unit="frame", dynamic_ncols=True, disable=not config.SHOW_PROGRESS,
                                   position=1 if overall is not None else 0, leave=overall is None):
            observations = [{"id": identity, "box": box} for identity, box in
                            zip(identities[frame_id], frame["boxes"], strict=True)]
            image = draw_tracks(read_image(frame), observations)
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if overall is not None:
                overall.update(1)
    finally:
        writer.release()
    tqdm.write(f"Video saved: {output}")
    return output


@torch.no_grad()
def run_sequence(sequence, model, encoder, split=None, *, overall=None):
    """Track one selected sequence, save MOT rows/preview, and report timing.

    Expire tracks before association if the gap already contains more than
    MAX_MISSED_FRAMES missing observations. After an unsuccessful match, age and
    expire again. A gap of exactly the allowed number can still reconnect.
    """
    settings = tracker_settings()
    context = tracking_context(sequence, split)
    output_dir = context["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tracks.txt"
    for name in ("tracks.txt", "summary.json", "video.mp4", "preview.jpg", ".tracks.pending.txt"):
        if (output_dir / name).exists():
            raise FileExistsError(f"{output_dir / name} exists. Choose a new output folder.")
    sequence_path, frame_range = context["sequence_path"], context["frame_range"]
    tqdm.write(f"{sequence} {context['split']}: frames {frame_range[0]}..{frame_range[1]}")
    resolved_detections = detection_path(sequence_path)
    frames = load_sequence(sequence_path, resolved_detections, frame_range,
                           box_origin=config.BOX_ORIGIN, min_score=config.DETECTION_MIN_SCORE)
    model.eval()
    encoder.eval()
    tracks, next_id, row_count, max_error = [], 1, 0, 0.0
    max_normalization_error = 0.0
    confirmed_ids = set()
    raw_path = output_dir / ".tracks.pending.txt"
    start = time.perf_counter()
    bar = tqdm(frames.values(), total=len(frames), desc=f"Track {sequence}",
               unit="frame", dynamic_ncols=True, disable=not config.SHOW_PROGRESS,
               position=1 if overall is not None else 0, leave=overall is None)
    try:
        with raw_path.open("x") as file:
            for frame in bar:
                frame_id = frame["frame_id"]
                image = read_image(frame)
                features, valid = extract_features(image, frame["boxes"], encoder, show_progress=False)
                keep = valid.numpy()
                current = dict(frame, boxes=frame["boxes"][keep], scores=frame["scores"][keep],
                               row_ids=frame["row_ids"][keep])
                features = features[valid]
                tracks = expire_tracks(tracks, frame_id)
                track_boxes = predict_track_boxes(tracks, frame_id)
                track_features = (torch.stack([track["feature"] for track in tracks]) if tracks
                                  else torch.empty((0, encoder.feature_dim)))
                graph = build_graph(track_boxes, current["boxes"], track_features, features,
                                    image_size=(frame["height"], frame["width"]))
                output = model(graph)
                error = output["marginal_error"].item()
                max_error = max(max_error, error)
                max_normalization_error = max(max_normalization_error, output["normalization_error"].item())
                matches, unmatched_tracks, unmatched_detections = match_tracks(
                    output["scores"], output["mask"], output["unmatched_track_scores"],
                    output["unmatched_detection_scores"], config.MATCH_THRESHOLD)
                previous_id = next_id
                tracks, next_id, observations = update_tracks(
                    tracks, matches, unmatched_tracks, unmatched_detections, current, features, next_id)
                for observation in observations:
                    if observation["confirmed"]:
                        confirmed_ids.add(observation["id"])
                    x1, y1, x2, y2 = observation["box"]
                    file.write(f"{frame_id},{observation['id']},{x1+config.BOX_ORIGIN:.3f},"
                               f"{y1+config.BOX_ORIGIN:.3f},{x2-x1:.3f},{y2-y1:.3f},"
                               f"{observation['score']:.6f},-1,-1,-1\n")
                row_count += len(observations)
                bar.set_postfix(frame=frame_id, detections=len(features), matched=len(matches),
                                new=next_id-previous_id, retained=len(tracks), confirmed=len(confirmed_ids),
                                norm_error=f"{error:.3g}", refresh=False)
                if overall is not None:
                    overall.update(1)
    finally:
        bar.close()
    raw_row_count = row_count
    row_count = export_confirmed(raw_path, output_path, confirmed_ids)
    raw_path.unlink()  # This run's temporary spool; never an existing user result.
    seconds = time.perf_counter() - start
    preview_path = output_dir / "preview.jpg"
    Image.fromarray(draw_tracks(image, [x for x in observations if x["id"] in confirmed_ids])).save(preview_path)
    summary = {
        "sequence": sequence, "frames": len(frames), "mot_rows": row_count,
        "raw_mot_rows": raw_row_count, "confirmed_ids": len(confirmed_ids),
        "output_filter": "offline_observation_count_confirmation",
        "detection_file": str(resolved_detections), "detections": detection_settings(),
        "ids_created": next_id-1, "seconds": seconds, "fps": len(frames)/seconds,
        "max_marginal_error": max_error, "max_normalization_error": max_normalization_error,
        "checkpoint": str(Path(config.CHECKPOINT_PATH).resolve()),
        "match_threshold": config.MATCH_THRESHOLD, "new_track_min_score": config.NEW_TRACK_MIN_SCORE,
        "max_missed_frames": config.MAX_MISSED_FRAMES,
        "lookback_frames": config.MAX_MISSED_FRAMES + 1, "frame_range": frame_range,
        "complete": True, "split": context["split"], "data_split": split_manifest(),
        "dataset_root": str(context["dataset_root"].resolve()), "dataset_split": context["dataset_split"],
        "sinkhorn_iterations": config.SINKHORN_ITERATIONS, "save_video": context["save_video"],
        "device": config.DEVICE, "method": "paper_inspired_attention", "model_version": MODEL_VERSION,
        "assignment": "negative_log_scores_with_track_and_detection_unmatched_costs",
        "tracker_settings": settings,
        "timing": "frame loop including image reads and output; excludes model/metadata loading, final preview and video",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    tqdm.write(f"{sequence}: {len(frames)} frames, {row_count} observations, {next_id-1} IDs; "
          f"{summary['fps']:.2f} FPS\nMOT output: {output_path}\nPreview: {preview_path}")
    if context["save_video"]:
        make_evaluation_video(sequence, split=context["split"], overall=overall)
    return summary


def main():
    """Load graph and appearance weights from one compatible checkpoint; track clips."""
    tracker_settings()
    if config.MATCH_THRESHOLD is None or not 0 <= config.MATCH_THRESHOLD <= 1:
        raise ValueError("Set MATCH_THRESHOLD between 0 and 1, then tune it on validation")
    if not isinstance(config.MAX_MISSED_FRAMES, int) or config.MAX_MISSED_FRAMES < 0:
        raise ValueError("MAX_MISSED_FRAMES must be a nonnegative integer")
    if not np.isfinite(config.NEW_TRACK_MIN_SCORE):
        raise ValueError("NEW_TRACK_MIN_SCORE must be finite")
    if config.TRACK_SPLIT != "val":
        raise ValueError("TRACK_SPLIT must be val; validation uses full VAL_SEQUENCES")
    validate_video_splits()
    if len(set(config.TEST_SEQUENCES)) != len(config.TEST_SEQUENCES):
        raise ValueError("TEST_SEQUENCES must contain distinct names")
    jobs = ([(sequence, config.TRACK_SPLIT) for sequence in config.VAL_SEQUENCES]
            if config.RUN_SPLIT_TRACKING else [])
    jobs += [(sequence, "test") for sequence in config.TEST_SEQUENCES]
    if not jobs:
        raise ValueError("Enable RUN_SPLIT_TRACKING or add TEST_SEQUENCES")
    total_frames = 0
    for sequence, split in jobs:
        context = tracking_context(sequence, split)
        first, last = context["frame_range"]
        total_frames += (last - first + 1) * (2 if context["save_video"] else 1)
        detection_path(context["sequence_path"])
        for name in ("tracks.txt", "summary.json", "video.mp4", "preview.jpg", ".tracks.pending.txt"):
            path = context["output_dir"] / name
            if path.exists():
                raise FileExistsError(f"{path} exists. Choose a new output folder.")
    if not Path(config.CHECKPOINT_PATH).is_file():
        raise FileNotFoundError(f"Missing {config.CHECKPOINT_PATH}. Run python train.py first.")
    tqdm.write(f"Loading checkpoint: {config.CHECKPOINT_PATH}")
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("This checkpoint uses an older graph/scoring architecture. Train a new RUN_NAME first.")
    check_checkpoint_split(checkpoint)
    temporal = checkpoint.get("temporal_sampling", {})
    if (temporal.get("protocol") not in ("historical_pair_window_v1", "chronological_track_state_v1")
            or temporal.get("max_frame_gap") != config.MAX_MISSED_FRAMES + 1):
        raise ValueError("Checkpoint was not trained with the current look-back window. "
                         "Train a fresh RUN_NAME with matching training/validation gaps.")
    if temporal.get("protocol") == "chronological_track_state_v1":
        if temporal.get("state_settings") != state_settings():
            tqdm.write("Tracking motion/history/lifetime settings differ from training. "
                       "This is a comparison run; restore the saved settings for matched behavior.")
    # Keep training-time feature/coordinate conventions; runtime device and
    # tracking thresholds may differ. Architecture comes from the checkpoint.
    for name in ("ENCODER_NAME", "USE_PRETRAINED_WEIGHTS", "FREEZE_ENCODER", "FEATURE_DIM",
                 "BOX_ORIGIN", "DETECTION_MIN_SCORE", "DETECTION_FILE", "CROP_SIZE",
                 "DETR_MODEL_NAME", "DETR_REVISION", "DETR_SHORT_SIDE", "DETR_MAX_SIZE",
                 "SINKHORN_ITERATIONS", "SINKHORN_SCORE_SCALE", "SINKHORN_UNMATCHED_LOGIT"):
        if checkpoint["settings"][name] != getattr(config, name):
            raise ValueError(f"{name} differs from the checkpoint; use its saved setting before tracking")
    model = AssociationModel(**checkpoint["model_settings"]).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    tqdm.write(f"Loading appearance encoder on {config.DEVICE}")
    encoder = load_encoder()
    if not config.FREEZE_ENCODER:
        if checkpoint.get("encoder_state_dict") is None:
            raise ValueError("Fine-tuned appearance weights are missing from this checkpoint")
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    if encoder.feature_settings != checkpoint["feature_settings"]:
        raise ValueError("Encoder preprocessing differs from the checkpoint")
    tqdm.write(f"Checkpoint epoch {checkpoint['epoch']}; provisional match threshold "
          f"{config.MATCH_THRESHOLD}. Tune thresholds on validation; these are not independent final-test scores.")
    summaries = []
    with tqdm(total=total_frames, desc="Tracking overall", unit="frame", position=0,
              smoothing=0, dynamic_ncols=True, disable=not config.SHOW_PROGRESS) as overall:
        for index, (sequence, split) in enumerate(jobs, 1):
            overall.set_postfix(video=f"{index}/{len(jobs)}", split=split,
                                sequence=sequence, refresh=False)
            summaries.append(run_sequence(sequence, model, encoder, split=split, overall=overall))
    return summaries


if __name__ == "__main__":
    main()
