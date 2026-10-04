"""Run `python train.py` in dragon-track; settings live in Utils/config.py.

Train attention/affinity/matching networks and optionally the DETR appearance encoder.
Training and validation process videos chronologically using past-only motion
predictions and appearance history. GT guides state updates after scoring
(teacher forcing); validation loss is not tracking-accuracy evaluation.
"""

import csv
from itertools import groupby, islice
from pathlib import Path
import random
import time

import numpy as np
import torch

from Utils import config
from Utils.data import load_sequence, load_ground_truth, read_image, match_detections_to_gt
from Utils.splits import validate_video_splits, full_frame_range, split_manifest
from Utils.detections import detection_path
from Utils.features import load_encoder, extract_features, extract_feature_batch
from Utils.graph import build_graph
from tqdm.auto import tqdm
from model import AssociationModel, MODEL_VERSION
from Utils.pairs import build_targets
from Utils.losses import compute_loss
from Utils.training_state import TrainingTrackState
from Utils.track_state import state_settings


def load_frame_pairs(sequences, split, *, max_frames=None):
    """Read metadata/GT once; return adjacent anchor pairs without loading images.

    Each entry is (earlier frame, current frame, earlier GT, current GT).
    iter_graphs uses the first frame to seed memory, then scores each current
    frame against accumulated tracks. Adjacent pairs only describe iteration order.
    Pairs never cross videos. max_frames optionally keeps the first N training
    frames per video; validation always uses full videos. Empty frames remain.
    """
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 2:
            raise ValueError("Training frame limit must be None or an integer of at least 2")
        if split != "train":
            raise ValueError("Frame limits apply only to training; validation uses full videos")
    validate_video_splits()
    allowed = {"train": config.TRAIN_SEQUENCES, "val": config.VAL_SEQUENCES}
    if split not in allowed or any(sequence not in allowed[split] for sequence in sequences):
        raise ValueError("Frame pairs must use the configured video list for their split")
    pairs = []
    for sequence in sequences:
        path = config.DATASET_ROOT / config.DATASET_SPLIT / sequence
        frame_range = full_frame_range(path)
        if max_frames is not None:
            frame_range = (frame_range[0], min(frame_range[1], frame_range[0] + max_frames - 1))
        frames = load_sequence(path, detection_path(path), frame_range,
                               box_origin=config.BOX_ORIGIN,
                               min_score=config.DETECTION_MIN_SCORE)
        gt = load_ground_truth(path, config.GT_FILE, box_origin=config.BOX_ORIGIN)
        ids = sorted(frames)
        sequence_pairs = [(frames[a], frames[b], gt[a], gt[b])
                          for a, b in zip(ids, ids[1:]) if b == a + 1]
        if not sequence_pairs:
            raise ValueError(f"{sequence}: select at least two consecutive frames")
        pairs.extend(sequence_pairs)
        tqdm.write(f"  {sequence} {split}: frames {frame_range[0]}..{frame_range[1]}, "
              f"{len(sequence_pairs)} current-frame anchors")
    if not pairs:
        raise ValueError("Select at least one sequence with consecutive frames")
    return pairs


def prepare_frame(frame, gt, encoder, extracted=None):
    """Extract one frame, retaining scores and labels aligned to valid features."""
    features, valid = (extract_features(read_image(frame), frame["boxes"], encoder, show_progress=False)
                       if extracted is None else extracted)
    keep = valid.numpy()
    current = dict(frame, boxes=frame["boxes"][keep], scores=frame["scores"][keep],
                   row_ids=frame["row_ids"][keep])
    labels = match_detections_to_gt(current["boxes"], gt, config.GT_MATCH_IOU)
    return current, features[valid], labels


def iter_graphs(frame_pairs, encoder, training=False):
    """Batch independent image extraction, then yield graphs chronologically.

    Only image features are precomputed. Each graph is scored before its labels
    update memory; future frames never enter earlier track states. Losses within
    a batch share encoder activations and must be backpropagated together at
    graph["batch_end"]. Batches never cross video boundaries.
    """
    batch_size = config.TRAIN_BATCH_SIZE if training else config.VAL_BATCH_SIZE
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("TRAIN_BATCH_SIZE and VAL_BATCH_SIZE must be positive integers")
    for directory, sequence_pairs in groupby(frame_pairs, key=lambda pair: pair[1]["image_path"].parent):
        state = TrainingTrackState(encoder.feature_dim)
        while batch := list(islice(sequence_pairs, batch_size)):
            for earlier, current, _, _ in batch:
                if earlier["image_path"].parent != directory:
                    raise ValueError("Training anchors cannot cross videos")
                if (earlier["height"], earlier["width"]) != (current["height"], current["width"]):
                    raise ValueError("Video frames must have the same dimensions")
            if state.last_update is None:
                earlier, _, earlier_gt, _ = batch[0]
                with torch.no_grad():
                    seed_frame, seed_features, seed_labels = prepare_frame(earlier, earlier_gt, encoder)
                state.update(seed_frame, seed_features, seed_labels)
            extracted = extract_feature_batch([read_image(pair[1]) for pair in batch],
                                              [pair[1]["boxes"] for pair in batch], encoder)
            for index, ((earlier, current, _, current_gt), feature_pair) in enumerate(zip(batch, extracted)):
                if earlier["frame_id"] != state.last_update or current["frame_id"] != state.last_update + 1:
                    raise ValueError("Training anchors must cover consecutive frames in order")
                boxes, features, labels, ages = state.inputs(current["frame_id"])
                observation, current_features, current_labels = prepare_frame(
                    current, current_gt, encoder, extracted=feature_pair)
                graph = build_graph(boxes, observation["boxes"], features, current_features,
                                    image_size=(current["height"], current["width"]))
                targets, valid = build_targets(labels, current_labels)
                graph["frame_gap"] = int(ages.max()) if len(ages) else 0
                graph["mean_frame_gap"] = float(ages.mean()) if len(ages) else 0.0
                graph["mean_history_size"] = (float(np.mean([
                    len(track["appearance_history"].samples) if "appearance_history" in track else 0
                    for track in state.tracks])) if state.tracks else 0.0)
                graph["batch_end"] = index == len(batch)-1
                graph["batch_size"] = len(batch)
                identities = tuple(np.where(item["valid_mask"], item["ids"], -1)
                                   for item in (labels, current_labels))
                yield graph, targets, valid, identities
                state.update(observation, current_features, current_labels)


def _run_epoch(model, frame_pairs, encoder, optimizer=None, *, overall=None):
    """Shared image-batch loop with sequential graph scoring; None means validation."""
    training = optimizer is not None
    batch_size = config.TRAIN_BATCH_SIZE if training else config.VAL_BATCH_SIZE
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    stage = "Train" if training else "Validate"
    previous_mode = model.training
    model.train(training)
    previous_encoder_mode = encoder.training
    encoder.train(training and not config.FREEZE_ENCODER)
    start = time.perf_counter()
    totals = dict(total=0.0, bce=0.0, contrastive=0.0, assignment=0.0, used_pairs=0,
                  skipped_pairs=0, positive_pairs=0, negative_pairs=0,
                  bce_weight_sum=0.0, contrastive_pairs=0, contrastive_positive_pairs=0,
                  contrastive_negative_pairs=0, max_marginal_error=0.0, max_normalization_error=0.0,
                  mean_frame_gap=0.0, max_frame_gap=0, mean_history_size=0.0)
    info = {"device": str(next(model.parameters()).device), "loss": "--", "skipped": 0}
    graphs = None
    parameters = [p for group in optimizer.param_groups for p in group["params"]] if training else []
    pending = 0
    batch_loss = None
    optimizer_steps = 0
    def step_batch(count):
        nonlocal optimizer_steps
        # Average over actual supervised graphs, including a short final batch.
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(count)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("Training produced nonfinite gradients")
        torch.nn.utils.clip_grad_norm_(parameters, config.GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    if training:
        optimizer.zero_grad(set_to_none=True)
    try:
        with torch.set_grad_enabled(training):
            graphs = tqdm(iter_graphs(frame_pairs, encoder, training=training), total=len(frame_pairs),
                          desc=stage, unit="frame", dynamic_ncols=True,
                          position=1 if overall is not None else 0, leave=overall is None,
                          disable=not config.SHOW_PROGRESS)
            for index, (graph, targets, valid, identities) in enumerate(graphs, 1):
                gap = graph["frame_gap"]
                totals["mean_frame_gap"] += graph["mean_frame_gap"]
                totals["mean_history_size"] += graph["mean_history_size"]
                totals["max_frame_gap"] = max(totals["max_frame_gap"], gap)
                info["max_age"] = gap
                info["history"] = f"{graph['mean_history_size']:.1f}"
                if not any((ids > 0).any() for ids in identities):
                    totals["skipped_pairs"] += 1
                else:
                    output = model(graph)
                    totals["max_marginal_error"] = max(totals["max_marginal_error"], output["marginal_error"].item())
                    totals["max_normalization_error"] = max(totals["max_normalization_error"], output["normalization_error"].item())
                    losses = compute_loss(output, targets, valid,
                                          identity_labels=identities, training=training)
                    for name in ("total", "bce", "contrastive", "assignment"):
                        if not torch.isfinite(losses[name]):
                            raise RuntimeError(f"{stage} frame pair {index}: nonfinite {name} loss")
                    if training:
                        batch_loss = losses["total"] if batch_loss is None else batch_loss + losses["total"]
                        pending += 1
                    totals["used_pairs"] += 1
                    for name in ("total", "bce", "contrastive", "assignment", "positive_pairs", "negative_pairs",
                                 "bce_weight_sum", "contrastive_pairs", "contrastive_positive_pairs",
                                 "contrastive_negative_pairs"):
                        value = losses[name]
                        totals[name] += value.item() if torch.is_tensor(value) else value
                if training and graph["batch_end"] and pending:
                    batch_loss.backward()
                    step_batch(pending)
                    pending, batch_loss = 0, None
                info["batch"] = graph["batch_size"]
                used = totals["used_pairs"]
                for name in ("total", "bce", "contrastive", "assignment"):
                    info["loss" if name == "total" else name] = f"{totals[name]/used:.4f}" if used else "--"
                info["BCE/weight"] = f"{totals['bce']/(totals['bce_weight_sum'] or 1):.4f}"
                info["contrast/pair"] = f"{totals['contrastive']/max(totals['contrastive_pairs'], 1):.4f}"
                info.pop("assignment", None)
                info.pop("bce", None)
                info.pop("contrastive", None)
                info["skipped"] = totals["skipped_pairs"]
                graphs.set_postfix(info, refresh=False)
                if overall is not None:
                    overall.update(1)
        totals["optimizer_steps"] = optimizer_steps
        if not totals["used_pairs"]:
            raise ValueError(f"{stage}: no supervised frame pairs; check GT and selected frames")
        # Divide epoch-wide sums by their actual populations, not mean-of-means.
        totals["bce_per_weight"] = totals["bce"] / (totals["bce_weight_sum"] or 1)
        totals["contrastive_per_pair"] = totals["contrastive"] / max(totals["contrastive_pairs"], 1)
        # Each supervised frame-pair graph contributes equally to the epoch loss.
        for name in ("total", "bce", "contrastive", "assignment"):
            totals[name] /= totals["used_pairs"]
        totals["seconds"] = time.perf_counter() - start
        totals["mean_frame_gap"] /= len(frame_pairs)
        totals["mean_history_size"] /= len(frame_pairs)
        return totals
    finally:
        if graphs is not None:
            graphs.close()
        model.train(previous_mode)
        encoder.train(previous_encoder_mode)


def train_one_epoch(model, frame_pairs, encoder, optimizer, *, overall=None):
    """Update association weights with Adam; return mean losses and counts."""
    return _run_epoch(model, frame_pairs, encoder, optimizer, overall=overall)


def validate(model, frame_pairs, encoder, *, overall=None):
    """Evaluate held-out pairs without gradients/updates; restore model mode."""
    return _run_epoch(model, frame_pairs, encoder, overall=overall)


def main(*, max_train_frames_per_sequence=None):
    """Load selected splits, run epochs, log losses, and save best/latest weights.

    Starts a fresh run. Choose a new RUN_NAME if checkpoint/history files exist.
    model.pt is selected by validation loss; last.pt is the most recent saved
    epoch. Validation videos never enter optimizer updates. Optional search-only
    frame cap keeps a consecutive prefix of every training video. Calling main()
    or running python train.py still uses all training frames.
    """
    if not isinstance(config.EPOCHS, int) or config.EPOCHS <= 0:
        raise ValueError("EPOCHS must be a positive integer")
    if not np.isfinite(config.LEARNING_RATE) or config.LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be finite and positive")
    for name in ("TRAIN_BATCH_SIZE", "VAL_BATCH_SIZE"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    validate_video_splits()
    recipe = state_settings()
    if not config.USE_PRETRAINED_WEIGHTS:
        raise ValueError("Training requires USE_PRETRAINED_WEIGHTS=True for the appearance encoder")
    output_dir = Path(config.OUTPUT_DIR)
    best_path = Path(config.CHECKPOINT_PATH)
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.csv"
    if best_path.resolve() == last_path.resolve():
        raise ValueError("CHECKPOINT_PATH must differ from outputs/<run_name>/last.pt")
    for path in (best_path, last_path, history_path):
        if path.exists():
            raise FileExistsError(f"{path} exists. Set a new RUN_NAME/output path for a fresh run.")

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    tqdm.write("Training sequences:")
    train_pairs = load_frame_pairs(config.TRAIN_SEQUENCES, "train",
                                   max_frames=max_train_frames_per_sequence)
    training_frame_ranges = {}
    for earlier, current, _, _ in train_pairs:
        bounds = training_frame_ranges.setdefault(current["sequence"],
                                                   [earlier["frame_id"], current["frame_id"]])
        bounds[1] = current["frame_id"]
    tqdm.write("Validation sequences:")
    val_pairs = load_frame_pairs(config.VAL_SEQUENCES, "val")
    data_split = split_manifest()
    tqdm.write(f"Loading appearance encoder and attention model on {config.DEVICE}")
    tqdm.write(f"Image batches: train={config.TRAIN_BATCH_SIZE}, validation={config.VAL_BATCH_SIZE}; "
               f"activation checkpointing={config.DETR_GRADIENT_CHECKPOINTING}")
    encoder = load_encoder()
    model = AssociationModel().to(config.DEVICE)
    parameters = list(model.parameters()) + [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.LEARNING_RATE)
    settings = {name: str(value) if isinstance(value, Path) else value
                for name, value in vars(config).items() if name.isupper()}
    model_settings = dict(feature_dim=config.FEATURE_DIM, hidden_dims=config.GRAPH_HIDDEN_DIMS)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch = float("inf"), 0
    history = []
    tqdm.write(f"Chronological track-state training: maximum observation gap {config.MAX_MISSED_FRAMES + 1}. "
               f"Motion={config.USE_MOTION_PREDICTION}, history={config.USE_APPEARANCE_HISTORY}. "
               "GT guides memory updates after scoring; validation loss is not tracking accuracy.")
    # Count all train AND validation pairs so ETA updates during the first epoch.
    with tqdm(total=config.EPOCHS * (len(train_pairs) + len(val_pairs)),
              desc="Training overall", unit="frame", position=0, smoothing=0,
              dynamic_ncols=True, disable=not config.SHOW_PROGRESS) as overall:
        for epoch in range(1, config.EPOCHS + 1):
            tqdm.write(f"\nEpoch {epoch}/{config.EPOCHS}")
            overall.set_postfix(epoch=f"{epoch}/{config.EPOCHS}", phase="train", refresh=False)
            training = train_one_epoch(model, train_pairs, encoder, optimizer, overall=overall)
            overall.set_postfix(epoch=f"{epoch}/{config.EPOCHS}", phase="validation", refresh=False)
            validation = validate(model, val_pairs, encoder, overall=overall)
            for label, values in (("Train", training), ("Validation", validation)):
                tqdm.write(f"  {label}: total={values['total']:.6f}, BCE={values['bce']:.6f}, "
                      f"contrastive={values['contrastive']:.6f}")
                tqdm.write(f"    BCE/weight={values['bce_per_weight']:.6f}, "
                           f"contrastive/pair={values['contrastive_per_pair']:.6f}, "
                           f"max marginal error={values['max_marginal_error']:.3g}")
            improved = validation["total"] < best_loss
            if improved:
                best_loss, best_epoch = validation["total"], epoch
            row = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in training.items()})
            row.update({f"val_{key}": value for key, value in validation.items()})
            history.append(row)
            checkpoint = {
                "method": "paper_inspired_attention", "model_version": MODEL_VERSION, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": encoder.state_dict() if not config.FREEZE_ENCODER else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "model_settings": model_settings, "settings": settings,
                "feature_settings": encoder.feature_settings, "seed": config.SEED,
                "data_split": data_split,
                "training_frame_limit": max_train_frames_per_sequence,
                "training_frame_ranges": training_frame_ranges,
                "temporal_sampling": {"protocol": "chronological_track_state_v1",
                                      "max_frame_gap": config.MAX_MISSED_FRAMES + 1,
                                      "validation": "chronological_teacher_forced",
                                      "state_settings": recipe},
                "best_val_loss": best_loss, "best_epoch": best_epoch,
                "train_metrics": training, "val_metrics": validation,
            }
            if improved:
                torch.save(checkpoint, best_path)
                tqdm.write(f"  Best validation checkpoint saved: {best_path}")
            if config.SAVE_EVERY_EPOCH or epoch == config.EPOCHS:
                torch.save(checkpoint, last_path)
            with history_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            if config.SAVE_PLOTS:
                from Utils.plots import auto_plot
                auto_plot("training", history_path)
    tqdm.write(f"\nFinished. Best epoch: {best_epoch}; validation loss: {best_loss:.6f}")
    tqdm.write(f"Weights: {best_path}\nLoss history: {history_path}")
    return history


if __name__ == "__main__":
    main()
