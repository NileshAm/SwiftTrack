"""Diagnose appearance before/after graph attention: python test.py

Uses validation labels for measurement and past-state updates only. No training,
weight updates, tracking assignment, or changes to config.py. See the guide in
Documentation/Implementation/21-feature-diagnostic.md for interpretation.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from Utils import config
from Utils.detections import file_digest
from Utils.features import load_encoder
from Utils.splits import check_checkpoint_split, validate_video_splits
from Utils.track_state import state_settings
from model import AssociationModel, MODEL_VERSION
from train import iter_graphs, load_frame_pairs


# Edit these defaults, or use the command-line options shown by --help.
CHECKPOINT_PATH = config.CHECKPOINT_PATH
MAX_FRAMES_PER_SEQUENCE = None  # None = full validation videos; 100 = first 100 frames each.
DEVICE = config.DEVICE
OUTPUT_ROOT = None  # None = <checkpoint folder>/feature_diagnostics/<timestamp>/
SHOW_PROGRESS = True
HISTOGRAM_BINS = np.linspace(-1.0, 1.0, 101)
TIE_TOLERANCE = 1e-6  # Near-equal scores count as ties, not correct predictions.


def load_models(checkpoint_path, device):
    """Load compatible saved weights and original encoder; never initialize a random diagnostic model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("Checkpoint architecture differs from the current residual model. "
                         "Run python train.py with a fresh RUN_NAME first; old reports remain usable for comparison.")
    check_checkpoint_split(checkpoint)
    temporal = checkpoint.get("temporal_sampling", {})
    if (temporal.get("protocol") != "chronological_track_state_v1"
            or temporal.get("max_frame_gap") != config.MAX_MISSED_FRAMES + 1
            or temporal.get("state_settings") != state_settings()):
        raise ValueError("Use the checkpoint's motion/history, confirmation, distance gate and look-back settings")
    for name in ("ENCODER_NAME", "USE_PRETRAINED_WEIGHTS", "FREEZE_ENCODER", "FEATURE_DIM",
                 "BOX_ORIGIN", "DETECTION_MIN_SCORE", "DETECTION_FILE", "CROP_SIZE", "GT_MATCH_IOU",
                 "DETR_MODEL_NAME", "DETR_REVISION", "DETR_SHORT_SIDE", "DETR_MAX_SIZE",
                 "SINKHORN_ITERATIONS", "SINKHORN_SCORE_SCALE", "SINKHORN_UNMATCHED_LOGIT"):
        if checkpoint["settings"][name] != getattr(config, name):
            raise ValueError(f"config.{name} differs from this checkpoint; restore its saved setting")
    model = AssociationModel(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    encoder = load_encoder(device)
    if not config.FREEZE_ENCODER:
        if checkpoint.get("encoder_state_dict") is None:
            raise ValueError("Fine-tuned encoder weights are missing from the checkpoint")
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if encoder.feature_settings != checkpoint["feature_settings"]:
        raise ValueError("Encoder preprocessing differs from the checkpoint")
    model.eval()
    encoder.eval()
    metadata = {key: checkpoint.get(key) for key in (
        "epoch", "model_version", "data_split", "training_frame_limit", "training_frame_ranges", "feature_settings")}
    return model, encoder, metadata


def new_stats():
    """Fixed-size accumulators: retain all pair counts without storing every embedding."""
    return dict(graphs=0, positives_before_gate=0, positives_kept=0, negatives_kept=0,
                ambiguous_tracks=0, sums=np.zeros((2, 2)),
                histograms=np.zeros((2, 2, len(HISTOGRAM_BINS)-1), dtype=np.int64),
                strict_wins=np.zeros(2, dtype=np.int64), ties=np.zeros(2, dtype=np.int64),
                margin_sums=np.zeros(2), candidate_nodes=0,
                zero_nodes=np.zeros(2, dtype=np.int64))


def observe_graph(stats, graph, output, targets, valid, feature_dim):
    """Compare identical labelled candidates; measure gate misses separately from feature ranking.

    A ranking query is a track with an allowed correct detection AND at least
    one allowed, labelled incorrect detection. Unknown identities are excluded.
    Both representations are measured before Sinkhorn/Hungarian/thresholding.
    """
    t = graph["num_tracks"]
    before = graph["node_features"][:, :feature_dim].detach().cpu().float()
    after = torch.cat((output["track_embeddings"], output["detection_embeddings"])).detach().cpu().float()
    targets, valid = targets.detach().cpu().bool(), valid.detach().cpu().bool()
    mask = graph["candidate_mask"].cpu().bool()
    if targets.shape != mask.shape or valid.shape != mask.shape:
        raise ValueError("Labels must align with the graph candidate mask")
    if len(before) != sum(mask.shape) or len(after) != len(before):
        raise ValueError("Embedding rows must align with track and detection rows")
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("Non-finite appearance/graph embeddings")
    correct = targets & valid
    positive = correct & mask
    negative = ~targets & valid & mask
    ambiguous = positive.any(dim=1) & negative.any(dim=1)
    labelled_candidates = positive | negative
    participating = torch.cat((labelled_candidates.any(dim=1), labelled_candidates.any(dim=0)))
    stats["graphs"] += 1
    stats["positives_before_gate"] += int(correct.sum())
    stats["positives_kept"] += int(positive.sum())
    stats["negatives_kept"] += int(negative.sum())
    stats["ambiguous_tracks"] += int(ambiguous.sum())
    stats["candidate_nodes"] += int(participating.sum())
    for stage, features in enumerate((before, after)):
        stats["zero_nodes"][stage] += int(((features.norm(dim=1) <= 1e-8) & participating).sum())
        features = F.normalize(features, dim=1, eps=1e-8)
        scores = (features[:t] @ features[t:].T).clamp(-1, 1)
        for label, selected in enumerate((positive, negative)):
            values = scores[selected].numpy()
            stats["sums"][stage, label] += values.sum(dtype=np.float64)
            stats["histograms"][stage, label] += np.histogram(values, bins=HISTOGRAM_BINS)[0]
        if ambiguous.any():
            pos = scores[ambiguous].masked_fill(~positive[ambiguous], -torch.inf).max(dim=1).values
            neg = scores[ambiguous].masked_fill(~negative[ambiguous], -torch.inf).max(dim=1).values
            margin = pos - neg
            stats["strict_wins"][stage] += int((margin > TIE_TOLERANCE).sum())
            stats["ties"][stage] += int((margin.abs() <= TIE_TOLERANCE).sum())
            stats["margin_sums"][stage] += float(margin.double().sum())


def summarize(stats):
    """Pool observations by their counts; represent unavailable measurements as None."""
    p, n, queries = stats["positives_kept"], stats["negatives_kept"], stats["ambiguous_tracks"]
    result = {key: stats[key] for key in ("graphs", "positives_before_gate", "positives_kept",
                                         "negatives_kept", "ambiguous_tracks", "candidate_nodes")}
    excluded = stats["positives_before_gate"] - p
    result.update(positives_excluded_by_gate=excluded,
                  gate_exclusion_percent=100*excluded/stats["positives_before_gate"] if stats["positives_before_gate"] else None)
    for stage, name in enumerate(("before", "after")):
        positive_mean = float(stats["sums"][stage, 0]/p) if p else None
        negative_mean = float(stats["sums"][stage, 1]/n) if n else None
        result[name] = dict(positive_mean=positive_mean, negative_mean=negative_mean,
                           separation=positive_mean-negative_mean if p and n else None,
                           strict_top1_percent=100*int(stats["strict_wins"][stage])/queries if queries else None,
                           tie_percent=100*int(stats["ties"][stage])/queries if queries else None,
                           mean_ranking_margin=float(stats["margin_sums"][stage]/queries) if queries else None,
                           zero_candidate_nodes=int(stats["zero_nodes"][stage]),
                           positive_histogram=stats["histograms"][stage, 0].tolist(),
                           negative_histogram=stats["histograms"][stage, 1].tolist())
    return result


def save_plot(result, path, title):
    """Plot exact streamed distributions, strict ranking/ties and gate counts."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    fig.suptitle(title + " — labelled feature diagnostic, not tracking accuracy")
    for ax, stage in zip(axes[0], ("before", "after")):
        for label, count in (("positive", result["positives_kept"]), ("negative", result["negatives_kept"])):
            if count:
                density = np.asarray(result[stage][label+"_histogram"])/(count*np.diff(HISTOGRAM_BINS))
                ax.stairs(density, HISTOGRAM_BINS, label="Same person" if label == "positive" else "Different people")
        ax.set(title=stage.capitalize()+" graph attention", xlabel="Cosine similarity", ylabel="Density", xlim=(-1, 1))
        if result["positives_kept"] or result["negatives_kept"]:
            ax.legend()
        else:
            ax.text(.5, .5, "No labelled candidates", transform=ax.transAxes, ha="center")
    ax = axes[1, 0]
    if result["ambiguous_tracks"]:
        wins = [result[s]["strict_top1_percent"] for s in ("before", "after")]
        ties = [result[s]["tie_percent"] for s in ("before", "after")]
        ax.bar(["Before", "After"], wins, label="Correct strictly ranks first")
        ax.bar(["Before", "After"], ties, bottom=wins, label="Tied; not counted correct")
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "No labelled competing candidates\nRanking is inconclusive", transform=ax.transAxes, ha="center")
    ax.set(title=f"Ambiguous track queries: {result['ambiguous_tracks']}", ylabel="Queries (%)", ylim=(0, 105))
    ax = axes[1, 1]
    bars = ax.bar(["Kept", "Excluded by gate"], [result["positives_kept"], result["positives_excluded_by_gate"]])
    ax.bar_label(bars)
    ax.set(title="Known correct pairs already present in memory", ylabel="Pair observations")
    try:
        fig.savefig(path, dpi=150)
    finally:
        fig.clear()


def save_report(folder, results, metadata):
    """Write per-video/pooled CSV, JSON, plots and a readable comparison."""
    columns = ["sequence", "stage", "positive_mean", "negative_mean", "separation",
               "strict_top1_percent", "tie_percent", "mean_ranking_margin", "zero_candidate_nodes",
               "graphs", "positives_kept", "negatives_kept", "ambiguous_tracks", "gate_exclusion_percent"]
    lines = ["# Appearance before and after graph attention", "",
             "GT-guided history; identical labelled candidates before and after attention. No training.", "",
             "| Video | Stage | Same-person mean | Different-person mean | Gap | Strict top-1 (%) | Ties (%) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value): return "N/A" if value is None else f"{value:.4f}"
    with (folder / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for sequence, result in results.items():
            for stage in ("before", "after"):
                row = dict(sequence=sequence, stage=stage)
                row.update({key: result[stage].get(key, result.get(key)) for key in columns[2:]})
                writer.writerow(row)
                lines.append(f"| {sequence} | {stage} | " + " | ".join(fmt(row[key]) for key in (
                    "positive_mean", "negative_mean", "separation", "strict_top1_percent", "tie_percent")) + " |")
            save_plot(result, folder / f"{sequence}.png", sequence)
            tqdm.write(f"{sequence}: gap {fmt(result['before']['separation'])} -> {fmt(result['after']['separation'])}; "
                       f"strict top-1 {fmt(result['before']['strict_top1_percent'])}% -> "
                       f"{fmt(result['after']['strict_top1_percent'])}%; queries={result['ambiguous_tracks']}; "
                       f"correct pairs excluded by gate={fmt(result['gate_exclusion_percent'])}%")
    lines += ["", "Higher separation and strict top-1 are desirable. Ties are reported separately, never resolved by row order.",
              "Only tracks with an allowed correct match and an allowed labelled incorrect competitor enter ranking.",
              "A gate exclusion is measured only among known same-ID pairs present in the current memory/detections; it is not detector recall.",
              "Unknown IDs are excluded from measurements but still participate in graph construction.",
              "Before uses actual appearance/history inputs without the four box coordinates. After includes graph processing of appearance and geometry.",
              "Histograms use every eligible pair, without negative sampling. Pooled results are weighted by observations, not averaged video percentages.",
              "Consistent deterioration in both separation and ranking supports testing a residual connection; this does not prove it will help HOTA/IDF1.",
              "High similarities alone do not prove collapse. Few competing candidates, ties and zero vectors need separate attention.",
              "GT updates history after scoring, so wrong past tracking assignments are not simulated. Use actual tracking evaluation to confirm any change.",
              "Prefixes, if selected, are preliminary checks and omit later scenes. No missing detections are recovered by this diagnostic.", ""]
    (folder / "report.md").write_text("\n".join(lines))
    (folder / "summary.json").write_text(json.dumps(
        dict(complete=True, metadata=metadata, histogram_bin_edges=HISTOGRAM_BINS.tolist(), results=results),
        indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_PER_SEQUENCE,
                        help="First N frames per validation video, including the initial memory seed (default: all)")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.max_frames is not None and (isinstance(args.max_frames, bool)
            or not isinstance(args.max_frames, int) or args.max_frames < 2):
        parser.error("--max-frames must be an integer of at least 2")
    validate_video_splits()
    checkpoint_path = args.checkpoint.resolve()
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    tqdm.write(f"Checkpoint: {checkpoint_path}\nRead-only feature diagnostic using GT-guided validation history.")
    model, encoder, metadata = load_models(checkpoint_path, args.device)
    jobs = []
    for sequence in config.VAL_SEQUENCES:
        pairs = load_frame_pairs([sequence], "val")
        if args.max_frames is not None:
            pairs = pairs[:args.max_frames-1]
        jobs.append((sequence, pairs))
    root = args.output_root if args.output_root is not None else checkpoint_path.parent / "feature_diagnostics"
    folder = root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    results, ranges, pooled = {}, {}, new_stats()
    with torch.inference_mode(), tqdm(total=sum(len(pairs) for _, pairs in jobs),
            desc="Feature diagnostic overall", unit="graph", position=0,
            dynamic_ncols=True, smoothing=0, disable=not SHOW_PROGRESS) as overall:
        for sequence, pairs in jobs:
            stats = new_stats()
            ranges[sequence] = [pairs[0][0]["frame_id"], pairs[-1][1]["frame_id"]]
            tqdm.write(f"Inspecting {sequence}: frames {ranges[sequence]}")
            overall.set_postfix(sequence=sequence)
            with tqdm(iter_graphs(pairs, encoder, training=False), total=len(pairs),
                      desc=sequence, unit="graph", position=1, leave=False,
                      dynamic_ncols=True, disable=not SHOW_PROGRESS) as frames:
                for graph, targets, valid, _ in frames:
                    output = model(graph)
                    observe_graph(stats, graph, output, targets, valid, encoder.feature_dim)
                    frames.set_postfix(queries=stats["ambiguous_tracks"], refresh=False)
                    overall.update(1)
            results[sequence] = summarize(stats)
            for key in pooled:
                pooled[key] += stats[key]
    results["COMBINED"] = summarize(pooled)
    metadata.update(checkpoint=str(checkpoint_path), checkpoint_sha256=file_digest(checkpoint_path),
                    device=args.device, frame_ranges=ranges, max_frames_per_sequence=args.max_frames,
                    state_settings=state_settings(), tie_tolerance=TIE_TOLERANCE,
                    history_protocol="GT-guided updates after scoring; not free-running tracking",
                    seconds=time.perf_counter()-start)
    save_report(folder, results, metadata)
    tqdm.write(f"\nReport: {folder / 'report.md'}\nCSV: {folder / 'summary.csv'}\nPlots: {folder}")
    return folder


if __name__ == "__main__":
    main()
