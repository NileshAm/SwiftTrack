"""Save run charts from CSV files without loading a model or opening windows.

Run `python main.py` for the configured run. See the plotting README for the
meaning of each chart. Existing Utils/plot.py contains interactive frame demos.
"""

import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def _read_rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No completed rows in {path}")
    return rows


def _values(rows, key):
    """Missing optional columns become gaps, never invented zeros."""
    return np.array([float(row[key]) if row.get(key) else np.nan for row in rows])


def _figure(rows=1, cols=1, *, width=12, height=7):
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, cols, squeeze=False)


def _save(figure, folder, name):
    """Write a PNG and release its artists; no pyplot/global backend changes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    try:
        figure.savefig(path, dpi=160, facecolor="white")
    finally:
        figure.clear()
    return path


def _epoch_axis(ax, ylabel):
    ax.set(xlabel="Epoch", ylabel=ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    if ax.lines:
        ax.legend(fontsize=9)


def plot_training(history_path, output_dir=None):
    """Plot completed epochs; mark the earliest minimum validation-loss epoch.

    Returns saved PNG paths. Can read older histories without assignment loss.
    The losses are graph averages, not accuracy; pair sampling differs by split.
    """
    history_path = Path(history_path)
    folder = Path(output_dir) if output_dir is not None else history_path.parent / "plots" / "training"
    rows = _read_rows(history_path)
    epochs = _values(rows, "epoch")
    validation = _values(rows, "val_total")
    if not np.isfinite(epochs).all() or not np.isfinite(validation).any():
        raise ValueError("History needs finite epochs and at least one finite val_total")
    best = np.flatnonzero(np.isfinite(validation))[np.argmin(validation[np.isfinite(validation)])]
    paths = []
    fig, axes = _figure(2, 2, height=8)
    sampling = ("Training/validation: chronological motion/history state (GT-guided updates)"
                if "val_mean_history_size" in rows[0] else
                "Training: random historical gaps; validation: deterministic historical gaps"
                if "val_max_frame_gap" in rows[0] else
                "Training: sampled pairs and temporal gaps; validation: all eligible pairs, adjacent frames")
    fig.suptitle(f"{history_path.parent.name} — association losses\n{sampling}")
    for ax, key, title in zip(axes.flat, ("total", "bce", "contrastive", "assignment"),
                              ("Total loss", "BCE loss", "Contrastive loss", "Assignment loss")):
        if key == "assignment" and all(
                np.all(_values(rows, f"{split}_{key}") == 0) for split in ("train", "val")):
            ax.set_axis_off()
            ax.text(0.5, 0.5, "WBCE + contrastive objective\nNo separate assignment loss",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        for split, label, color in (("train", "Training", "tab:blue"), ("val", "Validation", "tab:orange")):
            values = _values(rows, f"{split}_{key}")
            if np.isfinite(values).any():
                ax.plot(epochs, values, marker="o", markersize=3, label=label, color=color)
        if ax.lines:
            ax.axvline(epochs[best], color="gray", linestyle="--", label=f"Best total val: epoch {epochs[best]:g}")
        else:
            ax.text(0.5, 0.5, "Not recorded in this history", ha="center", transform=ax.transAxes)
        ax.set_title(title)
        _epoch_axis(ax, "Mean loss per supervised frame pair")
    paths.append(_save(fig, folder, "losses"))

    if "val_bce_per_weight" in rows[0]:
        fig, axes = _figure(1, 2, height=4.5)
        fig.suptitle("Loss diagnostics per pair — training samples negatives; validation uses all")
        for ax, key, title in zip(axes.flat, ("bce_per_weight", "contrastive_per_pair"),
                                  ("Weighted BCE / sum of weights", "Contrastive loss / number of pairs")):
            for split, label in (("train", "Training"), ("val", "Validation")):
                ax.plot(epochs, _values(rows, f"{split}_{key}"), marker="o", label=label)
            ax.set_title(title)
            _epoch_axis(ax, "Per-pair diagnostic (not the summed objective)")
        paths.append(_save(fig, folder, "losses_per_pair"))

    fig, axes = _figure(1, 2, height=4.5)
    fig.suptitle(f"{history_path.parent.name} — time per epoch")
    for split, label in (("train", "Training"), ("val", "Validation")):
        seconds = _values(rows, f"{split}_seconds")
        axes[0, 0].plot(epochs, seconds / 60, marker="o", label=label)
    axes[0, 1].plot(epochs, np.cumsum(_values(rows, "train_seconds") + _values(rows, "val_seconds")) / 3600,
                    marker="o", label="Training + validation")
    _epoch_axis(axes[0, 0], "Minutes")
    _epoch_axis(axes[0, 1], "Cumulative hours (excludes setup/saving/plots)")
    paths.append(_save(fig, folder, "timing"))

    fig, axes = _figure(1, 2, height=4.5)
    fig.suptitle(f"{history_path.parent.name} — supervision counts (not accuracy)")
    for ax, split, title in zip(axes.flat, ("train", "val"), ("Training", "Validation")):
        for key, label in (("used_pairs", "Supervised frame pairs"), ("skipped_pairs", "Skipped frame pairs"),
                           ("positive_pairs", "Positive cross-frame links"), ("negative_pairs", "Negative cross-frame links")):
            values = _values(rows, f"{split}_{key}")
            if np.isfinite(values).any():
                ax.plot(epochs, values, marker="o", markersize=3, label=label)
        ax.set_title(title)
        ax.set_yscale("symlog", linthresh=1)
        _epoch_axis(ax, "Count (log-like scale; zero is shown)")
    paths.append(_save(fig, folder, "supervision"))
    return paths


def plot_evaluation(evaluation_dir):
    """Plot native TrackEval tables separately per benchmark, without rescoring.

    Detailed CSV rates are fractions: multiply by 100 once. COMBINED is
    TrackEval's aggregate, never an average of per-video percentages. MOTA can
    be negative. Counts remain counts. Curves vary IoU, not model confidence.
    """
    evaluation_dir = Path(evaluation_dir)
    tables = sorted(evaluation_dir.glob("*/DragonTrack/pedestrian_detailed.csv"))
    if not tables:
        raise FileNotFoundError(f"No TrackEval detailed CSV in {evaluation_dir}; run evaluate.py first")
    paths = []
    for table in tables:
        rows = _read_rows(table)
        combined = next((row for row in rows if row["seq"] == "COMBINED"), None)
        sequences = [row for row in rows if row["seq"] != "COMBINED"]
        if combined is None or not sequences:
            raise ValueError(f"Expected per-sequence rows and COMBINED in {table}")
        benchmark = table.parent.parent.name
        folder = evaluation_dir / "plots" / benchmark
        title = f"{evaluation_dir.parent.parent.name} / {evaluation_dir.name} / {benchmark}"
        metrics = [("HOTA___AUC", "HOTA"), ("DetA___AUC", "Detection accuracy"),
                   ("AssA___AUC", "Association accuracy"), ("IDF1", "IDF1"),
                   ("MOTA", "MOTA"), ("CLR_Re", "Detection recall"), ("CLR_Pr", "Detection precision")]
        fig, axes = _figure(height=5)
        ax = axes[0, 0]
        values = [100 * float(combined[key]) for key, _ in metrics]
        bars = ax.barh([label for _, label in metrics], values, color="steelblue")
        ax.bar_label(bars, fmt="%.1f", padding=4)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(min(0, min(values)) - 5, max(100, max(values)) + 12)
        ax.set(xlabel="Score (%) — higher is better; MOTA may be negative",
               title=f"{title}\nCombined tracking metrics (TrackEval aggregation)")
        paths.append(_save(fig, folder, "overall_metrics"))

        names = [row["seq"] for row in sequences]
        y = np.arange(len(names))
        fig, axes = _figure(1, 2, width=14, height=max(4.5, len(names) * 0.6 + 2))
        fig.suptitle(f"{title} — per-video metrics")
        for ax, chosen in zip(axes.flat, ([("HOTA___AUC", "HOTA"), ("IDF1", "IDF1"), ("MOTA", "MOTA")],
                                        [("DetA___AUC", "DetA"), ("AssA___AUC", "AssA")])):
            width = 0.22
            for index, (key, label) in enumerate(chosen):
                ax.barh(y + (index - (len(chosen)-1)/2) * width,
                        _values(sequences, key) * 100, height=width, label=label)
            ax.set_yticks(y, names)
            ax.invert_yaxis()
            ax.set_xlabel("Score (%) — higher is better")
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.legend()
            ax.grid(axis="x", alpha=0.2)
        paths.append(_save(fig, folder, "per_sequence_metrics"))

        fig, axes = _figure(1, 2, width=14, height=max(4.5, len(names) * 0.6 + 2))
        fig.suptitle(f"{title} — tracking errors (lower is better)\n"
                     "FN: missed people; FP: extra boxes; IDSW: identity switches")
        gt = _values(sequences, "GT_Dets")
        for ax, normalized in zip(axes.flat, (False, True)):
            left = np.zeros(len(sequences))
            for key, label, color in (("CLR_FN", "FN", "tab:red"), ("CLR_FP", "FP", "tab:orange"),
                                      ("IDSW", "IDSW", "tab:purple")):
                values = _values(sequences, key)
                if normalized:
                    values = np.divide(values * 100, gt, out=np.full_like(gt, np.nan), where=gt > 0)
                ax.barh(y, values, left=left, label=label, color=color)
                left += values
            ax.set_yticks(y, names)
            ax.invert_yaxis()
            ax.legend()
            ax.set_xlabel("Errors per 100 GT detections (can exceed 100)" if normalized else "Error count")
            ax.grid(axis="x", alpha=0.2)
        paths.append(_save(fig, folder, "tracking_errors"))

        fig, axes = _figure(1, 2, height=5)
        fig.suptitle(f"{title} — detection and association diagnostics")
        ax = axes[0, 0]
        thresholds = sorted(int(key.split("___")[1]) for key in combined
                            if key.startswith("HOTA___") and key.split("___")[1].isdigit())
        for key, label in (("HOTA", "HOTA"), ("DetA", "DetA"), ("AssA", "AssA")):
            ax.plot(np.array(thresholds) / 100,
                    [100 * float(combined[f"{key}___{threshold}"]) for threshold in thresholds], label=label)
        ax.set(xlabel="Localization IoU threshold (not confidence)", ylabel="Combined score (%)", ylim=(0, 105))
        ax.legend()
        ax.grid(alpha=0.2)
        ax = axes[0, 1]
        for row in sequences:
            ax.scatter(100 * float(row["CLR_Re"]), 100 * float(row["CLR_Pr"]), label=row["seq"], s=45)
        ax.scatter(100 * float(combined["CLR_Re"]), 100 * float(combined["CLR_Pr"]),
                   marker="*", s=160, color="black", label="Combined")
        ax.set(xlabel="Detection recall (%)", ylabel="Detection precision (%)", xlim=(0, 105), ylim=(0, 105),
               title="One operating point per video; not a PR curve")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        paths.append(_save(fig, folder, "diagnostics"))
    return paths


def save_run_plots(run_dir, evaluation_dir=None):
    """Regenerate available charts; report missing stages without running them."""
    run_dir = Path(run_dir)
    paths = []
    if (run_dir / "history.csv").is_file():
        paths.extend(plot_training(run_dir / "history.csv"))
    else:
        print(f"Training plots skipped: no {run_dir / 'history.csv'}")
    evaluation_dir = Path(evaluation_dir) if evaluation_dir is not None else run_dir / "evaluation" / "val"
    if list(evaluation_dir.glob("*/DragonTrack/pedestrian_detailed.csv")):
        paths.extend(plot_evaluation(evaluation_dir))
    else:
        print(f"Evaluation plots skipped: no TrackEval tables in {evaluation_dir}")
    return paths


def auto_plot(stage, path):
    """A chart failure must not discard an already saved training/evaluation run."""
    from tqdm.auto import tqdm

    try:
        return plot_training(path) if stage == "training" else plot_evaluation(path)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        tqdm.write(f"Could not save {stage} plots: {error}. Saved metrics/weights are intact; retry with python main.py.")
        return []
