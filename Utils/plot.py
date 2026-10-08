import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Utils.data import read_image

def plot_with_boxes(frame, gt=None):
    img = read_image(frame)

    fig, ax = plt.subplots()
    data = frame
    if gt is not None:
        data = gt

    for i in range(len(data["boxes"])):
        box = data["boxes"][i]

        rect = patches.Rectangle(
            (box[0], box[1]),       # (x, y) coordinates of the top-left/anchor corner
            box[2]-box[0],             # Width of the rectangle
            box[3]-box[1],             # Height of the rectangle
            linewidth=1,     # Thickness of the border line
            edgecolor='r',   # Border color (e.g., 'r' for red, 'g' for green)
            facecolor='none' # Keep the inside transparent
        )
        if gt is not None:
            ax.text(box[0], box[1]-2, data["visibility"][i], fontsize=6, color="red")
            ax.text((box[2]+box[0])/2, box[1]-2, data["ids"][i], fontsize=8, color="red")
        else:
            ax.text(box[0], box[1]-2, data["scores"][i], fontsize=6, color="red")
            ax.text((box[2]+box[0])/2, box[1]-2, data["row_ids"][i], fontsize=8, color="red")
        ax.add_patch(rect)

    ax.imshow(img)
    plt.show()


def plot_graph_demo(images, boxes, graph, frame_ids, focus_track=0):
    """Show labelled detections, candidate edges, and appearance similarities.

    Returns the figure so main.py can save/show it. T/D labels are input rows,
    not persistent person IDs; highlighted edges are possibilities only.
    """
    import numpy as np

    t, d = graph["num_tracks"], graph["num_detections"]
    pairs = graph["candidate_pairs"].numpy()
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    figure.suptitle("How build_graph connects two frames — no identities assigned yet", fontsize=15)
    colors = ("deepskyblue", "darkorange")

    for ax, image, frame_boxes, frame_id, prefix, color in zip(
            axes[0], images, boxes, frame_ids, ("T", "D"), colors):
        ax.imshow(image)
        for row, box in enumerate(frame_boxes):
            x1, y1, x2, y2 = box
            edge_color = "yellow" if prefix == "T" and row == focus_track else color
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, fill=False, edgecolor=edge_color, linewidth=1.4,
            ))
            ax.text(x1, y1, f"{prefix}{row}", color="black", fontsize=8,
                    bbox=dict(facecolor=edge_color, edgecolor="none", alpha=0.85, pad=1))
        role = "earlier detections used as tracks" if prefix == "T" else "current detections"
        ax.set_title(f"Frame {frame_id}: {role}")
        ax.axis("off")

    # One line represents a candidate pair; messages exist in both directions.
    ax = axes[1, 0]
    track_y = np.linspace(0.95, 0.05, t) if t else np.empty(0)
    detection_y = np.linspace(0.95, 0.05, d) if d else np.empty(0)
    for track, detection in pairs:
        ax.plot([0, 1], [track_y[track], detection_y[detection]],
                color="lightgray", alpha=0.35, linewidth=0.6, zorder=1)
    for track, detection in pairs:
        if track == focus_track:
            ax.plot([0, 1], [track_y[track], detection_y[detection]],
                    color="tab:blue", alpha=0.8, linewidth=1.2, zorder=2)
    ax.scatter(np.zeros(t), track_y, color=colors[0], s=45, zorder=3)
    ax.scatter(np.ones(d), detection_y, color=colors[1], s=45, zorder=3)
    for row, y in enumerate(track_y):
        ax.text(-0.04, y, f"T{row} / node {row}", ha="right", va="center", fontsize=8)
    for row, y in enumerate(detection_y):
        ax.text(1.04, y, f"D{row} / node {t+row}", ha="left", va="center", fontsize=8)
    ax.set(xlim=(-0.4, 1.4), ylim=(-0.08, 1.08))
    ax.set_title(f"{t+d} nodes | {len(pairs)} candidates | {2*len(pairs)} directed edges")
    caption = (f"Blue: candidates from T{focus_track}. Each line has two directions."
               if 0 <= focus_track < t else "Each candidate line has two message directions.")
    ax.text(0.5, -0.03, caption, ha="center", fontsize=9)
    if not len(pairs):
        ax.text(0.5, 0.5, "No candidate edges", ha="center")
    ax.axis("off")

    ax = axes[1, 1]
    if t and d:
        similarity = np.full((t, d), np.nan)
        from Utils.graph import EDGE_FEATURE_NAMES
        column = EDGE_FEATURE_NAMES.index("cosine")
        similarity[pairs[:, 0], pairs[:, 1]] = graph["edge_attr"][:len(pairs), column].detach().numpy()
        heatmap = ax.imshow(similarity, vmin=-1, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(d), [f"D{i}" for i in range(d)], rotation=45)
        ax.set_yticks(range(t), [f"T{i}" for i in range(t)])
        ax.set_xlabel("Current-frame detection row")
        ax.set_ylabel("Earlier-frame track row")
        figure.colorbar(heatmap, ax=ax, label="Cosine similarity")
    else:
        ax.text(0.5, 0.5, "No pairs to compare", ha="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Appearance similarity only — not a learned match score")
    return figure


def plot_model_demo(before, after, targets, valid_pairs, history):
    """Compare model scores, labelled targets, and losses; return a figure.

    The last row/column show unmatched options. The bottom-right transport
    cell is bookkeeping mass, so hide it instead of treating it as a score.
    """
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    steps = max(0, len(history) - 1)
    figure.suptitle("Association model: two-frame learning demonstration (not tracking accuracy)", fontsize=15)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    t, d = before["scores"].shape
    for ax, output, title in zip(axes[0], (before, after),
            ("Before: random association weights",
             f"After {steps} updates on this same pair" if steps else "No learning updates requested / available")):
        values = output["transport"].detach().cpu().numpy().copy()
        values[-1, -1] = np.nan
        values[:t, :d][~output["mask"].detach().cpu().numpy()] = np.nan
        heatmap = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(d+1), [f"D{i}" for i in range(d)] + ["Unmatched"], rotation=60)
        ax.set_yticks(range(t+1), [f"T{i}" for i in range(t)] + ["Unmatched"])
        ax.axhline(t-0.5, color="white", linewidth=1)
        ax.axvline(d-0.5, color="white", linewidth=1)
        ax.set_title(title)
        figure.colorbar(heatmap, ax=ax, label="Soft normalized score")

    ax = axes[1, 0]
    if targets is not None and t and d:
        truth = targets.detach().cpu().numpy().copy()
        supervised = valid_pairs & before["mask"].detach().cpu()
        truth[~supervised.numpy()] = np.nan
        heatmap = ax.imshow(truth, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(d), [f"D{i}" for i in range(d)], rotation=60)
        ax.set_yticks(range(t), [f"T{i}" for i in range(t)])
        figure.colorbar(heatmap, ax=ax, ticks=[0, 1], label="0 = different person; 1 = same person")
    else:
        ax.text(0.5, 0.5, "No labelled pairs to display", ha="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Ground-truth targets (grey = ignored; used only in loss)")

    ax = axes[1, 1]
    if history:
        for name in ("total", "bce", "contrastive"):
            ax.plot(range(len(history)), [entry[name] for entry in history],
                    label=name, marker="o" if not steps else None)
        ax.legend()
        ax.set(xlabel="Updates on this pair (0 = before learning)", ylabel="Loss")
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Loss unavailable: no valid supervision", ha="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Loss on the same example — not validation performance")
    return figure
