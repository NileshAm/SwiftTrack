"""Small TrackEval helpers for the fixed validation comparison reports."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import trackeval

from Utils import config
from Utils.data import _sequence_info

SEQUENCES = {name: _sequence_info(config.DATASET_ROOT / config.DATASET_SPLIT / name).getint("seqLength")
             for name in config.VAL_SEQUENCES}


def load(run, sequences=SEQUENCES):
    root = Path(run) / "evaluation/val/inputs/MOT17"
    for sequence in sequences:
        for path in (root / "gt" / sequence / "gt/gt.txt",
                     root / "trackers/DragonTrack/data" / f"{sequence}.txt"):
            if not path.is_file():
                raise FileNotFoundError(path)
    dataset = trackeval.datasets.MotChallenge2DBox(dict(
        GT_FOLDER=str(root / "gt"), TRACKERS_FOLDER=str(root / "trackers"),
        OUTPUT_FOLDER=str(Path(run) / "evaluation/val/MOT17"), TRACKERS_TO_EVAL=["DragonTrack"],
        BENCHMARK="MOT17", SPLIT_TO_EVAL="train", SKIP_SPLIT_FOL=True,
        SEQ_INFO=sequences, DO_PREPROC=True, PRINT_CONFIG=False,
        CLASSES_TO_EVAL=["pedestrian"]))
    return {sequence: dataset.get_preprocessed_seq_data(
        dataset.get_raw_seq_data("DragonTrack", sequence), "pedestrian") for sequence in sequences}


def score(data_by_sequence):
    results = {}
    metrics = (trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False}))
    for metric in metrics:
        per_sequence = {sequence: metric.eval_sequence(data) for sequence, data in data_by_sequence.items()}
        combined = metric.combine_sequences(per_sequence)
        for key in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "CLR_TP", "CLR_FP", "CLR_FN", "IDSW"):
            if key in combined:
                value = combined[key]
                results[key] = (round(float(np.mean(value)) * 100, 3) if key in ("HOTA", "DetA", "AssA")
                                else round(float(value) * 100, 3) if key in ("MOTA", "IDF1")
                                else int(value))
    return results


def association_only(data_by_sequence):
    """Identity consistency among output boxes matched to GT at IoU 0.5."""
    per_sequence = {}
    for sequence, data in data_by_sequence.items():
        pairs = Counter()
        labels = defaultdict(set)
        last = {}
        changes = after_gap = 0
        for frame in range(data["num_timesteps"]):
            overlap = np.asarray(data["similarity_scores"][frame], dtype=np.float64)
            if not overlap.size:
                continue
            eligible = overlap >= 0.5 - np.finfo(float).eps
            objective = np.where(eligible, 1000 + overlap, 0.)
            gt_rows, track_rows = linear_sum_assignment(-objective)
            for gt_row, track_row in zip(gt_rows, track_rows):
                if not eligible[gt_row, track_row]:
                    continue
                gt_id = int(data["gt_ids"][frame][gt_row])
                track_id = int(data["tracker_ids"][frame][track_row])
                pairs[gt_id, track_id] += 1
                labels[gt_id].add(track_id)
                if gt_id in last and last[gt_id][1] != track_id:
                    changes += 1
                    after_gap += frame - last[gt_id][0] > 1
                last[gt_id] = (frame, track_id)
        gt_ids = sorted(labels)
        track_ids = sorted({track for _, track in pairs})
        if gt_ids and track_ids:
            counts = np.zeros((len(gt_ids), len(track_ids)), dtype=np.int64)
            gt_index = {value: index for index, value in enumerate(gt_ids)}
            track_index = {value: index for index, value in enumerate(track_ids)}
            for (gt_id, track_id), count in pairs.items():
                counts[gt_index[gt_id], track_index[track_id]] = count
            gt_rows, track_rows = linear_sum_assignment(-counts)
            consistent = int(counts[gt_rows, track_rows].sum())
        else:
            consistent = 0
        per_sequence[sequence] = dict(matched_boxes=sum(pairs.values()), consistent_boxes=consistent,
            people=len(gt_ids), single_id_people=sum(len(value) == 1 for value in labels.values()),
            identity_changes=changes, identity_changes_after_gap=after_gap)
    total = {key: sum(item[key] for item in per_sequence.values())
             for key in next(iter(per_sequence.values()))}
    total["consistent_pct"] = round(100 * total["consistent_boxes"] / total["matched_boxes"], 2) if total["matched_boxes"] else None
    total["single_id_pct"] = round(100 * total["single_id_people"] / total["people"], 2) if total["people"] else None
    return dict(overall=total, per_sequence=per_sequence)
