"""Score the authors' inference outputs with TrackEval on labelled MOT17 videos.

Run after infer.py: python Baselines/DragonTrackGithub/evaluate.py RUN_DIR
Scores are local validation measurements, not official MOTChallenge test scores.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import trackeval

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from Utils.data import _sequence_info  # noqa: E402


def score(sequences, data):
    result = {}
    metrics = (trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False}))
    for metric in metrics:
        individual = {sequence: metric.eval_sequence(data[sequence]) for sequence in sequences}
        combined = metric.combine_sequences(individual)
        for name, values in [("COMBINED", combined), *individual.items()]:
            row = result.setdefault(name, {})
            for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "CLR_TP", "CLR_FP", "CLR_FN", "IDSW"):
                if key not in values:
                    continue
                value = values[key]
                row[key] = (round(float(np.mean(value)) * 100, 3) if key in ("HOTA", "DetA", "AssA")
                            else round(float(value) * 100, 3) if key in ("IDF1", "MOTA")
                            else int(value))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("gt_ids_passed_to_tracker") is not False:
        raise ValueError("Missing identity-hiding provenance in inference manifest")
    root = Path(manifest["dataset_root"])
    sequences = tuple(manifest["sequences"])
    output = run / "evaluation"
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    output.mkdir(parents=True)
    results = {}
    for detector in manifest["detectors"]:
        folder = output / detector / "inputs/MOT17"
        seq_info = {}
        for sequence in sequences:
            sequence_path = root / sequence
            seq_info[sequence] = _sequence_info(sequence_path).getint("seqLength")
            gt_source = sequence_path / "gt/gt.txt"
            gt_target = folder / "gt" / sequence / "gt/gt.txt"
            gt_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(gt_source, gt_target)
            track_source = run / "tracks" / detector / f"{sequence}.txt"
            if not track_source.is_file():
                raise FileNotFoundError(track_source)
            track_target = folder / "trackers/DragonTrack/data" / f"{sequence}.txt"
            track_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(track_source, track_target)
        dataset = trackeval.datasets.MotChallenge2DBox(dict(
            GT_FOLDER=str(folder / "gt"), TRACKERS_FOLDER=str(folder / "trackers"),
            OUTPUT_FOLDER=str(output / detector / "MOT17"), TRACKERS_TO_EVAL=["DragonTrack"],
            BENCHMARK="MOT17", SPLIT_TO_EVAL="train", SKIP_SPLIT_FOL=True,
            SEQ_INFO=seq_info, DO_PREPROC=True, PRINT_CONFIG=False,
            CLASSES_TO_EVAL=["pedestrian"]))
        data = {sequence: dataset.get_preprocessed_seq_data(
            dataset.get_raw_seq_data("DragonTrack", sequence), "pedestrian") for sequence in sequences}
        results[detector] = score(sequences, data)
    (output / "results.json").write_text(json.dumps({
        "protocol": "TrackEval MOT17 pedestrian validation with official preprocessing",
        "upstream_commit": manifest["upstream_commit"],
        "checkpoint": manifest["checkpoint"], "scores": results,
    }, indent=2) + "\n")
    lines = ["# Authors' DragonTrack baseline evaluation", "",
             "Local MOT17 validation; not an official MOTChallenge test score.", "",
             "| Boxes | HOTA | DetA | AssA | IDF1 | MOTA | FP | FN | ID switches |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for detector, values in results.items():
        row = values["COMBINED"]
        lines.append("| " + " | ".join([detector] + [str(row[key]) for key in
            ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "CLR_FP", "CLR_FN", "IDSW")]) + " |")
    lines.extend(["", "Per-video scores and source/checkpoint identities are in `results.json` and `../manifest.json`.", ""])
    (output / "results.md").write_text("\n".join(lines))
    print(f"Authors' baseline evaluation: {output / 'results.md'}")
    return output


if __name__ == "__main__":
    main()
