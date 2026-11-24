"""Score GT, Tracktor, YOLOX or Iter tracker outputs with MOT TrackEval.

Run after infer.py: python Baselines/OurImplementation/evaluate.py RUN_DIR
Uses the same HOTA, CLEAR and Identity metric implementations as the authors'
baseline wrapper. These are local validation scores, not MOTChallenge test scores.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from Utils.data import _sequence_info  # noqa: E402
from Utils import mot_metrics  # noqa: E402
from evaluate import copy_mot_slice  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Fresh output directory used by infer.py")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("implementation") != "local_association_model":
        raise ValueError("This evaluator requires the local implementation's inference manifest")
    if manifest.get("gt_ids_passed_to_tracker") is not False:
        raise ValueError("Inference manifest does not confirm that GT IDs were hidden")
    root = Path(manifest["dataset_root"]) / manifest["dataset_split"]
    sequences = tuple(manifest["sequences"])
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError("Inference manifest needs distinct validation videos")
    output = run / "evaluation"
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    output.mkdir(parents=True)
    scores = {}
    for detector in manifest["detectors"]:
        folder = run / detector
        inputs = folder / "evaluation/val/inputs/MOT17"
        seq_info = {}
        for sequence in sequences:
            sequence_path = root / sequence
            length = _sequence_info(sequence_path).getint("seqLength")
            seq_info[sequence] = length
            track_dir = folder / "tracking/val" / sequence
            summary = json.loads((track_dir / "summary.json").read_text())
            source = manifest["sources"][detector][sequence]["file"]
            if (summary.get("complete") is not True or summary.get("sequence") != sequence
                    or summary.get("split") != "val" or summary.get("frames") != length
                    or summary.get("frame_range") != [1, length]
                    or summary.get("detection_file") != source
                    or summary.get("use_graph") != (manifest["graph"] == "on")):
                raise ValueError(f"Incomplete or mismatched tracker output: {track_dir}")
            gt_target = inputs / "gt" / sequence / "gt/gt.txt"
            track_target = inputs / "trackers/DragonTrack/data" / f"{sequence}.txt"
            copy_mot_slice(sequence_path / "gt/gt.txt", gt_target, (1, length), is_gt=True)
            count = copy_mot_slice(track_dir / "tracks.txt", track_target, (1, length), is_gt=False)
            if count != summary.get("mot_rows"):
                raise ValueError(f"Tracking row count changed: {track_dir}")
        data = mot_metrics.load(folder, seq_info)
        scores[detector] = {
            "COMBINED": mot_metrics.score(data),
            "per_sequence": {name: mot_metrics.score({name: values}) for name, values in data.items()},
            "association_only": mot_metrics.association_only(data),
        }
    (output / "results.json").write_text(json.dumps({
        "protocol": "TrackEval MOT17 pedestrian validation with MOT preprocessing",
        "implementation": manifest["implementation"],
        "checkpoint": manifest["checkpoint"], "manifest": str(run / "manifest.json"),
        "scores": scores,
    }, indent=2) + "\n")
    lines = ["# Our implementation: full-video MOT17 validation", "",
             "GT IDs were hidden from tracking. GT boxes are an upper control, not a deployable input.",
             "These validation videos were used during model development; these are not official test results.",
             "", "| Boxes | HOTA | DetA | AssA | IDF1 | MOTA | FP | FN | ID switches |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for detector, result in scores.items():
        row = result["COMBINED"]
        lines.append("| " + " | ".join([detector] + [str(row[key]) for key in
            ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "CLR_FP", "CLR_FN", "IDSW")]) + " |")
    lines += ["", "Per-video and association-only diagnostics are in `results.json`.",
              "Inference settings, detection hashes and checkpoint details are in `../manifest.json`.", ""]
    (output / "results.md").write_text("\n".join(lines))
    print(f"Our implementation evaluation: {output / 'results.md'}")
    return output


if __name__ == "__main__":
    main()
