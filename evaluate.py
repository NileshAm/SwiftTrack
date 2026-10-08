"""Run `python evaluate.py` after tracking the complete VAL_SEQUENCES videos.

Uses TrackEval's HOTA, CLEAR and Identity metrics with MOT preprocessing.
Only evaluation copies are cropped/rebased; source GT and predictions stay intact.
"""

from contextlib import redirect_stdout
import csv
import json
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

from Utils import config
from Utils.outputs import tracking_folder
from Utils.splits import full_frame_range, validate_video_splits, split_manifest


def copy_mot_slice(source, destination, frame_range, *, is_gt):
    """Copy selected MOT rows, shifting original frame IDs to start at 1.

    Preserve GT marks/classes/visibility so TrackEval applies its own rules.
    Prediction rows outside the requested split are errors, not silently removed.
    Empty prediction files are valid: the tracking summary proves frame coverage.
    """
    first, last = frame_range
    destination.parent.mkdir(parents=True, exist_ok=True)
    count, seen = 0, set()
    with Path(source).open(newline="") as input_file, destination.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        for line, row in enumerate(csv.reader(input_file), 1):
            if not row:
                continue
            required = 9 if is_gt else 7
            try:
                numbers = np.asarray(row, dtype=np.float64)
                if len(row) < required or not np.isfinite(numbers).all():
                    raise ValueError("missing or nonfinite values")
                frame_id, identity = int(numbers[0]), int(numbers[1])
                if numbers[0] != frame_id or frame_id < 1 or numbers[1] != identity:
                    raise ValueError("frame and identity must be integers; frame must be positive")
                if (numbers[4:6] <= 0).any():
                    raise ValueError("box width and height must be positive")
                if not is_gt:
                    if not first <= frame_id <= last:
                        raise ValueError("prediction frame is outside the selected split; rerun tracking")
                    if identity < 1 or (frame_id, identity) in seen:
                        raise ValueError("prediction IDs must be positive and unique in each frame")
                    seen.add((frame_id, identity))
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{source}, line {line}: {error}") from error
            if first <= frame_id <= last:
                row[0] = str(frame_id-first+1)
                writer.writerow(row)
                count += 1
    return count


def prepare_inputs(output_dir):
    """Check completed tracking coverage and prepare TrackEval's file layout."""
    groups, records = {}, []
    for sequence in tqdm(config.VAL_SEQUENCES, desc="Prepare evaluation", unit="sequence",
                         dynamic_ncols=True, disable=not config.SHOW_PROGRESS):
        benchmark = sequence.split("-", 1)[0]
        if benchmark not in ("MOT17", "MOT20"):
            raise ValueError("This evaluator currently supports MOT17 and MOT20 sequences")
        sequence_path = config.DATASET_ROOT / config.DATASET_SPLIT / sequence
        frame_range = full_frame_range(sequence_path)
        first, last = frame_range
        tracker_path = tracking_folder(sequence) / "tracks.txt"
        summary_path = tracker_path.with_name("summary.json")
        if not tracker_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing tracking output/summary for {sequence}. Run python tracking.py first.")
        summary = json.loads(summary_path.read_text())
        if (summary.get("complete") is not True or summary.get("sequence") != sequence
                or summary.get("split") != config.TRACK_SPLIT
                or summary.get("frame_range") != list(frame_range)
                or summary.get("frames") != last-first+1
                or summary.get("dataset_root") != str(Path(config.DATASET_ROOT).resolve())
                or summary.get("dataset_split") != config.DATASET_SPLIT
                or summary.get("data_split") != split_manifest()):
            raise ValueError(f"{sequence}: tracking coverage differs from this split. Rerun tracking in a new output folder.")
        root = output_dir / "inputs" / benchmark
        gt_path = sequence_path / config.GT_FILE
        gt_rows = copy_mot_slice(gt_path, root / "gt" / sequence / "gt" / "gt.txt", frame_range, is_gt=True)
        prediction_rows = copy_mot_slice(tracker_path, root / "trackers" / "DragonTrack" / "data" / f"{sequence}.txt",
                                        frame_range, is_gt=False)
        if prediction_rows != summary.get("mot_rows"):
            raise ValueError(f"{sequence}: MOT row count differs from its tracking summary")
        groups.setdefault(benchmark, {})[sequence] = last-first+1
        records.append({"sequence": sequence, "original_frame_range": frame_range,
                        "evaluation_frame_range": (1, last-first+1), "gt_rows": gt_rows,
                        "prediction_rows": prediction_rows, "tracking_summary": summary})
    return groups, records


def run_trackeval(output_dir, groups):
    """Call the installed official metrics and retain native summaries/logs."""
    if config.TRACKEVAL_ROOT is not None:
        root = Path(config.TRACKEVAL_ROOT)
        if not (root / "trackeval").is_dir():
            raise FileNotFoundError("TRACKEVAL_ROOT must contain the trackeval package directory")
        sys.path.insert(0, str(root.resolve()))
    summaries = {}
    # Native tables/import notices are saved; tqdm's stderr bar remains visible.
    with (output_dir / "trackeval.log").open("w") as log, redirect_stdout(log):
        try:
            import trackeval
        except ImportError as error:
            raise RuntimeError("TrackEval is missing. In dragon-track run: python -m pip install trackeval") from error
        print(f"TrackEval {getattr(trackeval, '__version__', 'source checkout')} from {trackeval.__file__}")
        # TrackEval appends Count after Identity. Update only after HOTA, CLEAR,
        # and Identity have all finished for a sequence; leave aggregation intact.
        with tqdm(total=sum(len(sequences) for sequences in groups.values()),
                  desc="Evaluation overall", unit="video", position=0, smoothing=0,
                  dynamic_ncols=True, disable=not config.SHOW_PROGRESS) as overall:
            class ProgressIdentity(trackeval.metrics.Identity):
                @staticmethod
                def get_name():
                    return "Identity"

                def eval_sequence(self, data):
                    result = super().eval_sequence(data)
                    overall.set_postfix(benchmark=benchmark, sequence=data["seq"], refresh=False)
                    overall.update(1)
                    return result

            for benchmark, sequences in groups.items():
                overall.set_postfix(benchmark=benchmark, phase="metrics", refresh=False)
                print(f"\nBenchmark: {benchmark}; split: {config.TRACK_SPLIT}; lengths: {sequences}")
                root = output_dir / "inputs" / benchmark
                dataset = trackeval.datasets.MotChallenge2DBox({
                    "GT_FOLDER": str(root / "gt"), "TRACKERS_FOLDER": str(root / "trackers"),
                    "OUTPUT_FOLDER": str(output_dir / benchmark), "TRACKERS_TO_EVAL": ["DragonTrack"],
                    "BENCHMARK": benchmark, "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True,
                    "SEQ_INFO": sequences, "DO_PREPROC": True, "PRINT_CONFIG": False,
                    "CLASSES_TO_EVAL": ["pedestrian"],
                })
                evaluator = trackeval.Evaluator({
                    "USE_PARALLEL": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                    "PRINT_RESULTS": True, "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True,
                    "PLOT_CURVES": False, "BREAK_ON_ERROR": True,
                    "LOG_ON_ERROR": str(output_dir / "trackeval_errors.log"),
                })
                metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                           ProgressIdentity({"PRINT_CONFIG": False})]
                _, messages = evaluator.evaluate([dataset], metrics, show_progressbar=config.SHOW_PROGRESS)
                if messages[dataset.get_name()]["DragonTrack"] != "Success":
                    raise RuntimeError("TrackEval did not complete; inspect trackeval.log")
                path = output_dir / benchmark / "DragonTrack" / "pedestrian_summary.txt"
                header, values = path.read_text().splitlines()[:2]
                summaries[benchmark] = dict(zip(header.split(), values.split(), strict=True))
        version = getattr(trackeval, "__version__", "source checkout")
    return summaries, version


def main():
    """Evaluate the same split/output used by tracking.py, never a full-video mismatch."""
    if config.TRACK_SPLIT != "val":
        raise ValueError("TRACK_SPLIT must be val; evaluation scores full VAL_SEQUENCES")
    validate_video_splits()
    output_dir = Path(config.EVALUATION_OUTPUT_DIR)
    if (output_dir / "results.md").exists():
        raise FileExistsError("Evaluation results already exist. Choose a new EVALUATION_OUTPUT_DIR to rerun.")
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, records = prepare_inputs(output_dir)
    print(f"Evaluating {config.TRACK_SPLIT}; full TrackEval tables/logs go to {output_dir}", flush=True)
    summaries, version = run_trackeval(output_dir, groups)
    report = [f"# Tracking evaluation — {config.TRACK_SPLIT}", "",
              f"TrackEval {version}; MOT preprocessing enabled; partial reproduction.",
              "Whole validation videos, separate from the training videos.",
              f"Training videos: {', '.join(config.TRAIN_SEQUENCES)}.", "",
              "| Sequence | Original frames |", "| --- | --- |"]
    report.extend(f"| {r['sequence']} | {r['original_frame_range'][0]}–{r['original_frame_range'][1]} |" for r in records)
    report.extend(["", "HOTA, IDF1 and MOTA are TrackEval percentage values; IDSW is a count.", "",
                   "| Benchmark | HOTA | IDF1 | MOTA | IDSW |", "| --- | --- | --- | --- | --- |"])
    for benchmark, values in summaries.items():
        report.append(f"| {benchmark} | {values['HOTA']} | {values['IDF1']} | {values['MOTA']} | {values['IDSW']} |")
        print(f"{benchmark}: HOTA={values['HOTA']}, IDF1={values['IDF1']}, "
              f"MOTA={values['MOTA']}, ID switches={values['IDSW']}", flush=True)
    report.extend(["", "Training and validation use separate complete videos.",
                   "Validation is used for checkpoint selection and tuning; this is not a final test score.",
                   "Native per-sequence and combined outputs are in each benchmark's DragonTrack folder."])
    (output_dir / "run_info.json").write_text(json.dumps({
        "trackeval_version": version, "split": config.TRACK_SPLIT,
        "data_split": split_manifest(),
        "records": records, "metrics": summaries,
    }, indent=2) + "\n")
    (output_dir / "results.md").write_text("\n".join(report) + "\n")
    if config.SAVE_PLOTS:
        from Utils.plots import auto_plot
        plot_paths = auto_plot("evaluation", output_dir)
        if plot_paths:
            with (output_dir / "results.md").open("a") as stream:
                stream.write("\n## Evaluation plots\n\n")
                for path in plot_paths:
                    stream.write(f"- [{path.parent.name}: {path.stem}]({path.relative_to(output_dir).as_posix()})\n")
            print(f"Plots: {output_dir / 'plots'}", flush=True)
    print(f"Report: {output_dir / 'results.md'}", flush=True)
    return summaries


if __name__ == "__main__":
    main()
