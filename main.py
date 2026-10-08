"""Save plots for an existing run: python main.py [--run-dir outputs/run_name].

Reads saved metrics only; never trains, tracks, or evaluates again.
"""

import argparse
from pathlib import Path

from Utils import config
from Utils.plots import save_run_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Existing run; default: config.OUTPUT_DIR")
    parser.add_argument("--evaluation-dir", type=Path, help="Optional custom evaluation folder")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir is not None else config.OUTPUT_DIR
    evaluation_dir = args.evaluation_dir
    if evaluation_dir is None and args.run_dir is None:
        evaluation_dir = config.EVALUATION_OUTPUT_DIR
    for path in save_run_plots(run_dir, evaluation_dir):
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
