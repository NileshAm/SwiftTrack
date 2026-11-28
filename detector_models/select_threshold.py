"""Choose one declared operating point from a detector diagnostic report."""

import argparse
import json
from pathlib import Path


def choose(report, min_precision=.70):
    data = json.loads(Path(report).read_text())
    rows = [row for row in data["results"] if row["sequence"] == "COMBINED"
            and row["iou"] == .5 and row["score"] != "all"]
    eligible = [row for row in rows if row["precision"] is not None and row["precision"] >= min_precision]
    pool = eligible or rows
    # Balanced operating point first; recall breaks effectively equal F1 values.
    winner = max(pool, key=lambda row: (row["f1"] or -1, row["recall"] or -1, row["precision"] or -1))
    return {"policy": "maximum F1 subject to minimum precision; fallback maximum F1",
            "minimum_precision": min_precision, "constraint_satisfied": winner in eligible,
            "operating_point": winner}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-precision", type=float, default=.70)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    selected = choose(args.report, args.min_precision)
    output = args.output or args.report.with_name("selected_operating_point.json")
    output.write_text(json.dumps(selected, indent=2) + "\n")
    print(f"Selected score {selected['operating_point']['score']}: {output}")


if __name__ == "__main__":
    main()

