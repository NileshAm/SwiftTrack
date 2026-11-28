"""Build the consolidated detector/tracker ranking without rerunning experiments."""

import argparse
import csv
import json
from pathlib import Path

from .registry import ORDER, SPECS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for name in ORDER:
        arm = args.output_root / name
        limitation = arm / "limitation.json"
        if limitation.is_file():
            item = json.loads(limitation.read_text())
            rows.append({"detector": name, "status": "unavailable", "reason": item["reason"]})
            continue
        detector_report = arm / "detector_evaluation/summary.json"
        selected_path = arm / "detector_evaluation/selected_operating_point.json"
        tracking_path = arm / "tracking_osnet_no_gcn/tracking_result.json"
        if not detector_report.is_file():
            rows.append({"detector": name, "status": "pending"})
            continue
        detector_data = json.loads(detector_report.read_text())
        selected = json.loads(selected_path.read_text())["operating_point"]
        localization = next((row for row in detector_data["results"]
                             if row["sequence"] == "COMBINED" and row["iou"] == .75
                             and str(row["score"]) == str(selected["score"])), None)
        source_summaries = [source["metadata"] for source in detector_data["sources"].values()]
        detector_fps = (sum(item["identity"]["frames"] for item in source_summaries) /
                        sum(item["seconds"] for item in source_summaries))
        tracking = json.loads(tracking_path.read_text()) if tracking_path.is_file() else {"metrics": {}}
        metrics = tracking["metrics"]
        rows.append({"detector": name, "status": "complete" if metrics else "detection_complete",
                     "threshold": selected["score"], "precision": selected["precision"],
                     "recall": selected["recall"], "f1": selected["f1"],
                     "f1_iou75": None if localization is None else localization["f1"],
                     "small_recall": selected["height_lt_50_recall"],
                     "occluded_recall": selected["visibility_lt_0.3_recall"],
                     "detector_fps": detector_fps, "tracker_fps": tracking.get("tracker_fps"),
                     **{key: metrics.get(key) for key in ("HOTA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN")}})
    fields = sorted({key for row in rows for key in row}, key=lambda x: (x != "detector", x))
    with (args.output_root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    ranked = sorted((r for r in rows if r.get("HOTA") is not None), key=lambda r: r["HOTA"], reverse=True)
    lines = ["# Crowded pedestrian detector comparison", "",
             "All completed tracking arms use frozen OSNet, the same association checkpoint, GCN disabled, "
             "MOT17-02-SDP and MOT17-09-SDP. No detector was trained by this suite.", "",
             "| Detector | Status | P | R | F1 | F1@.75 | Small R | Occluded R | Det FPS | Track FPS | HOTA | IDF1 | MOTA | IDSW | FP | FN |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def pct(value): return "" if value is None else f"{100*value:.2f}" if value <= 1 else f"{value:.3f}"
    for row in rows:
        lines.append("| " + " | ".join([SPECS[row["detector"]].title, row["status"],
            *[pct(row.get(k)) for k in ("precision", "recall", "f1", "f1_iou75", "small_recall", "occluded_recall")],
            *["" if row.get(k) is None else f"{row[k]:.2f}" for k in ("detector_fps", "tracker_fps")],
            *["" if row.get(k) is None else str(row[k]) for k in ("HOTA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN")]]) + " |")
        if row.get("reason"):
            lines.append(f"\n**{SPECS[row['detector']].title} limitation:** {row['reason']}")
    lines += ["", "## Fixed Tracktor baseline", "",
              "Tracktor + frozen OSNet without GCN: detection precision **99.61**, recall **51.46**; "
              "HOTA **44.206**, IDF1 **51.182**, MOTA **50.310**, IDSW **226**, FP **48**, FN **11605**.",
              "", "## Ranking and recommendation", ""]
    if ranked:
        lines += [f"1. **{SPECS[row['detector']].title}** — HOTA {row['HOTA']}, IDF1 {row['IDF1']}"
                  for row in ranked]
        lines += ["", "Fine-tune only after comparing detection recall at similar precision and end-to-end HOTA/IDF1. "
                  "A detector with the highest box F1 is not automatically the best tracking detector."]
    else:
        lines.append("No end-to-end tracking arm is complete yet.")
    (args.output_root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(args.output_root / "comparison.md")


if __name__ == "__main__":
    main()
