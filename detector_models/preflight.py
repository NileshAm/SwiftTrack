"""Read-only dependency/file checks. It never downloads weights or runs inference."""

import argparse
import importlib.util
import json
from pathlib import Path

from Utils import config
from .registry import ORDER, SPECS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=config.PROJECT_ROOT / "outputs/crowded_detector_suite_v1")
    parser.add_argument("--deim-repo", type=Path, default=config.PROJECT_ROOT / "detector_models/official/DEIMv2")
    parser.add_argument("--deim-weights", type=Path, default=config.PROJECT_ROOT / "detector_models/weights/deimv2_dinov3_s_coco.pth")
    args = parser.parse_args(argv)
    checkpoint = config.PROJECT_ROOT / "outputs/detector_suite_v1/sources/tracktor/association/model.pt"
    report = {"dataset": {name: (config.DATASET_ROOT / config.DATASET_SPLIT / name).is_dir()
                          for name in config.VAL_SEQUENCES},
              "association_checkpoint": {"path": str(checkpoint), "exists": checkpoint.is_file()},
              "models": {}}
    for name in ORDER:
        spec = SPECS[name]
        item = {"official_repo": spec.official_repo, "checkpoint": spec.checkpoint,
                "runtime": spec.runtime, "ready": False}
        if name == "crowddet": item["reason"] = spec.availability_note
        elif name == "iter_e2edet": item["reason"] = spec.availability_note
        elif name == "deimv2_s":
            item.update(repo_exists=(args.deim_repo / "engine").is_dir(), weights_exist=args.deim_weights.is_file())
            item["ready"] = item["repo_exists"] and item["weights_exist"]
        
        elif spec.runtime == "rfdetr":
            item["ready"] = (
                importlib.util.find_spec("rfdetr") is not None
            )
            item["reason"] = (
                "Package available; checkpoint loading and "
                "inference still need verification."
                if item["ready"]
                else "Install official rfdetr package."
            )

        else:
            item.update(ultralytics_installed=importlib.util.find_spec("ultralytics") is not None)
            item["ready"] = item["ultralytics_installed"]
        report["models"][name] = item
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); print(f"Saved {path}")


if __name__ == "__main__":
    main()

