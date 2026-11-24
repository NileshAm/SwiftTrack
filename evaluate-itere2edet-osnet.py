"""Run Iter-E2EDET boxes through fine-tuned OSNet without GCN, then TrackEval."""

import argparse
import importlib.util
from pathlib import Path

from Utils import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter-checkpoint", type=Path, required=True,
                        help="Same authors' CrowdHuman model.pth used during export")
    parser.add_argument("--cache-dir", type=Path,
                        default=config.PROJECT_ROOT / "detections/iter_e2edet_resolution_v2/higher_960_1600")
    parser.add_argument("--osnet-weights", type=Path,
                        default=config.PROJECT_ROOT / "outputs/osnet_finetune/lr_1e-05/best.pt")
    parser.add_argument("--association-checkpoint", type=Path,
                        default=config.PROJECT_ROOT / "outputs/encoder_comparison_reid_v1/model.pt")
    parser.add_argument("--detection-min-score", type=float, default=0.5)
    parser.add_argument("--new-track-min-score", type=float, default=0.7)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Check the cache and model on one frame without running evaluation")
    args = parser.parse_args(argv)
    path = Path(__file__).with_name("evaluate-tracker-variants.py")
    spec = importlib.util.spec_from_file_location("tracker_variant_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forwarded = ["--detections", "iter_e2edet", "--encoder", "finetuned_osnet",
                 "--graph", "off", "--iter-checkpoint", str(args.iter_checkpoint),
                 "--iter-cache-dir", str(args.cache_dir),
                 "--finetuned-osnet-weights", str(args.osnet_weights),
                 "--finetuned-osnet-association", str(args.association_checkpoint),
                 "--detection-min-score", str(args.detection_min_score),
                 "--new-track-min-score", str(args.new_track_min_score),
                 "--device", args.device]
    if args.output_dir is not None:
        forwarded += ["--output-dir", str(args.output_dir)]
    if args.smoke_test:
        forwarded.append("--smoke-test")
    return module.main(forwarded)


if __name__ == "__main__":
    main()
