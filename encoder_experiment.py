"""Run matched encoder experiments without changing Utils/config.py.

Example: python encoder_experiment.py --encoder reid --stage smoke
Each encoder trains its own association model; never reuse another encoder's weights.
"""
import argparse
import importlib

from Utils import config


def configure(encoder, epochs=None):
    name, width = {
        "densenet": ("densenet121", 1024),
        "detr": ("detr_resnet50", 256),
        "reid": ("osnet_reid", 512),
    }[encoder]
    config.ENCODER_NAME, config.FEATURE_DIM = name, width
    config.FREEZE_ENCODER = config.USE_PRETRAINED_WEIGHTS = True
    config.RUN_NAME = f"encoder_comparison_{encoder}_v1"
    config.OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / config.RUN_NAME
    config.CHECKPOINT_PATH = config.OUTPUT_DIR / "model.pt"
    config.TRACK_OUTPUT_DIR = config.OUTPUT_DIR / "tracking" / config.TRACKING_VARIANT / config.TRACK_SPLIT
    config.EVALUATION_OUTPUT_DIR = config.OUTPUT_DIR / "evaluation" / config.TRACKING_VARIANT / config.TRACK_SPLIT
    config.TEST_OUTPUT_DIR = config.OUTPUT_DIR / "tracking" / config.TRACKING_VARIANT / "test"
    config.TEST_SEQUENCES = []
    config.RUN_SPLIT_TRACKING = True
    config.SAVE_VIDEO = config.SAVE_TEST_VIDEO = False
    if epochs is not None:
        config.EPOCHS = epochs


def smoke():
    import numpy as np
    import torch
    from Utils.data import load_sequence, read_image
    from Utils.detections import detection_path
    from Utils.features import load_encoder, extract_features
    from Utils.graph import build_graph
    from Utils.losses import compute_loss
    from model import AssociationModel

    path = config.DATASET_ROOT / config.DATASET_SPLIT / config.VAL_SEQUENCES[0]
    frames = load_sequence(path, detection_path(path), (1, 1), box_origin=config.BOX_ORIGIN,
                           min_score=config.DETECTION_MIN_SCORE)
    frame = frames[1]
    encoder = load_encoder()
    image = read_image(frame)
    features, valid = extract_features(image, frame["boxes"][:4], encoder, show_progress=False)
    boxes, features = frame["boxes"][:4][valid.numpy()], features[valid]
    assert len(features) > 0, "Smoke frame needs usable detections"
    assert features.shape == (len(boxes), config.FEATURE_DIM)
    assert torch.isfinite(features).all()
    # Identity labels here are synthetic: checks gradients, not tracking accuracy.
    graph = build_graph(boxes, boxes, features, features, image.shape[:2])
    model = AssociationModel().to(config.DEVICE)
    output = model(graph)
    truth = torch.eye(len(boxes))
    loss = compute_loss(output, truth, torch.ones_like(truth, dtype=torch.bool))["total"]
    loss.backward()
    assert torch.isfinite(loss), "Nonfinite graph loss"
    assert any(p.grad is not None for p in model.parameters())
    # Existing crop API must retain detection order, including invalid boxes.
    probe = np.vstack([boxes[0], [0, 0, 0, 0], boxes[0]])
    repeated, mask = extract_features(image, probe, encoder, show_progress=False)
    assert mask.tolist() == [True, False, True]
    assert torch.allclose(repeated[0], repeated[2], atol=1e-5)
    print(f"PASS: {config.ENCODER_NAME}, features={tuple(features.shape)}, "
          f"finite loss/backward, invalid-box mask, repeatability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("densenet", "detr", "reid"), required=True)
    parser.add_argument("--stage", choices=("smoke", "train", "track", "evaluate", "all"), default="smoke")
    parser.add_argument("--epochs", type=int, help="Total epoch target, same budget for each comparison")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    configure(args.encoder, args.epochs)
    print(f"Encoder: {config.ENCODER_NAME}; run: {config.OUTPUT_DIR}")
    if args.stage == "smoke":
        smoke()
        return
    stages = ("train", "track", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        importlib.import_module("tracking" if stage == "track" else stage).main()


if __name__ == "__main__":
    main()
