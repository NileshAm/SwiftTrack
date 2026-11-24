"""Fine-tune the official CrowdHuman Iter-E2EDET on disjoint MOT17 videos.

Run with --help. The model and authors' matcher/loss are unchanged. Best model
selection uses validation detector F1 at IoU 0.5; no tracking is run here.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Utils import config as project_config
from Utils.data import load_ground_truth, load_image_frames, read_image
from Utils.detections import file_digest
from Utils.splits import validate_video_splits
from evaluate_detector import COUNT_FIELDS, metrics, score_frame


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/"ThirdParty/IterE2EDET/projects/crowd-e2e-sparse-rcnn"
VENDOR = ROOT/"ThirdParty/IterE2EDET"


def model_config(checkpoint, device, train_short, train_max, output_dir):
    """Load the authors' architecture and keep 960/1600 validation inference."""
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Missing vendored authors' source: {SOURCE}")
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(SOURCE))
    from config import add_sparsercnn_config
    from detectron2.config import get_cfg
    from models.detector import SparseRCNN  # noqa: F401: register architecture

    cfg = get_cfg()
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(str(SOURCE/"configs/50e.6h.500pro.ignore.yaml"))
    cfg.MODEL.WEIGHTS = str(checkpoint.resolve())
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MIN_SIZE_TRAIN = (train_short,)
    cfg.INPUT.MAX_SIZE_TRAIN = train_max
    cfg.INPUT.MIN_SIZE_TEST = 960
    cfg.INPUT.MAX_SIZE_TEST = 1600
    cfg.OUTPUT_DIR = str(output_dir.resolve())
    cfg.freeze()
    return cfg


def save_checkpoint(path, value):
    pending = path.with_name(path.name+".pending")
    torch.save(value, pending)
    pending.replace(path)


def validate(model, scores, device):
    """Full-video detector diagnostic with original-image boxes and no NMS."""
    from detectron2.data import transforms as T

    resize = T.ResizeShortestEdge([960, 960], 1600)
    counts = {(iou, score): dict.fromkeys(COUNT_FIELDS, 0)
              for iou in (0.5, 0.75) for score in scores}
    total = sum(len(load_image_frames(project_config.DATASET_ROOT/project_config.DATASET_SPLIT/name))
                for name in project_config.VAL_SEQUENCES)
    model.eval()
    with torch.no_grad(), tqdm(total=total, desc="Validate Iter-E2EDET", unit="frame") as bar:
        for name in project_config.VAL_SEQUENCES:
            path = project_config.DATASET_ROOT/project_config.DATASET_SPLIT/name
            frames = load_image_frames(path)
            truth = load_ground_truth(path)
            for frame_id, frame in frames.items():
                rgb = read_image(frame)
                resized = resize.get_transform(rgb).apply_image(rgb)
                tensor = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)), device=device)
                result = model([{"image": tensor, "height": frame["height"],
                                 "width": frame["width"]}])[0]["instances"].to("cpu")
                boxes = result.pred_boxes.tensor.numpy().copy()
                confidence = result.scores.numpy()
                classes = result.pred_classes.numpy()
                if not np.isfinite(boxes).all() or not np.isfinite(confidence).all():
                    raise FloatingPointError(f"Nonfinite detector output: {name} frame {frame_id}")
                boxes[:, 0::2] = boxes[:, 0::2].clip(0, frame["width"])
                boxes[:, 1::2] = boxes[:, 1::2].clip(0, frame["height"])
                valid = (classes == 0) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                for (iou, score), aggregate in counts.items():
                    selected = boxes[valid & (confidence >= score)]
                    observed = score_frame(selected, truth[frame_id], iou)
                    for key, value in observed.items():
                        aggregate[key] += value
                bar.update(1)
    rows = [{"iou": iou, "score": score, **metrics(count)}
            for (iou, score), count in counts.items()]
    chosen = max((row for row in rows if row["iou"] == 0.5),
                 key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return chosen, rows


def train_epoch(model, loader, optimizer, accumulation, device, amp, overall):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    with tqdm(loader, desc="Train Iter-E2EDET", unit="batch", leave=False) as bar:
        for index, batch in enumerate(bar):
            remaining = len(loader) - (index//accumulation)*accumulation
            group_size = min(accumulation, remaining)
            context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if amp else nullcontext())
            with context:
                losses = model(batch)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at batch {index+1}")
            (loss/group_size).backward()
            if (index+1) % accumulation == 0 or index+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            bar.set_postfix(loss=f"{total/(index+1):.3f}", refresh=False)
            overall.update(1)
    return total/len(loader)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"weights/iter_e2edet/model.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/itere2edet_mot17_finetune_v1")
    parser.add_argument("--epochs", type=int, default=5, help="Total target epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1, help="1 uses every training frame")
    parser.add_argument("--train-short-side", type=int, default=960)
    parser.add_argument("--train-max-size", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scores", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA bfloat16 autocast; keep off if unsupported")
    parser.add_argument("--resume", action="store_true", help="Continue from latest.pth in --output-dir")
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.accumulation < 1 or args.frame_stride < 1
            or args.train_short_side < 32 or args.train_max_size < args.train_short_side
            or not all(math.isfinite(x) and x > 0 for x in (args.lr, args.backbone_lr))
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or any(not math.isfinite(score) or not 0.05 <= score <= 1 for score in args.scores)):
        parser.error("Invalid epoch, batch, resolution, optimizer or score setting")
    if args.amp and (not args.device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
        parser.error("--amp requires a CUDA GPU with bfloat16 support")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not args.checkpoint.is_file():
        parser.error(f"Missing official checkpoint: {args.checkpoint}")
    validate_video_splits()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error(f"{root} is nonempty; use --resume or a new --output-dir")
    root.mkdir(parents=True, exist_ok=True)
    cfg = model_config(args.checkpoint, args.device, args.train_short_side, args.train_max_size, root)
    from dataset_mapper import SparseRCNNDatasetMapper
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from Utils.itere2edet_training import MOTIterDataset

    mapper = SparseRCNNDatasetMapper(cfg, is_train=True)
    dataset = MOTIterDataset(project_config.TRAIN_SEQUENCES, mapper, args.frame_stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=lambda batch: batch)
    recipe = {"source_checkpoint_sha256": file_digest(args.checkpoint),
              "config_sha256": hashlib.sha256(cfg.dump().encode()).hexdigest(),
              "train_sources": dataset.sources, "train_sequences": project_config.TRAIN_SEQUENCES,
              "val_sequences": project_config.VAL_SEQUENCES, "batch_size": args.batch_size,
              "accumulation": args.accumulation, "frame_stride": args.frame_stride,
              "train_short_side": args.train_short_side, "train_max_size": args.train_max_size,
              "val_short_side": 960, "val_max_size": 1600, "lr": args.lr,
              "backbone_lr": args.backbone_lr, "weight_decay": args.weight_decay,
              "scores": sorted(set(args.scores)), "seed": args.seed, "amp": args.amp}
    model = build_model(cfg)
    groups = []
    for backbone in (False, True):
        params = [value for name, value in model.named_parameters()
                  if value.requires_grad and ("backbone" in name) == backbone]
        if params:
            groups.append({"params": params, "lr": args.backbone_lr if backbone else args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    latest = root/"latest.pth"
    best = root/"best.pth"
    if args.resume:
        if not latest.is_file():
            parser.error(f"Missing resume checkpoint: {latest}")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["recipe"] != recipe:
            parser.error("Resume settings or training data changed; use a new output directory")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(args.device)
        start_epoch, best_f1 = saved["epoch"]+1, saved["best_f1"]
        history = saved["history"]
        selected = history[-1]
        published = torch.load(best, map_location="cpu", weights_only=False) if best.is_file() else None
        if published is None or selected["f1"] > published["selected"]["f1"]:
            if selected["f1"] != best_f1:
                raise ValueError("Missing selected best checkpoint; restore this run before resuming")
            save_checkpoint(best, {"model": saved["model"], "epoch": saved["epoch"],
                                   "selected": selected, "recipe": recipe})
    else:
        DetectionCheckpointer(model).load(str(args.checkpoint.resolve()))
        start_epoch, best_f1, history = 1, -1.0, []
        (root/"recipe.json").write_text(json.dumps(recipe, indent=2)+"\n")
    model.to(args.device)
    if start_epoch > args.epochs:
        print(f"Already trained through epoch {start_epoch-1}; raise --epochs to continue")
        return
    steps_per_epoch = len(loader)
    with tqdm(total=(args.epochs-start_epoch+1)*steps_per_epoch,
              desc="Iter-E2EDET fine-tuning", unit="batch") as overall:
        for epoch in range(start_epoch, args.epochs+1):
            loss = train_epoch(model, loader, optimizer, args.accumulation, args.device, args.amp, overall)
            selected, rows = validate(model, recipe["scores"], args.device)
            selected = {"epoch": epoch, "train_loss": loss, **selected}
            history.append(selected)
            (root/f"validation_epoch_{epoch}.json").write_text(json.dumps(rows, indent=2)+"\n")
            improved = selected["f1"] > best_f1
            if improved:
                best_f1 = selected["f1"]
            save_checkpoint(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                     "epoch": epoch, "best_f1": best_f1, "history": history,
                                     "recipe": recipe})
            if improved:
                save_checkpoint(best, {"model": model.state_dict(), "epoch": epoch,
                                       "selected": selected, "recipe": recipe})
            (root/"history.json").write_text(json.dumps(history, indent=2)+"\n")
            tqdm.write(f"Epoch {epoch}: train loss {loss:.4f}, validation F1 {selected['f1']:.4f} "
                       f"at score {selected['score']:.2f}; best F1 {best_f1:.4f}")
    print(f"Best model: {best}")
    print(f"Latest training state: {latest}")


if __name__ == "__main__":
    main()
