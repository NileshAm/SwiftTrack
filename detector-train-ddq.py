"""Fine-tune DDQ in dragon-track using the winning full-run RT-DETR settings.

Run python detector-train-ddq.py. Config supplies the same train/val video lists;
CLI settings are independent and never edit config.py. See docs/28-ddq-training.
"""

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Utils import config
from Utils.ddq import DDQDetector, MODELS
from Utils.ddq_training import (DDQCollator, checkpoint_state, install_mot_loss,
                                load_training_state, total_loss, validation_forward,
                                enable_activation_checkpointing)
from Utils.detector_training import build_datasets
from Utils.detector_selection import select_operating_point, selection_key
from evaluate_detector import COUNT_FIELDS, metrics, score_frame

VERSION = 'mot_ddq_finetune_v1'


def atomic_save(value, path):
    pending = path.with_name(path.name+'.pending')
    torch.save(value, pending)
    pending.replace(path)


def write_json(value, path):
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def rng_state():
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume on the same CUDA device setup')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def train_epoch(model, loader, optimizer, accumulation, overall):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running, samples = 0., 0
    with tqdm(loader, desc='Train DDQ', unit='batch', position=1, leave=False,
              disable=not config.SHOW_PROGRESS) as bar:
        for index, (batch, records) in enumerate(bar):
            count = len(records)
            group_start = index // accumulation * accumulation
            group_images = min(accumulation*loader.batch_size,
                               len(loader.dataset)-group_start*loader.batch_size)
            data = model.data_preprocessor(batch, training=True)
            loss = total_loss(model(**data, mode='loss'))
            (loss*count/group_images).backward()
            running += loss.item()*count
            samples += count
            # Release forward tensors before constructing the next large DDQ graph.
            del data, loss
            if (index+1) % accumulation == 0 or index+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            bar.set_postfix(loss=f'{running/samples:.4f}', refresh=False)
            overall.update(1)
    return running/samples


@torch.no_grad()
def validate_epoch(model, loader, person_id, args, overall):
    model.eval()
    thresholds = sorted(set(args.scores+[args.score]))
    totals, running, samples = {}, 0., 0
    with tqdm(loader, desc='Validate DDQ', unit='batch', position=1, leave=False,
              disable=not config.SHOW_PROGRESS) as bar:
        for batch, records in bar:
            loss, predictions = validation_forward(model, batch)
            running += loss.item()*len(records)
            samples += len(records)
            for prediction, (gt, frame) in zip(predictions, records, strict=True):
                selected = prediction.labels == person_id
                boxes = prediction.bboxes[selected].float().cpu().numpy().copy()
                scores = prediction.scores[selected].float().cpu().numpy()
                if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite DDQ predictions')
                boxes[:, 0::2] = boxes[:, 0::2].clip(0, frame['width'])
                boxes[:, 1::2] = boxes[:, 1::2].clip(0, frame['height'])
                valid = ((boxes[:, 2:]-boxes[:, :2]) > 1e-5).all(axis=1)
                for iou in (.5, .75):
                    for score in thresholds:
                        counts = score_frame(boxes[valid & (scores >= score)], gt, iou)
                        total = totals.setdefault((frame['sequence'], iou, score), dict.fromkeys(COUNT_FIELDS, 0))
                        for field, value in counts.items():
                            total[field] += value
            overall.update(1)
    rows = [dict(sequence=seq, iou=iou, score=score, **metrics(counts))
            for (seq, iou, score), counts in totals.items()]
    for iou in (.5, .75):
        for score in thresholds:
            counts = {f: sum(c[f] for (_, i, s), c in totals.items() if (i, s)==(iou, score))
                      for f in COUNT_FIELDS}
            rows.append(dict(sequence='COMBINED', iou=iou, score=score, **metrics(counts)))
    primary = select_operating_point(rows, score=args.score, sweep=args.selection_mode=='sweep',
                                    min_precision=args.min_precision, min_recall=args.min_recall)
    if primary['gt'] == 0:
        raise ValueError('Validation has no usable GT')
    return running/samples, primary, rows


def write_history(history, output):
    with (output/'history.csv.pending').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output/'history.csv.pending').replace(output/'history.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for field in ('train_loss', 'val_loss'):
        axes[0].plot([r['epoch'] for r in history], [r[field] for r in history], label=field)
    for field in ('precision', 'recall', 'f1'):
        axes[1].plot([r['epoch'] for r in history], [r[field] for r in history], label=field)
    axes[0].set(title='DDQ loss (validation: distinct decoder only)', xlabel='Epoch', ylabel='Loss')
    axes[1].set(title='Selected validation operating point / IoU 0.5', xlabel='Epoch', ylabel='Rate', ylim=(0, 1))
    for ax in axes: ax.legend()
    fig.tight_layout()
    fig.savefig(output/'training.png', dpi=160)
    plt.close(fig)


def export_best(output):
    """Portable MMDetection state_dict; never replace the live training model."""
    saved = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    atomic_save(dict(state_dict=saved['model'], meta=dict(
        dataset_meta=dict(classes=saved['classes']), ddq_training=dict(
            recipe=saved['recipe'], epoch=saved['epoch'], selection=saved['best_selection']))),
        output/'best.pth')


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=MODELS, default='5scale')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=5, help='Total target; completed epochs resume')
    parser.add_argument('--image-size', type=int, default=960)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True,
                        help='Recompute activations on backward to reduce VRAM; keeps FP32 and batch settings')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--backbone-lr', type=float, default=5e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--train-fraction', type=float, default=1.)
    parser.add_argument('--score', type=float, default=.25)
    parser.add_argument('--selection-mode', choices=('fixed','sweep'), default='sweep')
    parser.add_argument('--scores', type=float, nargs='+', default=[.15,.2,.225,.25,.275,.3,.35,.4])
    parser.add_argument('--min-precision', type=float, default=.7)
    parser.add_argument('--min-recall', type=float, default=.55)
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--max-frames', type=int, help='Smoke check: first N selected frames per train/val video')
    parser.add_argument('--checkpoint', type=Path, help='Trusted initial DDQ checkpoint; resume uses output-dir')
    parser.add_argument('--weights-dir', type=Path, default=config.PROJECT_ROOT/'outputs'/'ddq_weights')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.accumulation, args.frame_stride) < 1:
        parser.error('epochs, batch-size, accumulation and frame-stride must be positive')
    if args.image_size < 128 or args.image_size % 32:
        parser.error('image-size must be a multiple of 32, at least 128')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('max-frames must be positive')
    if not 0 < args.train_fraction <= 1:
        parser.error('train-fraction must be in (0,1]')
    if any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.backbone_lr)) or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('learning rates must be finite and positive; weight decay finite and nonnegative')
    if any(not 0 < s < 1 for s in args.scores+[args.score]):
        parser.error('confidence thresholds must be in (0,1)')
    if not (0 <= args.min_precision <= 1 and 0 <= args.min_recall <= 1 and 0 < config.DETECTOR_TRAIN_IGNORE_IOU <= 1):
        parser.error('invalid precision/recall floor or ignore IoU')
    args.scores = sorted(set(args.scores))
    args.output_dir = args.output_dir or config.PROJECT_ROOT/'outputs'/f'detector_ddq_{args.variant}_mot17_v1'
    return args


def run(args):
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu only for a small check.')
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    train_data, val_data = build_datasets(args.frame_stride, args.max_frames, train_fraction=args.train_fraction)
    detector_args = argparse.Namespace(**vars(args), input_mode='square', short_side=800,
                                       max_size=1333, min_score=0., max_detections=300)
    detector = DDQDetector(detector_args, trainable=True)
    model = detector.model
    if args.activation_checkpointing:
        enable_activation_checkpointing(model)
    install_mot_loss(model, config.DETECTOR_TRAIN_IGNORE_IOU)
    recipe = dict(version=VERSION, detector=detector.recipe,
                  settings={k:v for k,v in vars(args).items() if k not in ('output_dir','epochs','device','checkpoint','weights_dir')},
                  seed=config.SEED, box_origin=config.BOX_ORIGIN, ignore_iou=config.DETECTOR_TRAIN_IGNORE_IOU,
                  augmentation='train_horizontal_flip_0.5', train=train_data.sources, val=val_data.sources)
    output = args.output_dir
    resume = None
    existing = {p.name for p in output.iterdir()}- {'.lock'}
    if 'last.pt' in existing:
        resume = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if resume['recipe'] != recipe:
            raise ValueError('DDQ settings/data changed. Choose a fresh --output-dir.')
        if not (output/'best.pt').is_file():
            raise FileNotFoundError('best.pt missing; restore it before resuming')
        best = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        if best['recipe'] != recipe or selection_key(best['best_selection']) < selection_key(resume['best_selection']):
            raise ValueError('Best/latest checkpoint mismatch; restore checkpoints from this run')
        # best may be one epoch ahead if interrupted between the two atomic writes.
        if best['epoch'] > resume['epoch']:
            resume = best
        elif best['best_selection'] != resume['best_selection']:
            raise ValueError('Best/latest checkpoint selection mismatch')
        del best
    elif existing:
        # Recover a baseline interrupted after best.pt was written, or before any weights.
        if 'best.pt' in existing:
            resume = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
            if resume['recipe'] != recipe:
                raise ValueError('DDQ settings/data changed; use a fresh --output-dir')
        elif existing != {'settings.json'} or json.loads((output/'settings.json').read_text()) != recipe:
            raise FileExistsError('Nonempty output without resumable DDQ checkpoint')
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith('backbone.') else other).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=backbone, lr=args.backbone_lr),
                                  dict(params=other, lr=args.lr)], weight_decay=args.weight_decay)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=DDQCollator(args.image_size, detector.person_id, augment=True))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=DDQCollator(args.image_size, detector.person_id))
    history, start, best_selection = [], 0, dict(f1=-1., eligible=False)
    if resume is not None:
        load_training_state(model, resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        history, start, best_selection = resume['history'], resume['epoch']+1, resume['best_selection']
        restore_rng(resume['rng'])
        del resume
    write_json(recipe, output/'settings.json')
    tqdm.write(f'DDQ {args.variant}: {len(train_data)} train / {len(val_data)} val frames; '
               f'{args.image_size}px; batch {args.batch_size} x accumulation {args.accumulation}; '
               f'next epoch {start}, target {args.epochs}')
    steps = max(0, args.epochs-max(1,start)+1)*(len(train_loader)+len(val_loader))
    steps += len(val_loader) if start==0 else 0
    with tqdm(total=steps, desc='DDQ training overall', unit='batch', position=0,
              smoothing=0, disable=not config.SHOW_PROGRESS) as overall:
        for epoch in range(start, args.epochs+1):
            begin = time.perf_counter()
            overall.set_postfix(epoch=f'{epoch}/{args.epochs}', phase='train' if epoch else 'baseline')
            train_loss = train_epoch(model, train_loader, optimizer, args.accumulation, overall) if epoch else None
            overall.set_postfix(epoch=f'{epoch}/{args.epochs}', phase='validation')
            val_loss, primary, rows = validate_epoch(model, val_loader, detector.person_id, args, overall)
            improved = selection_key(primary) > selection_key(best_selection)
            if improved:
                best_selection = {k:primary[k] for k in ('score','precision','recall','f1','eligible')}
            history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                                **{k:primary[k] for k in ('precision','recall','f1','tp','fp','fn','eligible')},
                                selected_score=primary['score'], seconds=time.perf_counter()-begin))
            saved = dict(recipe=recipe, epoch=epoch, model=checkpoint_state(model), optimizer=optimizer.state_dict(),
                         classes=list(model.dataset_meta['classes']), history=history,
                         best_selection=best_selection, rng=rng_state())
            if improved:
                atomic_save(saved, output/'best.pt')
            atomic_save(saved, output/'last.pt')
            write_json(dict(epoch=epoch, selection_score=primary['score'], results=rows,
                            metric='diagnostic detection F1 at IoU 0.5; not tracking metrics',
                            val_loss='distinct-decoder only; no dense/encoder/denoising losses'),
                       output/f'validation_epoch_{epoch:03d}.json')
            write_history(history, output)
            if improved:
                export_best(output)
            tqdm.write(f"Epoch {epoch}: F1={primary['f1']:.4f}, P={primary['precision']}, "
                       f"R={primary['recall']:.4f}, score={primary['score']}, eligible={primary['eligible']}")
    write_history(history, output)
    export_best(output)
    tqdm.write(f'Best DDQ export: {output / "best.pth"}; rerun with a higher --epochs target to continue.')
    return output


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another DDQ training process is using this output directory') from None
        return run(args)


if __name__ == '__main__':
    main()
