"""Fine-tune RT-DETRv2 on TRAIN_SEQUENCES and evaluate VAL_SEQUENCES each epoch.

Run: python detector-train.py
Settings live in Utils/config.py under DETECTOR FINE-TUNING; CLI overrides are temporary.
See Documentation/Implementation/25-detector-training.md.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from Utils import config
from Utils.detector_training import build_datasets, DetectionCollator, install_mot_loss, move_batch
from Utils.detections import file_digest
from Utils.detector_selection import select_operating_point, selection_key
from evaluate_detector import COUNT_FIELDS, score_frame, metrics

VERSION = 'mot_rtdetr_finetune_v1'


def atomic_save(value, path):
    pending = path.with_name(path.name + '.pending')
    torch.save(value, pending)
    pending.replace(path)


def write_json(value, path):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed; resume with the original device setup')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def train_epoch(model, loader, optimizer, device, accumulation, overall):
    """AdamW updates with correctly sized final accumulation group and finite-gradient checks."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.
    samples = 0
    with tqdm(loader, desc='Train detector', unit='batch', position=1, leave=False,
              disable=not config.SHOW_PROGRESS) as bar:
        for index, (batch, records) in enumerate(bar):
            count = len(records)
            group_start = index // accumulation * accumulation
            # Weight by images, including a short final minibatch and accumulation group.
            group_images = min(accumulation * loader.batch_size, len(loader.dataset)-group_start*loader.batch_size)
            output = model(**move_batch(batch, device))
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite detector training loss; no checkpoint saved for this epoch')
            (loss * count / group_images).backward()
            loss_total += loss.item()*count
            samples += count
            if (index+1) % accumulation == 0 or index+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            bar.set_postfix(loss=f'{loss_total/samples:.4f}', refresh=False)
            overall.update(1)
    return loss_total/samples


@torch.no_grad()
def validate_epoch(model, processor, loader, device, person_id, threshold, overall,
                   *, scores=None, sweep=False, min_precision=0., min_recall=0.):
    """Validation loss and per-sequence detection counts, using original untouched GT."""
    model.eval()
    thresholds = sorted(set((scores or [.1, .2, .3, .4, .5, .6, .7]) + [threshold]))
    totals = {}
    loss_sum, samples = 0., 0
    with tqdm(loader, desc='Validate detector', unit='batch', position=1, leave=False,
              disable=not config.SHOW_PROGRESS) as bar:
        for batch, records in bar:
            output = model(**move_batch(batch, device))
            if not torch.isfinite(output.loss):
                raise FloatingPointError('Nonfinite detector validation loss')
            loss_sum += output.loss.item()*len(records)
            samples += len(records)
            sizes = torch.tensor([[frame['height'], frame['width']] for _, frame in records], device=device)
            results = processor.post_process_object_detection(output, target_sizes=sizes,
                        threshold=0., use_focal_loss=model.config.use_focal_loss)
            for result, (gt, frame) in zip(results, records, strict=True):
                selected = result['labels'] == person_id
                boxes = result['boxes'][selected].float().cpu().numpy()
                scores = result['scores'][selected].float().cpu().numpy()
                if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite detector predictions')
                boxes[:, 0::2] = boxes[:, 0::2].clip(0, frame['width'])
                boxes[:, 1::2] = boxes[:, 1::2].clip(0, frame['height'])
                valid = ((boxes[:, 2:]-boxes[:, :2]) > 1e-5).all(axis=1)
                for iou in (.5, .75):
                    for score in thresholds:
                        counts = score_frame(boxes[valid & (scores >= score)], gt, iou)
                        key = (frame['sequence'], iou, score)
                        total = totals.setdefault(key, dict.fromkeys(COUNT_FIELDS, 0))
                        for field, value in counts.items():
                            total[field] += value
            overall.update(1)
    rows = [{'sequence': seq, 'iou': iou, 'score': score, **metrics(counts)}
            for (seq, iou, score), counts in totals.items()]
    for iou in (.5, .75):
        for score in thresholds:
            total = {field: sum(c[field] for (seq, i, s), c in totals.items() if (i, s)==(iou, score))
                     for field in COUNT_FIELDS}
            rows.append({'sequence': 'COMBINED', 'iou': iou, 'score': score, **metrics(total)})
    primary = select_operating_point(rows, score=threshold, sweep=sweep,
                                     min_precision=min_precision, min_recall=min_recall)
    if primary['gt'] == 0 or primary['f1'] is None:
        raise ValueError('Validation contains no usable GT or detections')
    return loss_sum/samples, primary, rows


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
        axes[1].plot([r['epoch'] for r in history], [np.nan if r[field] is None else r[field] for r in history], label=field)
    axes[0].set(title='Detector loss (train includes denoising)', xlabel='Epoch', ylabel='Loss')
    axes[1].set(title='Selected validation operating point / IoU 0.5', xlabel='Epoch', ylabel='Rate', ylim=(0,1))
    for ax in axes: ax.legend()
    fig.tight_layout()
    fig.savefig(output/'training.png', dpi=160)
    plt.close(fig)


def export_best(model, processor, output):
    """Export the best detector in Hugging Face format, with an integrity manifest written last."""
    saved = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    folder = output/'best'
    folder.mkdir(exist_ok=True)
    model.save_pretrained(folder)
    processor.save_pretrained(folder)
    files = {p.name: file_digest(p) for p in folder.iterdir() if p.name in
             ('config.json', 'preprocessor_config.json', 'model.safetensors')}
    if len(files) != 3:
        raise RuntimeError('Expected a single safetensors model export with model/processor config')
    write_json({'version': VERSION, 'epoch': saved['epoch'], 'best_f1': saved['best_f1'],
                'selection': saved.get('best_selection', {'score': saved['recipe']['settings']['score'],
                                                         'f1': saved['best_f1'], 'eligible': True}),
                'recipe': saved['recipe'], 'files': files}, folder/'training_manifest.json')
    tqdm.write(f"Best detector: {folder} (epoch {saved['epoch']}, validation F1={saved['best_f1']:.4f})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=config.DETECTOR_TRAIN_OUTPUT_DIR)
    parser.add_argument('--epochs', type=int, default=config.DETECTOR_TRAIN_EPOCHS)
    parser.add_argument('--image-size', type=int, default=config.DETECTOR_TRAIN_IMAGE_SIZE)
    parser.add_argument('--batch-size', type=int, default=config.DETECTOR_TRAIN_BATCH_SIZE)
    parser.add_argument('--accumulation', type=int, default=config.DETECTOR_TRAIN_ACCUMULATION)
    parser.add_argument('--lr', type=float, default=config.DETECTOR_TRAIN_LR)
    parser.add_argument('--backbone-lr', type=float, default=config.DETECTOR_TRAIN_BACKBONE_LR)
    parser.add_argument('--weight-decay', type=float, default=config.DETECTOR_TRAIN_WEIGHT_DECAY)
    parser.add_argument('--frame-stride', type=int, default=config.DETECTOR_TRAIN_FRAME_STRIDE)
    parser.add_argument('--train-fraction', type=float, default=1.,
                        help='Fraction of frames spread across each training video; validation unchanged')
    parser.add_argument('--score', type=float, default=config.DETECTOR_TRAIN_SCORE)
    parser.add_argument('--selection-mode', choices=('fixed', 'sweep'), default='fixed')
    parser.add_argument('--scores', type=float, nargs='+', help='Validation confidence grid, no extra model inference')
    parser.add_argument('--min-precision', type=float, default=0.)
    parser.add_argument('--min-recall', type=float, default=0.)
    parser.add_argument('--model-name', default=config.DETECTOR_MODEL_NAME)
    parser.add_argument('--revision', default=None, help='Pinned revision required when changing the model')
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--max-frames', type=int, help='Smoke check only: first N selected frames per train/val video')
    args = parser.parse_args(argv)
    if not 0 < args.train_fraction <= 1:
        parser.error('train-fraction must be in (0,1]')
    if args.revision is None:
        if args.model_name != config.DETECTOR_MODEL_NAME:
            parser.error('Specify --revision for a different model; do not reuse another model revision')
        args.revision = config.DETECTOR_REVISION
    if not (0 <= args.min_precision <= 1 and 0 <= args.min_recall <= 1):
        parser.error('minimum precision/recall must be in [0,1]')
    if args.scores is not None:
        if any(not 0 < s < 1 for s in args.scores):
            parser.error('scores must be in (0,1)')
        args.scores = sorted(set(args.scores))
    if any(v < 1 for v in (args.epochs, args.batch_size, args.accumulation, args.frame_stride)):
        parser.error('epochs, batch-size, accumulation and frame-stride must be positive')
    if args.image_size < 128 or args.image_size % 32:
        parser.error('image-size must be a multiple of 32, at least 128 for 300 queries')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('max-frames must be positive')
    if not (0 < args.score < 1 and 0 < config.DETECTOR_TRAIN_IGNORE_IOU <= 1):
        parser.error('score must be in (0,1); ignore IoU must be in (0,1]')
    if not (math.isfinite(args.lr) and math.isfinite(args.backbone_lr) and math.isfinite(args.weight_decay)
            and args.lr > 0 and args.backbone_lr > 0 and args.weight_decay >= 0):
        parser.error('learning rates must be finite and positive; weight-decay must be finite and nonnegative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu only for a small smoke check.')
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    train_data, val_data = build_datasets(args.frame_stride, args.max_frames, train_fraction=args.train_fraction)
    tqdm.write(f'Detector frames: {len(train_data)} training, {len(val_data)} validation '
               f'(training fraction={args.train_fraction:g}, stride={args.frame_stride})')
    new_options = ('selection_mode', 'scores', 'min_precision', 'min_recall', 'model_name', 'revision',
                   'train_fraction')
    recipe = {'version': VERSION, 'model': args.model_name, 'revision': args.revision,
              'settings': {k: v for k, v in vars(args).items() if k not in ('output_dir','epochs','device', *new_options)},
              'seed': config.SEED, 'box_origin': config.BOX_ORIGIN, 'ignore_iou': config.DETECTOR_TRAIN_IGNORE_IOU,
              'augmentation': 'train_horizontal_flip_0.5', 'train': train_data.sources, 'val': val_data.sources}
    # Leave legacy fixed-threshold recipes unchanged so existing runs still resume.
    if args.train_fraction != 1.:
        recipe['training_subset'] = {'fraction': args.train_fraction, 'sampling': 'evenly_spaced_per_video_v1'}
    if args.selection_mode != 'fixed' or args.scores is not None or args.min_precision or args.min_recall:
        recipe['selection'] = {k: getattr(args, k) for k in new_options[:4]}
    validation_options = dict(scores=args.scores, sweep=args.selection_mode=='sweep',
                              min_precision=args.min_precision, min_recall=args.min_recall)
    output = args.output_dir
    candidates = []
    for name in ('last.pt', 'best.pt'):
        if (output/name).is_file():
            saved = torch.load(output/name, map_location='cpu', weights_only=True)
            if saved['recipe'] != recipe:
                raise ValueError('Detector settings/data changed. Choose a fresh --output-dir; old run preserved.')
            candidates.append(saved)
    if output.exists() and any(output.iterdir()) and not candidates:
        # A stopped baseline may have written only its matching settings file.
        baseline_only = ({p.name for p in output.iterdir()} == {'settings.json'}
                         and json.loads((output/'settings.json').read_text()) == recipe)
        if not baseline_only:
            raise FileExistsError('Nonempty detector output without resumable checkpoint. Choose a fresh --output-dir.')
    resume = max(candidates, key=lambda s: s['epoch']) if candidates else None
    if resume is not None and not (output/'best.pt').is_file():
        raise FileNotFoundError('best.pt is missing; restore it before resuming')
    if resume is not None:
        best = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        if (best['best_f1'] != resume['best_f1']
                or resume['best_f1'] != max(resume['history'], key=selection_key)['f1']):
            raise ValueError('Best/latest detector checkpoints disagree; restore files from the same run')
        del best
    processor = RTDetrImageProcessor.from_pretrained(args.model_name, revision=args.revision,
                                                    size={'height': args.image_size, 'width': args.image_size})
    model = RTDetrV2ForObjectDetection.from_pretrained(args.model_name, revision=args.revision,
                                                      use_pretrained_backbone=False).to(device)
    model.config.anchor_image_size = None
    model.model.config.anchor_image_size = None
    model.model.encoder.eval_size = None
    model.config.eval_size = None
    person_ids = [int(k) for k, v in model.config.id2label.items() if v.lower()=='person']
    if len(person_ids) != 1:
        raise ValueError('Pretrained model must have exactly one person class')
    install_mot_loss(model, config.DETECTOR_TRAIN_IGNORE_IOU)
    backbone, other = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (backbone if name.startswith('model.backbone.') else other).append(param)
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.backbone_lr},
                                  {'params': other, 'lr': args.lr}], weight_decay=args.weight_decay)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=DetectionCollator(processor, person_ids[0], augment=True))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=DetectionCollator(processor, person_ids[0]))
    history, best_f1, start = [], -1., 1
    best_selection = {'f1': -1., 'eligible': False, 'score': args.score}
    if resume is not None:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        history, best_f1, start = resume['history'], resume['best_f1'], resume['epoch']+1
        best_selection = resume.get('best_selection', {'f1': best_f1, 'eligible': True, 'score': args.score})
        restore_rng(resume['rng'])
        tqdm.write(f'Resuming detector at epoch {start}; total target {args.epochs}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(recipe, output/'settings.json')
    tqdm.write(f'Training frames: {len(train_data)}; validation frames: {len(val_data)}; input: {args.image_size}')
    steps = len(train_loader)+len(val_loader)
    with tqdm(total=max(0, args.epochs-start+1)*steps + (len(val_loader) if resume is None else 0),
              desc='Detector training overall', unit='batch',
              position=0, smoothing=0, disable=not config.SHOW_PROGRESS) as overall:
        if resume is None:
            overall.set_postfix(epoch='0 (pretrained baseline)', phase='validation', refresh=False)
            begin = time.perf_counter()
            val_loss, primary, rows = validate_epoch(model, processor, val_loader, device, person_ids[0], args.score,
                                                     overall, **validation_options)
            best_f1 = primary['f1']
            best_selection = {k: primary[k] for k in ('score', 'precision', 'recall', 'f1', 'eligible')}
            history.append({'epoch': 0, 'train_loss': None, 'val_loss': val_loss,
                            **{k: primary[k] for k in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')},
                            'seconds': time.perf_counter()-begin})
            if 'selection' in recipe:
                history[-1].update(selected_score=primary['score'], eligible=primary['eligible'])
            saved = {'recipe': recipe, 'epoch': 0, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'history': history, 'best_f1': best_f1, 'best_selection': best_selection, 'rng': rng_state()}
            atomic_save(saved, output/'best.pt')
            atomic_save(saved, output/'last.pt')
            write_json({'epoch': 0, 'selection_score': primary['score'], 'results': rows,
                        'metric': 'pretrained baseline diagnostic F1 at IoU 0.5'}, output/'validation_epoch_000.json')
            write_history(history, output)
            export_best(model, processor, output)
            tqdm.write(f'Pretrained baseline: validation F1={best_f1:.4f}')
        for epoch in range(start, args.epochs+1):
            begin = time.perf_counter()
            overall.set_postfix(epoch=f'{epoch}/{args.epochs}', phase='train', refresh=False)
            train_loss = train_epoch(model, train_loader, optimizer, device, args.accumulation, overall)
            overall.set_postfix(epoch=f'{epoch}/{args.epochs}', phase='validation', refresh=False)
            val_loss, primary, rows = validate_epoch(model, processor, val_loader, device, person_ids[0], args.score,
                                                     overall, **validation_options)
            row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                   **{k: primary[k] for k in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')},
                   'seconds': time.perf_counter()-begin}
            if 'selection' in recipe:
                row.update(selected_score=primary['score'], eligible=primary['eligible'])
            history.append(row)
            improved = selection_key(primary) > selection_key(best_selection)
            if improved:
                best_f1 = primary['f1']
                best_selection = {k: primary[k] for k in ('score', 'precision', 'recall', 'f1', 'eligible')}
            saved = {'recipe': recipe, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'history': history, 'best_f1': best_f1, 'best_selection': best_selection, 'rng': rng_state()}
            if improved:
                atomic_save(saved, output/'best.pt')
            atomic_save(saved, output/'last.pt')
            write_json({'epoch': epoch, 'selection_score': primary['score'], 'results': rows,
                        'metric': 'selected diagnostic F1 at IoU 0.5; not COCO AP or MOT tracking scores'},
                       output/f'validation_epoch_{epoch:03d}.json')
            write_history(history, output)
            # Keep best/ usable even if the next epoch is interrupted.
            if improved:
                current = model.training
                export_best(model, processor, output)
                model.train(current)
            tqdm.write(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
                       f"precision={primary['precision']}, recall={primary['recall']:.4f}, F1={primary['f1']:.4f}, "
                       f"score={primary['score']}, eligible={primary['eligible']}")
    # Also repairs an interrupted export when rerun after reaching the epoch target.
    export_best(model, processor, output)
    return output


if __name__ == '__main__':
    main()
