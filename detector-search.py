"""Read saved detector sweeps, or train a small isolated hyperparameter grid.

Edit the settings below, then run python detector-search.py --mode reports|train.
Never edits Utils/config.py. See Documentation/Implementation/26-detector-search.md.
"""

import argparse
import csv
import fcntl
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from tqdm.auto import tqdm

from Utils import config
from Utils.detector_selection import select_operating_point, selection_key
from Utils.detector_training import build_datasets

# START HERE: scores are evaluated on the same predictions, not separate training runs.
MODE = 'reports'
REPORT_RUNS = sorted(config.PROJECT_ROOT.glob('outputs/detector*/settings.json'))
SEARCH_DIR = config.PROJECT_ROOT / 'outputs' / 'detector_search' / 'detector_search_lr_0.4'
EPOCHS = 3  # Independent total budget for each trial; early epochs performed best so far.
TRAIN_FRACTION = .4  # 25% of frames spread across EVERY training video; full validation.
SCORES = [.10, .15, .20, .25, .30, .35, .40, .50]  # Validation thresholds to sweep; never interpolate between them.
MIN_PRECISION = .70
MIN_RECALL = .50  # Increase to >.5146 to require more recall than the current Tracktor baseline.
MODEL_VARIANTS = [dict(name=config.DETECTOR_MODEL_NAME, revision=config.DETECTOR_REVISION)]
# Model and revision stay paired. Add a larger RT-DETRv2 variant only with its own pinned revision.
PARAM_GRID = {'lr': [1e-5, 2e-5, 5e-5], 'image_size': [640, 960], 'weight_decay': [1e-3,1e-4, 1e-5], 'accumulation': [2,4,8]}  # Independent hyperparameter grid for each model variant.
FIXED = {'batch_size': 2, 'frame_stride': 1}
BACKBONE_LR_RATIO = .1  # Override by adding backbone_lr to PARAM_GRID if needed.


def write_json(value, path):
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def checkpoint_epochs(folder):
    """Read actual saved checkpoint epochs; reports do not prove weights still exist."""
    available = {}
    for name in ('best.pt', 'last.pt'):
        path = folder/name
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            available[int(saved['epoch'])] = str(path)
            del saved
    return available


def collect_reports(folders, *, scores, min_precision, min_recall):
    """Rank only recorded thresholds; never interpolate nonexistent results/checkpoints."""
    rows = []
    validation_identity = None
    for folder in folders:
        settings = json.loads((folder/'settings.json').read_text())
        identity = {'val': settings['val'], 'box_origin': settings['box_origin']}
        if validation_identity is None:
            validation_identity = identity
        elif identity != validation_identity:
            raise ValueError('Reports use different validation videos/frames/GT; compare them separately')
        saved = checkpoint_epochs(folder)
        for path in sorted(folder.glob('validation_epoch_*.json')):
            report = json.loads(path.read_text())
            for source in report['results']:
                if source['sequence'] != 'COMBINED' or source['iou'] != .5 or source['score'] not in scores:
                    continue
                chosen = select_operating_point([source], sweep=True, min_precision=min_precision, min_recall=min_recall)
                rows.append({'run': str(folder), 'epoch': report['epoch'],
                             **{k: chosen[k] for k in ('score','precision','recall','f1','tp','fp','fn','eligible')},
                             'checkpoint': saved.get(report['epoch'], ''),
                             'model': settings['model'], 'lr': settings['settings']['lr'],
                             'backbone_lr': settings['settings']['backbone_lr'],
                             'image_size': settings['settings']['image_size'],
                             'report': str(path)})
    return sorted(rows, key=lambda r: (selection_key(r), r['recall'], r['precision'] or 0.), reverse=True)


def save_ranking(rows, folder):
    if not rows:
        raise ValueError('No saved rows match the requested thresholds')
    with (folder/'ranking.csv.pending').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder/'ranking.csv.pending').replace(folder/'ranking.csv')
    eligible = [r for r in rows if r['eligible']]
    retained = [r for r in eligible if r['checkpoint']]
    write_json({'best_eligible': eligible[0] if eligible else None,
                'best_available_checkpoint': retained[0] if retained else None,
                'best_fallback': rows[0] if not eligible else None,
                'note': 'Validation selection only. A report does not recreate discarded epoch weights.'}, folder/'best.json')
    lines = ['# Detector search results', '',
             'Ranks F1 at IoU 0.5, requiring the configured minimum precision and recall first. '
             'Raw number of predictions is not the objective. Ineligible rows are fallback diagnostics.', '',
             '| Run | Epoch | Score | Precision | Recall | F1 | Eligible | Weights retained |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows[:20]:
        numbers = ['n/a' if row[k] is None else f'{row[k]*100:.2f}%' for k in ('precision','recall','f1')]
        lines.append(f"| {Path(row['run']).name} | {row['epoch']} | {row['score']} | " + ' | '.join(numbers) +
                     f" | {row['eligible']} | {bool(row['checkpoint'])} |")
    lines += ['', 'Existing report mode only evaluates scores already recorded. Training mode evaluates the configured '
              'score grid each epoch and retains its selected best weights. Epoch zero is the pretrained baseline.', '',
              'These are detector diagnostics, not HOTA/IDF1/COCO AP. Validate the chosen model once on an untouched '
              'labeled test set before treating it as a generalization estimate.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n')
    top = eligible[0] if eligible else rows[0]
    tqdm.write(f"Best {'eligible' if eligible else 'INELIGIBLE fallback'}: epoch={top['epoch']} score={top['score']} "
               f"F1={top['f1']:.4f} recall={top['recall']:.4f}; retained weights={bool(top['checkpoint'])}")


def train_grid(args):
    allowed = {'lr','backbone_lr','image_size','batch_size','accumulation','weight_decay','frame_stride'}
    if not PARAM_GRID or (set(PARAM_GRID) | set(FIXED))-allowed:
        raise ValueError('Unsupported/empty training grid')
    if any(not isinstance(v,(list,tuple)) or not v for v in PARAM_GRID.values()):
        raise ValueError('Each grid parameter needs a nonempty list of values')
    if any(not variant.get('name') or not variant.get('revision') for variant in MODEL_VARIANTS) or not MODEL_VARIANTS:
        raise ValueError('Each model variant needs its own model name and pinned revision')
    # Full validation and exact GT metadata are part of the search identity.
    train_data, val_data = build_datasets(1, args.max_frames, train_fraction=args.train_fraction)
    tqdm.write(f'Search data: {len(train_data)} training frames before any trial stride, '
               f'{len(val_data)} validation frames; training fraction={args.train_fraction:g}')
    data = {'train': train_data.sources, 'val': val_data.sources, 'box_origin': config.BOX_ORIGIN,
            'seed': config.SEED, 'ignore_iou': config.DETECTOR_TRAIN_IGNORE_IOU}
    del train_data, val_data
    trials = []
    for variant in MODEL_VARIANTS:
        for values in product(*PARAM_GRID.values()):
            settings = {**FIXED, **dict(zip(PARAM_GRID, values, strict=True))}
            settings.setdefault('backbone_lr', settings['lr']*BACKBONE_LR_RATIO)
            trials.append({'model': variant, 'settings': settings})
    code = {name: sha256((config.PROJECT_ROOT/name).read_bytes()).hexdigest() for name in
            ('detector-train.py','Utils/detector_training.py','Utils/detector_selection.py','evaluate_detector.py')}
    manifest = {'version': 1, 'epochs': args.epochs, 'scores': args.scores,
                'min_precision': args.min_precision, 'min_recall': args.min_recall,
                'max_frames': args.max_frames, 'train_fraction': args.train_fraction,
                'sampling': 'evenly_spaced_per_video_v1', 'data': data, 'code': code, 'trials': trials}
    path = args.output_dir/'search.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Search grid/code/data changed. Choose a new --output-dir; existing trials preserved.')
    else:
        if any(p.name != '.lock' for p in args.output_dir.iterdir()):
            raise FileExistsError('Choose an empty search output directory')
        write_json(manifest, path)
    completed = []
    with tqdm(total=len(trials), desc='Detector search overall', unit='trial', smoothing=0,
              disable=not config.SHOW_PROGRESS) as bar:
        for index, trial in enumerate(trials, 1):
            folder = args.output_dir/f'trial_{index:03d}'
            done = folder/'search_complete.json'
            if done.exists():
                state = json.loads(done.read_text())
                from Utils.detections import file_digest
                for name, digest in state['files'].items():
                    if file_digest(folder/name) != digest:
                        raise ValueError(f'Completed trial files changed: {folder/name}')
                tqdm.write(f'Skipping completed trial {index}/{len(trials)}')
            else:
                command = [sys.executable, str(config.PROJECT_ROOT/'detector-train.py'),
                           '--output-dir', str(folder), '--epochs', str(args.epochs), '--device', args.device,
                           '--model-name', trial['model']['name'], '--revision', trial['model']['revision'],
                           '--selection-mode', 'sweep', '--score', str(args.scores[0]),
                           '--scores', *map(str,args.scores), '--min-precision', str(args.min_precision),
                           '--min-recall', str(args.min_recall), '--train-fraction', str(args.train_fraction)]
                for key, value in trial['settings'].items():
                    command += ['--'+key.replace('_','-'), str(value)]
                if args.max_frames is not None:
                    command += ['--max-frames', str(args.max_frames)]
                tqdm.write(f'Trial {index}/{len(trials)}: {trial}')
                bar.set_postfix(trial=index, refresh=False)
                # Separate process releases GPU memory after every trial and preserves config.py.
                subprocess.run(command, cwd=config.PROJECT_ROOT, check=True)
                from Utils.detections import file_digest
                files = ['best.pt','last.pt','best/training_manifest.json','history.csv']
                files += [p.name for p in folder.glob('validation_epoch_*.json')]
                write_json({'files': {name: file_digest(folder/name) for name in files}}, done)
            completed.append(folder)
            save_ranking(collect_reports(completed, scores=args.scores, min_precision=args.min_precision,
                                         min_recall=args.min_recall), args.output_dir)
            bar.update(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('reports','train'), default=MODE)
    parser.add_argument('--runs', type=Path, nargs='+', help='Existing detector run directories for reports mode')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--train-fraction', type=float, default=TRAIN_FRACTION,
                        help='Fraction of each training video sampled across time; validation stays full')
    parser.add_argument('--scores', type=float, nargs='+', default=SCORES)
    parser.add_argument('--min-precision', type=float, default=MIN_PRECISION)
    parser.add_argument('--min-recall', type=float, default=MIN_RECALL)
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--max-frames', type=int, help='Smoke check only; caps both training and validation')
    args = parser.parse_args(argv)
    if not 0 < args.train_fraction <= 1:
        parser.error('train-fraction must be in (0,1]')
    if args.epochs < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('epochs/max-frames must be positive')
    if any(not 0 < s < 1 for s in args.scores) or not 0 <= args.min_precision <= 1 or not 0 <= args.min_recall <= 1:
        parser.error('scores must be in (0,1); precision/recall floors in [0,1]')
    args.scores = sorted(set(args.scores))
    args.output_dir = (args.output_dir or (SEARCH_DIR if args.mode=='train' else
                       config.PROJECT_ROOT/'outputs'/'detector_search_reports'/str(time.time_ns()))).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another search is using this output directory') from None
        if args.mode == 'train':
            train_grid(args)
        else:
            if any(p.name != '.lock' for p in args.output_dir.iterdir()):
                raise FileExistsError('Choose a new report output directory')
            folders = args.runs or [p.parent for p in REPORT_RUNS]
            rows = collect_reports(folders, scores=args.scores, min_precision=args.min_precision, min_recall=args.min_recall)
            write_json({'runs': [str(p) for p in folders], 'scores': args.scores,
                        'min_precision': args.min_precision, 'min_recall': args.min_recall}, args.output_dir/'search.json')
            save_ranking(rows, args.output_dir)
    tqdm.write(f'Search report: {args.output_dir / "report.md"}')
    return args.output_dir


if __name__ == '__main__':
    main()
