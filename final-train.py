"""Ordered final MOT17 experiments. Default previews commands; --run executes them."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from tqdm.auto import tqdm

from Utils import config
from Utils.splits import validate_video_splits

ROOT = Path(__file__).resolve().parent


def detector_flags(root, name):
    kind = name.split('_')[0]
    cache = root / 'caches' / name
    tuned = name.endswith('tuned')
    if kind == 'tracktor':
        return ['--tracktor-cache-dir', root / 'caches/tracktor']
    if kind == 'yolox':
        return ['--yolox-cache-dir', cache] + (['--yolox-checkpoint', root / 'models/yolox_tuned/best.pth'] if tuned else [])
    if kind == 'iter':
        weights = root / 'models/iter_tuned/best.pth' if tuned else ROOT / 'weights/iter_e2edet/model.pth'
        return ['--iter-cache-dir', cache, '--iter-checkpoint', weights]
    return ['--rtdetr-cache-dir', cache, '--rtdetr-image-size', 960, '--rtdetr-cache-min-score', .05] + (
        ['--rtdetr-weights-dir', root / 'models/rtdetr_tuned/best'] if tuned else [])


def build_stages(root, args):
    jobs = []
    models, results, caches = root / 'models', root / 'results', root / 'caches'
    sources = ('tracktor', 'yolox_pre', 'rtdetr_pre', 'iter_pre', 'yolox_tuned', 'rtdetr_tuned', 'iter_tuned')

    def add(name, script, *values):
        jobs.append((name, script, list(map(str, values))))

    def train_assoc(name, encoder, graph, detector='tracktor', tuned_osnet=False, score=.5, birth=.7):
        flags = detector_flags(root, detector)
        values = ['--detectors', detector.split('_')[0], '--encoder', encoder, '--graph', graph,
                  *flags, '--detection-min-score', score, '--new-track-min-score', birth,
                  '--epochs', args.association_epochs, '--output-dir', models / name]
        if tuned_osnet:
            values += ['--finetuned-osnet-weights', models / 'osnet_tuned/best.pt']
        add(name, 'Baselines/OurImplementation/train.py', *values)

    def infer_eval(name, detector='tracktor', graph='on', score=.5, birth=.7):
        flags = detector_flags(root, detector)
        add(name + '_infer', 'Baselines/OurImplementation/infer.py', '--checkpoint', models / name / 'model.pt',
            '--detectors', detector.split('_')[0], '--graph', graph, *flags,
            '--detection-min-score', score, '--new-track-min-score', birth,
            '--output-dir', results / name)
        add(name + '_eval', 'Baselines/OurImplementation/evaluate.py', results / name)

    def tracker_only(name, graph):
        add(name + '_gt_infer', 'Baselines/OurImplementation/infer.py', '--checkpoint', models / name / 'model.pt',
            '--detectors', 'gt', '--graph', graph, '--detection-min-score', .5,
            '--new-track-min-score', .7, '--output-dir', results / (name + '_gt'))
        add(name + '_gt_eval', 'Baselines/OurImplementation/evaluate.py', results / (name + '_gt'))

    add('tracktor_import', 'import_tracktor.py', '--archive', config.TRACKTOR_ARCHIVE, '--train-val-only',
        '--cache-dir', caches / 'tracktor')
    add('authors_train', 'Baselines/DragonTrackGithub/train.py', '--epochs', args.authors_epochs,
        '--batch-size', 2, '--output-dir', models / 'authors')
    for encoder in ('densenet', 'detr', 'osnet'):
        name = 'encoder_' + encoder + '_on'
        train_assoc(name, encoder, 'on')
        infer_eval(name)
        tracker_only(name, 'on')
    train_assoc('encoder_osnet_off', 'osnet', 'off')
    infer_eval('encoder_osnet_off', graph='off')
    tracker_only('encoder_osnet_off', 'off')
    add('osnet_tune', 'osnet-train.py', '--output-dir', models / 'osnet_tuned',
        '--epochs', args.osnet_epochs, '--lr', args.osnet_lr)
    for graph in ('on', 'off'):
        name = 'osnet_tuned_' + graph
        train_assoc(name, 'osnet', graph, tuned_osnet=True)
        infer_eval(name, graph=graph)
        tracker_only(name, graph)

    add('yolox_pre_cache', 'detect-yolox.py', '--split', 'trainval', '--width', 960, '--height', 544,
        '--min-score', .05, '--weights-dir', root / 'source_weights/yolox', '--cache-dir', caches / 'yolox_pre')
    add('rtdetr_pre_cache', 'detect.py', '--backend', 'rtdetrv2', '--split', 'trainval', '--image-size', 960,
        '--min-score', .05, '--cache-dir', caches / 'rtdetr_pre')
    add('iter_pre_cache', 'detect-itere2edet.py', '--checkpoint', ROOT / 'weights/iter_e2edet/model.pth',
        '--split', 'all', '--short-side', 960, '--max-size', 1600, '--min-score', .05,
        '--cache-dir', caches / 'iter_pre')
    add('yolox_tune', 'detector-train-yolox.py', '--output-dir', models / 'yolox_tuned',
        '--weights-dir', root / 'source_weights/yolox', '--epochs', args.detector_epochs,
        '--width', 960, '--height', 544)
    add('rtdetr_tune', 'detector-train.py', '--output-dir', models / 'rtdetr_tuned',
        '--epochs', args.detector_epochs, '--image-size', 960,
        '--selection-mode', 'sweep', '--scores', .2, .3, .4, .5)
    add('iter_tune', 'detector-train-itere2edet.py', '--output-dir', models / 'iter_tuned',
        '--epochs', args.detector_epochs, '--train-short-side', 960, '--train-max-size', 1600)
    add('yolox_tuned_cache', 'detect-yolox.py', '--split', 'trainval', '--width', 960, '--height', 544,
        '--min-score', .05, '--checkpoint', models / 'yolox_tuned/best.pth', '--cache-dir', caches / 'yolox_tuned')
    add('rtdetr_tuned_cache', 'detect.py', '--backend', 'rtdetrv2', '--split', 'trainval', '--image-size', 960,
        '--min-score', .05, '--weights-dir', models / 'rtdetr_tuned/best', '--cache-dir', caches / 'rtdetr_tuned')
    add('iter_tuned_cache', 'detect-itere2edet.py', '--checkpoint', models / 'iter_tuned/best.pth',
        '--split', 'all', '--short-side', 960, '--max-size', 1600, '--min-score', .05,
        '--cache-dir', caches / 'iter_tuned')

    for source in sources:
        kind = source.split('_')[0]
        if source == 'tracktor':
            add('detector_report_tracktor', 'evaluate_detector.py', '--backend', 'tracktor',
                '--tracktor-cache-dir', caches / 'tracktor', '--output-dir', results / 'detector_tracktor')
        else:
            backend = {'yolox': 'yolox', 'rtdetr': 'rtdetrv2', 'iter': 'iter_e2edet'}[kind]
            add('detector_report_' + source, 'evaluate_detector.py', '--backend', backend,
                '--cache-dir', caches / source, '--scores', .3, .4, .5,
                '--output-dir', results / ('detector_' + source))
        add('score_' + source, 'final-score-search.py', '--checkpoint', models / 'osnet_tuned_off/model.pt',
            '--detector', kind, *detector_flags(root, source), '--output-dir', results / ('score_' + source))

    for source in ('yolox_tuned', 'rtdetr_tuned', 'iter_tuned'):
        kind = source.split('_')[0]
        name = 'adapted_' + source
        train_assoc(name, 'osnet', 'off', source, tuned_osnet=True,
                    score='@score:' + source, birth='@birth:' + source)
        add('score_' + name, 'final-score-search.py', '--checkpoint', models / name / 'model.pt',
            '--detector', kind, *detector_flags(root, source), '--output-dir', results / ('score_' + name))

    add('authors_infer', 'Baselines/DragonTrackGithub/infer.py', '--checkpoint',
        models / f'authors/models_v3/epoch_{args.authors_epochs}.pth', '--detectors', 'gt', 'tracktor', 'yolox',
        '--tracktor-cache-dir', caches / 'tracktor', '--yolox-cache-dir', caches / 'yolox_pre',
        '--output-dir', results / 'authors')
    add('authors_eval', 'Baselines/DragonTrackGithub/evaluate.py', results / 'authors')
    add('report', 'final-report.py', '--root', root)
    return jobs


def resolved(values, root):
    output = []
    for value in values:
        if value.startswith('@'):
            field, source = value[1:].split(':', 1)
            result = json.loads((root / 'results' / ('score_' + source) / 'best.json').read_text())
            value = str(result['detection_min_score' if field == 'score' else 'new_track_min_score'])
        output.append(value)
    return output


def save_state(path, state):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(state, indent=2) + '\n')
    pending.replace(path)


def stage_artifacts(name, root, authors_epochs):
    """Paths whose presence proves a stage published its final output."""
    models, results, caches = root / 'models', root / 'results', root / 'caches'
    videos = [*config.TRAIN_SEQUENCES, *config.VAL_SEQUENCES]
    if name == 'tracktor_import':
        return [caches / 'tracktor' / config.DATASET_SPLIT / video / 'summary.json' for video in videos]
    if name.endswith('_cache'):
        return [caches / name.removesuffix('_cache') / config.DATASET_SPLIT / video / 'summary.json'
                for video in videos]
    if name == 'authors_train':
        return [models / 'authors/models_v3' / f'epoch_{authors_epochs}.pth']
    if name == 'authors_infer':
        return [results / 'authors/manifest.json']
    if name == 'authors_eval':
        return [results / 'authors/evaluation/results.json']
    if name == 'report':
        return [root / 'comparison.md']
    if name.startswith('detector_report_'):
        return [results / ('detector_' + name.removeprefix('detector_report_')) / 'summary.json']
    if name.startswith('score_'):
        return [results / name / 'best.json']
    if name.endswith('_gt_infer'):
        return [results / (name.removesuffix('_gt_infer') + '_gt') / 'manifest.json']
    if name.endswith('_gt_eval'):
        return [results / (name.removesuffix('_gt_eval') + '_gt') / 'evaluation/results.json']
    if name.endswith('_infer'):
        return [results / name.removesuffix('_infer') / 'manifest.json']
    if name.endswith('_eval'):
        return [results / name.removesuffix('_eval') / 'evaluation/results.json']
    if name == 'osnet_tune':
        return [models / 'osnet_tuned/last.pt', models / 'osnet_tuned/best.pt']
    if name == 'yolox_tune':
        return [models / 'yolox_tuned/last.pt', models / 'yolox_tuned/best.pth']
    if name == 'rtdetr_tune':
        return [models / 'rtdetr_tuned/last.pt', models / 'rtdetr_tuned/best/training_manifest.json']
    if name == 'iter_tune':
        return [models / 'iter_tuned/latest.pth', models / 'iter_tuned/best.pth']
    if name.startswith(('encoder_', 'osnet_tuned_', 'adapted_')):
        return [models / name / 'last.pt', models / name / 'model.pt']
    raise ValueError(f'No output contract for stage {name}')


def recovery_command(name, values, root):
    """Restart an interrupted epoch, retaining each trainer's completed checkpoints."""
    output = Path(values[values.index('--output-dir') + 1]) if '--output-dir' in values else None
    if name.startswith('detector_report_'):
        if output and output.exists():
            shutil.rmtree(output)
    elif name.endswith('_eval') or name == 'authors_eval':
        run = Path(values[-1])
        partial = run / 'evaluation'
        if partial.exists():
            shutil.rmtree(partial)
    elif name.endswith('_infer') or name == 'authors_infer':
        if output and output.exists():
            shutil.rmtree(output)
    elif name == 'authors_train' and output and output.exists():
        if list((output / 'models_v3').glob('epoch_*.pth')):
            values.append('--resume')
        else:
            shutil.rmtree(output)
    elif name == 'iter_tune' and output and output.exists():
        if (output / 'latest.pth').is_file():
            values.append('--resume')
        else:
            shutil.rmtree(output)
    elif name in ('yolox_tune', 'rtdetr_tune', 'osnet_tune') or name.startswith(('encoder_', 'osnet_tuned_', 'adapted_')):
        if output and output.exists():
            checkpoint = output / 'last.pt'
            if name.startswith(('encoder_', 'osnet_tuned_', 'adapted_')):
                if checkpoint.is_file() or (output / 'model.pt').is_file():
                    values.append('--resume')
                else:
                    shutil.rmtree(output)
            elif not checkpoint.is_file() and not (output / 'best.pt').is_file():
                shutil.rmtree(output)
    return values


def stop_stage(child):
    """Stop the whole stage process group, including nested search/evaluation jobs."""
    if child.poll() is not None:
        return
    for sig, seconds in ((signal.SIGINT, 20), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:
            continue
    child.wait()


def epoch_progress(name, root):
    folder = root / 'models' / name
    if name == 'authors_train':
        epochs = [int(path.stem.split('_')[-1]) for path in (root / 'models/authors/models_v3').glob('epoch_*.pth')]
        return max(epochs) if epochs else 0
    if name == 'iter_tune':
        folder = root / 'models/iter_tuned'
    if name in ('osnet_tune', 'yolox_tune', 'rtdetr_tune'):
        folder = root / 'models' / name.replace('_tune', '_tuned')
    history = folder / 'history.csv'
    if history.is_file():
        lines = history.read_text().strip().splitlines()
        if len(lines) > 1:
            try:
                return int(lines[-1].split(',')[0])
            except ValueError:
                pass
    history = folder / 'history.json'
    if history.is_file():
        rows = json.loads(history.read_text())
        return int(rows[-1]['epoch']) if rows else 0
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'final-ouput')
    parser.add_argument('--run', action='store_true', help='Explicitly start all training and evaluations')
    parser.add_argument('--pause', action='store_true', help='Request a safe pause of the active run')
    parser.add_argument('--resume', action='store_true', help='Continue after an interrupted stage')
    parser.add_argument('--authors-epochs', type=int, default=5)
    parser.add_argument('--association-epochs', type=int, default=5)
    parser.add_argument('--detector-epochs', type=int, default=5)
    parser.add_argument('--osnet-epochs', type=int, default=5)
    parser.add_argument('--osnet-lr', type=float, default=1e-5)
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    if args.pause:
        if args.run or args.resume:
            parser.error('--pause is a separate command')
        manifest = root / 'run.json'
        if not manifest.is_file():
            parser.error(f'No active run in {root}')
        state = json.loads(manifest.read_text())
        if state.get('status') != 'running':
            parser.error(f"Run is {state.get('status')}, not running")
        (root / 'pause.request').write_text('pause\n')
        print(f'Pause requested for {state.get("active")}; current epoch will restart on resume.')
        return
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if min(args.authors_epochs, args.association_epochs, args.detector_epochs, args.osnet_epochs) < 1:
        parser.error('Epochs must be positive')
    validate_video_splits()
    jobs = build_stages(root, args)
    if not args.run:
        for number, (name, script, values) in enumerate(jobs, 1):
            print(f'{number:02d} {name}: {sys.executable} {ROOT / script} {" ".join(values)}')
        print(f'\n{len(jobs)} stages previewed; nothing started. Use --run to execute.')
        return
    manifest = root / 'run.json'
    recipe = {'authors_epochs': args.authors_epochs, 'association_epochs': args.association_epochs,
              'detector_epochs': args.detector_epochs, 'osnet_epochs': args.osnet_epochs,
              'osnet_lr': args.osnet_lr, 'train': config.TRAIN_SEQUENCES, 'val': config.VAL_SEQUENCES,
              'dataset': str(config.DATASET_ROOT.resolve())}
    if args.resume:
        if not manifest.is_file():
            parser.error(f'Missing {manifest}')
        state = json.loads(manifest.read_text())
        if state['recipe'] != recipe:
            parser.error('Run recipe changed; use a new output directory')
        if state.get('status') == 'running' and state.get('runner_pid'):
            try:
                os.kill(state['runner_pid'], 0)
            except ProcessLookupError:
                pass  # The previous runner exited unexpectedly; recover its active stage.
            else:
                parser.error('The previous runner is still active; pause it before resuming')
        for completed in state['completed']:
            missing = [str(path) for path in stage_artifacts(completed, root, args.authors_epochs)
                       if not path.is_file()]
            if missing:
                parser.error(f'Completed stage {completed} lost its output: {missing}')
    else:
        if root.exists() and any(p.name not in ('README.md', '.gitignore') for p in root.iterdir()):
            parser.error(f'Output contains another run: {root}')
        root.mkdir(parents=True, exist_ok=True)
        state = {'recipe': recipe, 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'completed': [], 'status': 'ready'}
        save_state(manifest, state)
    pause_file = root / 'pause.request'
    pause_file.unlink(missing_ok=True)
    state['status'] = 'running'
    state['runner_pid'] = os.getpid()
    save_state(manifest, state)
    try:
        with tqdm(total=len(jobs), initial=len(state['completed']), desc='Final experiment', unit='stage') as bar:
            for name, script, raw in jobs:
                if name in state['completed']:
                    continue
                if pause_file.exists():
                    state['status'] = 'paused'
                    save_state(manifest, state)
                    print('Paused between stages; resume with python final-train.py --run --resume')
                    return
                values = resolved(raw, root)
                if args.resume and state.get('active') == name:
                    values = recovery_command(name, values, root)
                command = [sys.executable, '-u', str(ROOT / script), *values]
                state['active'] = name
                state['command'] = command
                save_state(manifest, state)
                tqdm.write(f'\n{name}: {" ".join(command)}')
                child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
                last_display = 0
                try:
                    while child.poll() is None:
                        if pause_file.exists():
                            stop_stage(child)
                            state['status'] = 'paused'
                            save_state(manifest, state)
                            print(f'Paused at {name}; resume with python final-train.py --run --resume')
                            return
                        if time.monotonic() - last_display > 10:
                            epoch = epoch_progress(name, root)
                            if name == 'authors_train':
                                budget = args.authors_epochs
                            elif name == 'osnet_tune':
                                budget = args.osnet_epochs
                            elif name in ('yolox_tune', 'rtdetr_tune', 'iter_tune'):
                                budget = args.detector_epochs
                            elif name.startswith(('encoder_', 'osnet_tuned_', 'adapted_')) and not name.endswith(('_infer', '_eval')):
                                budget = args.association_epochs
                            else:
                                budget = None
                            bar.set_postfix(stage=name, epoch=f'{epoch}/{budget}' if budget else '-', refresh=True)
                            last_display = time.monotonic()
                        time.sleep(.5)
                except KeyboardInterrupt:
                    stop_stage(child)
                    state['status'] = 'paused'
                    save_state(manifest, state)
                    print(f'Paused at {name}; resume with python final-train.py --run --resume')
                    return
                if child.returncode:
                    state['status'] = 'failed'
                    save_state(manifest, state)
                    raise subprocess.CalledProcessError(child.returncode, command)
                artifacts = stage_artifacts(name, root, args.authors_epochs)
                missing = [str(path) for path in artifacts if not path.is_file()]
                if missing:
                    state['status'] = 'failed'
                    save_state(manifest, state)
                    raise RuntimeError(f'Stage {name} exited without required output: {missing}')
                state.setdefault('artifacts', {})[name] = [str(path) for path in artifacts]
                state['completed'].append(name)
                state.pop('active', None)
                state.pop('command', None)
                save_state(manifest, state)
                bar.update(1)
        state['status'] = 'complete'
        save_state(manifest, state)
    finally:
        pause_file.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
