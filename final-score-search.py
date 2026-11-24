"""Find validation tracking thresholds for one fixed GCN-off checkpoint and detector."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--detector', choices=('tracktor', 'yolox', 'rtdetr', 'iter'), required=True)
    parser.add_argument('--tracktor-cache-dir', type=Path)
    parser.add_argument('--yolox-cache-dir', type=Path)
    parser.add_argument('--yolox-checkpoint', type=Path)
    parser.add_argument('--rtdetr-cache-dir', type=Path)
    parser.add_argument('--rtdetr-weights-dir', type=Path)
    parser.add_argument('--rtdetr-image-size', type=int, default=960)
    parser.add_argument('--rtdetr-cache-min-score', type=float, default=.05)
    parser.add_argument('--iter-cache-dir', type=Path)
    parser.add_argument('--iter-checkpoint', type=Path)
    parser.add_argument('--scores', nargs='+', type=float, default=[.3, .4, .5])
    parser.add_argument('--birth-scores', nargs='+', type=float, default=[.5, .7])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not all(0 <= x <= 1 for x in args.scores + args.birth_scores):
        parser.error('Scores must be in [0,1]')
    flags = {'tracktor': ('tracktor_cache_dir',), 'yolox': ('yolox_cache_dir', 'yolox_checkpoint'),
             'rtdetr': ('rtdetr_cache_dir', 'rtdetr_weights_dir'),
             'iter': ('iter_cache_dir', 'iter_checkpoint')}[args.detector]
    if getattr(args, flags[0]) is None or (args.detector == 'iter' and args.iter_checkpoint is None):
        parser.error('Detector cache and required checkpoint must be supplied')
    source = []
    for key in flags:
        value = getattr(args, key)
        if value is not None:
            source.extend(['--' + key.replace('_', '-'), str(value)])
    if args.detector == 'rtdetr':
        source.extend(['--rtdetr-image-size', str(args.rtdetr_image_size),
                       '--rtdetr-cache-min-score', str(args.rtdetr_cache_min_score)])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    scores = [.5] if args.detector == 'tracktor' else sorted(set(args.scores))
    for score in scores:
        for birth in sorted(set(args.birth_scores)):
            folder = output / f'det_{score:g}_new_{birth:g}'
            report = folder / 'evaluation/results.json'
            if not report.is_file():
                if folder.exists():
                    # Inference and evaluation write into a fresh trial directory.
                    # An interrupted trial has no final report, so redo only that trial.
                    shutil.rmtree(folder)
                subprocess.run([sys.executable, str(ROOT / 'Baselines/OurImplementation/infer.py'),
                                '--checkpoint', str(args.checkpoint), '--detectors', args.detector,
                                '--graph', 'off', *source, '--detection-min-score', str(score),
                                '--new-track-min-score', str(birth), '--output-dir', str(folder)],
                               cwd=ROOT, check=True)
                subprocess.run([sys.executable, str(ROOT / 'Baselines/OurImplementation/evaluate.py'), str(folder)],
                               cwd=ROOT, check=True)
            metrics = json.loads(report.read_text())['scores'][args.detector]['COMBINED']
            rows.append({'detection_min_score': score, 'new_track_min_score': birth, 'metrics': metrics,
                         'run': str(folder)})
    winner = max(rows, key=lambda row: (row['metrics']['HOTA'], row['metrics']['IDF1'],
                                        -row['detection_min_score'], -row['new_track_min_score']))
    (output / 'trials.json').write_text(json.dumps(rows, indent=2) + '\n')
    (output / 'best.json').write_text(json.dumps(winner, indent=2) + '\n')
    print(f"Best {args.detector}: HOTA={winner['metrics']['HOTA']}, "
          f"detection={winner['detection_min_score']}, birth={winner['new_track_min_score']}")


if __name__ == '__main__':
    main()
