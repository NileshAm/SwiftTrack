"""Sequential tracking/evaluation with and without graph refinement; no training.

python gcn_ablation.py                  # OSNet, with graph then direct cosine
python gcn_ablation.py --encoder all    # Repeat for DenseNet, DETR, OSNet

Both arms retain the checkpoint's learned Sinkhorn unmatched preference, score
scale, candidate gate and Hungarian assignment. This is an inference bypass
ablation, not a separately retrained/calibrated no-graph model.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

from encoder_experiment import configure
from Utils import config
from Utils.detections import file_digest


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def save_report(root, rows):
    fields = ['encoder', 'variant', 'status', 'HOTA', 'IDF1', 'AssA', 'MOTA',
              'IDSW', 'CLR_Re', 'CLR_Pr', 'fps', 'checkpoint_epoch', 'error']
    with (root / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Graph refinement ablation', '',
             'Same encoder checkpoint, detections, motion/history, gate and assignment per encoder.',
             'Without graph: raw appearance cosine, with the SAME learned Sinkhorn matcher.',
             'Inference bypass only; matcher calibration was learned with graph embeddings.',
             'Scores are validation percentages; IDSW is a count. FPS excludes detector inference.', '',
             '| Encoder | Variant | Status | HOTA | IDF1 | AssA | MOTA | IDSW | FPS |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(k, '')) for k in
                     ['encoder', 'variant', 'status', 'HOTA', 'IDF1', 'AssA', 'MOTA', 'IDSW', 'fps']) + ' |')
        if row.get('error'):
            lines.extend(['', f"Error ({row['encoder']}/{row['variant']}): {row['error']}", ''])
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=['reid', 'detr', 'densenet', 'all'], default='reid')
    args = parser.parse_args()
    encoders = ['densenet', 'detr', 'reid'] if args.encoder == 'all' else [args.encoder]
    # Fail before starting a long run if a requested checkpoint/evaluator is absent.
    import trackeval  # noqa: F401
    import torch
    import tracking
    import evaluate
    for encoder in encoders:
        configure(encoder)
        if not config.CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(config.CHECKPOINT_PATH)
    root = config.PROJECT_ROOT / 'outputs' / 'gcn_ablations' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    print(f'Results: {root}', flush=True)
    rows = []
    for encoder in encoders:
        configure(encoder)
        checkpoint_path = config.CHECKPOINT_PATH
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        identity = {'path': str(checkpoint_path), 'sha256': file_digest(checkpoint_path),
                    'epoch': checkpoint['epoch']}
        del checkpoint
        for use_graph in [True, False]:
            configure(encoder)
            variant = 'with_gcn' if use_graph else 'without_gcn'
            folder = root / encoder / variant
            folder.mkdir(parents=True)
            config.TRACKING_VARIANT = variant
            config.TRACK_OUTPUT_DIR = folder / 'tracking' / 'val'
            config.EVALUATION_OUTPUT_DIR = folder / 'evaluation' / 'val'
            config.TEST_OUTPUT_DIR = folder / 'tracking' / 'test'
            row = {'encoder': encoder, 'variant': variant, 'status': 'failed',
                   'checkpoint_epoch': identity['epoch']}
            print(f'\n=== {encoder}: {variant} ===', flush=True)
            with (folder / 'execution.log').open('w') as log:
                with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                    try:
                        if file_digest(checkpoint_path) != identity['sha256']:
                            raise RuntimeError('Checkpoint changed during ablation; stop concurrent training.')
                        manifest = {'checkpoint': identity, 'use_graph': use_graph,
                                    'protocol': 'frozen_checkpoint_graph_bypass_same_sinkhorn',
                                    'settings': {k: v for k, v in vars(config).items() if k.isupper()}}
                        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
                        summaries = tracking.main(use_graph=use_graph)
                        scores = evaluate.main()['MOT17']
                        row.update({k: scores[k] for k in ['HOTA', 'IDF1', 'AssA', 'MOTA', 'IDSW', 'CLR_Re', 'CLR_Pr']})
                        row['fps'] = round(sum(s['frames'] for s in summaries) / sum(s['seconds'] for s in summaries), 3)
                        row['status'] = 'complete'
                    except Exception as error:
                        traceback.print_exc()
                        row['error'] = str(error)
            rows.append(row)
            save_report(root, rows)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f'\nFinished. Comparison: {root / "comparison.md"}', flush=True)
    if any(row['status'] != 'complete' for row in rows):
        raise SystemExit('Some arms failed; inspect comparison.md and their execution.log files.')


if __name__ == '__main__':
    main()
