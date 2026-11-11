"""Compare DDQ and RT-DETR evaluation summaries using identical videos and frame coverage.

python compare-detectors.py --reports path/to/rtdetr/summary.json path/to/ddq/summary.json
"""

import argparse
from datetime import datetime
import json
from pathlib import Path


def compare(paths, *, iou=.5, min_precision=.70, min_recall=.55):
    """Compare measured numeric thresholds; never interpolate scores or mix validation sets."""
    results, expected = [], None
    for path in paths:
        report = json.loads(path.read_text())
        identity = {
            'split': report['split'], 'policy': report['policy'],
            'coverage': {name: (source['metadata'].get('identity', {}).get('sequence_path'),
                               source['evaluated_frames']) for name, source in report['sources'].items()},
            'counts': {row['sequence']: (row['frames'], row['gt']) for row in report['results']
                       if row['iou'] == iou and row['score'] == 'all'},
        }
        if expected is not None and identity != expected:
            raise ValueError(f'Incompatible evaluation videos/coverage/GT counts/policy: {path}')
        expected = identity
        rows = [r for r in report['results'] if r['sequence']=='COMBINED' and r['iou']==iou
                and isinstance(r['score'], (int,float)) and r['f1'] is not None]
        if not rows:
            raise ValueError(f'No numeric confidence sweep at IoU {iou}: {path}')
        eligible = [r for r in rows if r['precision'] is not None and r['precision'] >= min_precision
                    and r['recall'] is not None and r['recall'] >= min_recall]
        best = max(rows, key=lambda r: (r['f1'], r['recall']))
        chosen = max(eligible, key=lambda r: (r['f1'], r['recall'])) if eligible else best
        results.append(dict(report=str(path.resolve()), backend=report['backend'], eligible=bool(eligible),
                            selected=chosen, best_f1=best))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True, help='evaluate_detector.py summary.json files')
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--min-precision', type=float, default=.70)
    parser.add_argument('--min-recall', type=float, default=.55)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if len(args.reports) < 2:
        parser.error('Provide at least two evaluation summaries')
    if not (0 < args.iou <= 1 and 0 <= args.min_precision <= 1 and 0 <= args.min_recall <= 1):
        parser.error('IoU must be in (0,1]; precision/recall floors in [0,1]')
    rows = compare(args.reports, iou=args.iou, min_precision=args.min_precision, min_recall=args.min_recall)
    out = args.output_dir or Path(__file__).resolve().parent/'outputs'/'detector_comparison'/f'{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(dict(iou=args.iou, min_precision=args.min_precision,
                                                      min_recall=args.min_recall, results=rows), indent=2)+'\n')
    lines = ['# Detector comparison', '', f'IoU {args.iou}; required precision ≥{args.min_precision:.0%}, '
             f'recall ≥{args.min_recall:.0%}. Highest measured F1 meeting both; otherwise a clearly marked fallback.', '',
             '| Source | Score | Precision | Recall | F1 | TP | FP | Meets goals |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        r = row['selected']
        precision = f"{r['precision']:.2%}" if r['precision'] is not None else 'n/a'
        lines.append(f"| {row['backend']} | {r['score']} | {precision} | {r['recall']:.2%} | {r['f1']:.2%} | "
                     f"{r['tp']} | {r['fp']} | {'yes' if row['eligible'] else 'NO — fallback'} |")
    lines += ['', 'These are detector diagnostics, not tracking accuracy or COCO AP. Each model may need a '
              'different confidence threshold. Compare at similar precision as well as maximum F1. '
              'Input resolution, pretrained versus fine-tuned weights and output caps remain part of each source recipe.', '',
              '## Source reports', ''] + [f"- [{r['backend']}]({r['report']})" for r in rows]
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:8+len(rows)]))
    print(f'Comparison saved: {out / "comparison.md"}')
    return out


if __name__ == '__main__':
    main()
