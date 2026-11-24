"""Summarize the final run's separate detector, tracker, and full-pipeline results."""
import argparse
import json
from pathlib import Path

SOURCES = ('tracktor', 'yolox_pre', 'rtdetr_pre', 'iter_pre',
           'yolox_tuned', 'rtdetr_tuned', 'iter_tuned')
TRACKING = ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'CLR_FP', 'CLR_FN', 'IDSW')


def tracking_row(root, name, detector, score_search=False, authors=False):
    path = (root / 'results' / ('score_' + name) / 'best.json' if score_search else
            root / 'results' / name / 'evaluation/results.json')
    data = json.loads(path.read_text())
    metrics = (data['metrics'] if score_search else data['scores'][detector]['COMBINED'])
    return [name, *[str(metrics[key]) for key in TRACKING]]


def table(lines, title, headers, rows):
    lines.extend([f'## {title}', '', '| ' + ' | '.join(headers) + ' |',
                  '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |'])
    for row in rows:
        lines.append('| ' + ' | '.join(map(str, row)) + ' |')
    lines.append('')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    lines = ['# Final MOT17 experiment progression', '',
             'Local development validation on MOT17-02-SDP and MOT17-09-SDP, not official test scores.',
             'GT identities are removed from tracker inputs. Detector thresholds were selected on these same validation videos.', '']
    names = ('encoder_densenet_on', 'encoder_detr_on', 'encoder_osnet_on')
    table(lines, '1. Appearance encoder comparison on Tracktor boxes', ['Model', *TRACKING],
          [tracking_row(root, name, 'tracktor') for name in names])
    names = ('encoder_osnet_on', 'encoder_osnet_off', 'osnet_tuned_on', 'osnet_tuned_off')
    table(lines, '2. GCN and OSNet fine-tuning on Tracktor boxes', ['Model', *TRACKING],
          [tracking_row(root, name, 'tracktor') for name in names])
    names = ('encoder_densenet_on', 'encoder_detr_on', 'encoder_osnet_on',
             'encoder_osnet_off', 'osnet_tuned_on', 'osnet_tuned_off')
    table(lines, '3. Tracker alone with GT boxes and hidden IDs', ['Model', *TRACKING],
          [tracking_row(root, name + '_gt', 'gt') for name in names])
    detector_rows = []
    for source in SOURCES:
        report = json.loads((root / 'results' / ('detector_' + source) / 'summary.json').read_text())
        selected = json.loads((root / 'results' / ('score_' + source) / 'best.json').read_text())
        threshold = selected['detection_min_score']
        for iou in (.5, .75):
            matches = [row for row in report['results'] if row['sequence'] == 'COMBINED'
                       and row['iou'] == iou and row['score'] == threshold]
            if not matches:
                raise ValueError(f'Missing detector score {source}: IoU {iou}, threshold {threshold}')
            row = matches[0]
            detector_rows.append([source, iou, threshold, *[round(row[key] * 100, 2) if row[key] is not None else 'n/a'
                            for key in ('precision', 'recall', 'f1', 'height_lt_50_recall', 'visibility_lt_0.3_recall')]])
    table(lines, '4. Detectors alone', ['Source', 'IoU', 'Score', 'Precision %', 'Recall %', 'F1 %',
                                       'Small recall %', 'Heavy occlusion recall %'], detector_rows)
    table(lines, '5. Same OSNet tracker without GCN across detectors', ['Boxes', *TRACKING],
          [tracking_row(root, source, source.split('_')[0], score_search=True) for source in SOURCES])
    table(lines, '6. Detector-specific association training', ['Pipeline', *TRACKING],
          [tracking_row(root, 'adapted_' + source, source.split('_')[0], score_search=True)
           for source in ('yolox_tuned', 'rtdetr_tuned', 'iter_tuned')])
    table(lines, '7. Authors CUDA baseline', ['Boxes', *TRACKING],
          [tracking_row(root, 'authors', source) for source in ('gt', 'tracktor', 'yolox')])
    target = root / 'comparison.md'
    target.write_text('\n'.join(lines) + '\n')
    print(f'Final report: {target}')


if __name__ == '__main__':
    main()
