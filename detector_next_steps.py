"""Reuse suite caches to measure complementary coverage; no inference or training.

python detector_next_steps.py
python detector_next_steps.py --max-frames 30
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from Utils import config
from Utils.data import load_sequence, load_ground_truth, match_detections_to_gt
from Utils.detections import file_digest
from evaluate_detector import score_frame


def matched_ids(boxes, gt, iou):
    result = match_detections_to_gt(boxes, gt, iou)
    return set(map(int, result['ids'][result['valid_mask']]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', type=Path, default=Path('outputs/detector_suite_v1'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--iou', type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.iou <= 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('IoU must be in (0,1]; max-frames must be positive')
    root = args.suite_dir.resolve()
    plan = json.loads((root/'plan.json').read_text())
    names = plan['sources']
    reports, thresholds, frames, provenance = {}, {}, {}, {}
    # Validate every input before creating a new report directory.
    for name in names:
        reports[name] = json.loads((root/'sources'/name/'detector_evaluation'/'summary.json').read_text())
        thresholds[name] = json.loads((root/'sources'/name/'operating_point.json').read_text())['selected']['score']
        if reports[name]['split'] != 'val':
            raise ValueError(f'{name}: expected validation report')
        if set(reports[name]['sources']) != set(plan['settings']['VAL_SEQUENCES']):
            raise ValueError(f'{name}: mismatched validation videos')
        provenance[name] = reports[name]['sources']
        for seq, source in provenance[name].items():
            path = Path(source['path'])
            if not path.is_file() or file_digest(path) != source['sha256']:
                raise ValueError(f'Missing or changed cache: {path}')
            if source['evaluated_frames'][0] != 1:
                raise ValueError('Only complete frame ranges starting at 1 are supported')
    out = args.output_dir or root/'next_steps'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    rows, bins, patterns = [], [], []
    for seq in plan['settings']['VAL_SEQUENCES']:
        lengths = {provenance[n][seq]['evaluated_frames'][1] for n in names}
        if len(lengths) != 1:
            raise ValueError('Mismatched cache evaluation coverage')
        length = min(lengths.pop(), args.max_frames or 10**9)
        seqpath = Path(plan['settings']['DATASET_ROOT'])/plan['settings']['DATASET_SPLIT']/seq
        gt_frames = load_ground_truth(seqpath, config.GT_FILE, box_origin=1)
        frames = {n: load_sequence(seqpath, provenance[n][seq]['path'], (1,length),
                                  box_origin=1, min_score=None) for n in names}
        for fid in tqdm(range(1,length+1), desc=seq, unit='frame'):
            gt = gt_frames[fid]
            valid = gt['valid_mask']
            targets = set(map(int, gt['ids'][valid]))
            for mode in ('selected', 'cache_floor'):
                boxes = {n: frames[n][fid]['boxes'][frames[n][fid]['scores'] >=
                         (thresholds[n] if mode == 'selected' else -np.inf)] for n in names}
                hits = {n: matched_ids(boxes[n], gt, args.iou) for n in names}
                baseline = hits['tracktor']
                union = set().union(*hits.values())
                joint = matched_ids(np.concatenate(list(boxes.values()), axis=0), gt, args.iou)
                patterns.append(dict(sequence=seq, frame=fid, mode=mode, gt=len(targets),
                    all_detect=len(set.intersection(*hits.values())),
                    tracktor_only=len(baseline-set().union(*(hits[n] for n in names if n!='tracktor'))),
                    missed_all=len(targets-union), independent_union=len(union), joint_union=len(joint)))
                for n in names:
                    counts = score_frame(boxes[n], gt, args.iou)
                    pair_joint = (baseline if n == 'tracktor' else
                                  matched_ids(np.concatenate([boxes['tracktor'],boxes[n]]),gt,args.iou))
                    rows.append(dict(sequence=seq, frame=fid, mode=mode, source=n, gt=len(targets),
                        tp=counts['tp'], fp=counts['fp'], fn=counts['fn'],
                        ignored=counts['ignored'], duplicates=counts['duplicates'],
                        extra_hits=len(hits[n]-baseline), lost_hits=len(baseline-hits[n]),
                        joint_added=len(pair_joint)-len(baseline)))
                    for index in np.flatnonzero(valid):
                        identity = int(gt['ids'][index])
                        height = gt['boxes'][index,3]-gt['boxes'][index,1]
                        visibility = gt['visibility'][index]
                        bins.append(dict(source=n, mode=mode,
                            height='<50' if height<50 else '50-100' if height<100 else '>=100',
                            visibility='<0.3' if visibility<.3 else '0.3-0.7' if visibility<.7 else '>=0.7',
                            hit=int(identity in hits[n]), extra=int(identity in hits[n]-baseline)))
    def save_csv(path, records):
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    save_csv(out/'frame_counts.csv',rows)
    save_csv(out/'coverage_patterns.csv',patterns)
    summary=[]
    for mode in ('selected','cache_floor'):
        for n in names:
            subset=[r for r in rows if r['mode']==mode and r['source']==n]
            counts={k:sum(r[k] for r in subset) for k in ('gt','tp','fp','fn','extra_hits','lost_hits','joint_added')}
            tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
            summary.append(dict(source=n,mode=mode,**counts,precision=tp/max(tp+fp,1),
                                recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1)))
    save_csv(out/'summary.csv',summary)
    grouped=[]
    for mode in ('selected','cache_floor'):
        for n in names:
            for attribute in ('height','visibility'):
                for bucket in sorted({r[attribute] for r in bins}):
                    values=[r for r in bins if r['mode']==mode and r['source']==n and r[attribute]==bucket]
                    grouped.append(dict(source=n,mode=mode,attribute=attribute,bucket=bucket,
                        gt=len(values),tp=sum(r['hit'] for r in values),extra_hits=sum(r['extra'] for r in values)))
    save_csv(out/'size_visibility.csv',grouped)
    lines=['# Cached detector complementarity','',
        'Validation diagnostic, not an independent final test. Uses the existing detector evaluator ignore policy; '
        'this differs from official MOT distractor preprocessing. No detector or tracker was trained.',
        '',f'IoU: {args.iou}. Selected thresholds: {thresholds}', '',
        '| Source | Mode | Precision | Recall | F1 | Extra hits vs Tracktor | Source FP | Joint added TP |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['source']} | {r['mode']} | {r['precision']:.2%} | {r['recall']:.2%} | "
                     f"{r['f1']:.2%} | {r['extra_hits']} | {r['fp']} | {r['joint_added']} |")
    lines += ['', 'Extra hits compare separate one-to-one assignments; joint added TP rematches concatenated '
        'Tracktor + alternative proposals. Both use GT and are diagnostic upper bounds, not deployable fusion. '
        'Source FP is not the incremental FP of a fused detector. Cache-floor results cannot recover discarded proposals.',
        '', '## Next decision', '',
        'Consider Tracktor + the source with useful joint added TP at an acceptable FP cost. '
        'Test duplicate suppression without GT before tracking any fused output. '
        'Do not automatically switch weights or resolution based on these upper bounds.', '',
        'Existing confidence curves are reproduced in confidence_sweep.csv and precision_recall.png. '
        'Thresholds were tuned on validation. The completed suite already tested detector-specific association '
        'training and ByteTrack-inspired recovery; recovery did not improve the baseline.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sweeps=[]
    for n in names:
        for r in reports[n]['results']:
            if r['sequence']=='COMBINED' and r['iou']==args.iou:
                sweeps.append(dict(source=n,**r))
    save_csv(out/'confidence_sweep.csv',sweeps)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for n in names:
        values=[r for r in sweeps if r['source']==n and isinstance(r['score'],(int,float))]
        ax.plot([r['recall'] for r in values],[r['precision'] for r in values],marker='.',label=n)
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Saved validation confidence sweeps')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'precision_recall.png',dpi=160); plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(suite=str(root),iou=args.iou,
        max_frames=args.max_frames,thresholds=thresholds,sources=provenance),indent=2)+'\n')
    print(f'Report: {out / "report.md"}')
    return out


if __name__=='__main__':
    main()
