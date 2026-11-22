"""Sweep Tracktor-first fusion using existing caches. Run: python detector_fusion.py"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from Utils import config
from Utils.data import load_sequence, load_ground_truth, pairwise_iou
from Utils.detections import file_digest
from evaluate_detector import COUNT_FIELDS, score_frame, metrics


def fuse(base, candidates, scores, threshold, overlap):
    """Keep all baseline boxes; greedily add confident non-overlapping proposals.

    Ground truth is never an input. Tracktor scores are not averaged with detector
    confidence. Additional boxes are suppressed against baseline and prior adds.
    """
    kept = [box.copy() for box in base]
    confidence = [1.0] * len(base)
    for index in np.argsort(-scores, kind='stable'):
        if scores[index] < threshold:
            continue
        box = candidates[index]
        if kept and (pairwise_iou(box[None], np.asarray(kept)) >= overlap).any():
            continue
        kept.append(box.copy())
        confidence.append(float(scores[index]))
    return np.asarray(kept,dtype=np.float32).reshape(-1,4), np.asarray(confidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir',type=Path,default=Path('outputs/detector_suite_v1'))
    parser.add_argument('--source',choices=['rtdetr','yolo26_s','rfdetr_s'],default='rtdetr')
    parser.add_argument('--scores',type=float,nargs='+',default=[.2,.25,.3,.4,.5,.6,.7])
    parser.add_argument('--overlaps',type=float,nargs='+',default=[.3,.5,.7])
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if any(not np.isfinite(x) or not 0<=x<=1 for x in args.scores) or any(
            not np.isfinite(x) or not 0<x<=1 for x in args.overlaps):
        parser.error('Scores must be in [0,1]; overlap thresholds in (0,1]')
    if args.max_frames is not None and args.max_frames<1:
        parser.error('max-frames must be positive')
    root=args.suite_dir.resolve()
    plan=json.loads((root/'plan.json').read_text())
    reports={n:json.loads((root/'sources'/n/'detector_evaluation'/'summary.json').read_text())
             for n in ['tracktor',args.source]}
    jobs=[]
    for seq in plan['settings']['VAL_SEQUENCES']:
        sources={n:r['sources'][seq] for n,r in reports.items()}
        coverage={tuple(s['evaluated_frames']) for s in sources.values()}
        if len(coverage)!=1 or next(iter(coverage))[0]!=1:
            raise ValueError(f'Inconsistent coverage: {seq}')
        length=next(iter(coverage))[1]
        for source in sources.values():
            path=Path(source['path'])
            if not path.is_file() or file_digest(path)!=source['sha256']:
                raise ValueError(f'Missing or changed source: {path}')
        jobs.append((seq,sources,min(length,args.max_frames or length)))
    out=args.output_dir or root/'fusion'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    grid=[(s,o) for s in sorted(set(args.scores)) for o in sorted(set(args.overlaps))]
    totals={setting:dict.fromkeys(COUNT_FIELDS,0) for setting in [None]+grid}
    for seq,sources,length in jobs:
        seqpath=Path(plan['settings']['DATASET_ROOT'])/plan['settings']['DATASET_SPLIT']/seq
        frames={n:load_sequence(seqpath,s['path'],(1,length),box_origin=1,min_score=None)
                for n,s in sources.items()}
        gt=load_ground_truth(seqpath,config.GT_FILE,box_origin=1)
        handles={setting:(out/f'score_{setting[0]:g}_overlap_{setting[1]:g}'/seq) for setting in grid}
        streams={}
        try:
            for setting,directory in handles.items():
                directory.mkdir(parents=True)
                streams[setting]=(directory/'det.txt').open('w',encoding='utf-8')
            for fid in tqdm(range(1,length+1),desc=f'Fuse {seq}',unit='frame'):
                base=frames['tracktor'][fid]['boxes']
                alternative=frames[args.source][fid]
                for setting in [None]+grid:
                    boxes,scores=(base,np.ones(len(base))) if setting is None else fuse(
                        base,alternative['boxes'],alternative['scores'],*setting)
                    # Export first, evaluate afterward: labels never select boxes.
                    if setting is not None:
                        for box,score in zip(boxes,scores):
                            x,y,x2,y2=map(float,box)
                            streams[setting].write(f'{fid},-1,{x+1:.6f},{y+1:.6f},{x2-x:.6f},{y2-y:.6f},{score:.8f},-1,-1,-1\n')
                    result=score_frame(boxes,gt[fid],.5)
                    for key,value in result.items():
                        totals[setting][key]+=value
        finally:
            for stream in streams.values(): stream.close()
    rows=[dict(source='tracktor' if key is None else f'tracktor+{args.source}',
               score=None if key is None else key[0],overlap=None if key is None else key[1],
               **metrics(value)) for key,value in totals.items()]
    with (out/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    ranked=sorted(rows[1:],key=lambda r:(r['f1'] or 0,r['recall'] or 0),reverse=True)
    (out/'summary.json').write_text(json.dumps(dict(source=args.source,max_frames=args.max_frames,
        results=rows,best_detector_f1=ranked[0],provenance=reports),indent=2)+'\n')
    baseline=rows[0]
    lines=['# Tracktor-first fusion validation sweep','',
        'GT-free box construction; validation-tuned detector diagnostics at IoU 0.5. '
        'Uses evaluate_detector.py ignore policy, not official MOT tracking evaluation. '
        'Tracktor boxes are preserved; extra boxes use their original detector confidence. '
        'Tracktor score 1 is a placeholder, not calibrated confidence.', '',
        '| Score | Overlap | Precision | Recall | F1 | Extra TP | Extra FP |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in [baseline]+ranked:
        lines.append(f"| {r['score']} | {r['overlap']} | {r['precision']:.2%} | {r['recall']:.2%} | "
                     f"{r['f1']:.2%} | {r['tp']-baseline['tp']} | {r['fp']-baseline['fp']} |")
    lines+=['','MOT-format det.txt files are exported per setting/video with box origin 1. '
        'These exports cover validation only and are not registered training caches. '
        'Do not point training at them. Detector-F1 selection does not establish better HOTA/IDF1. '
        'Tracking integration and detector-specific association adaptation remain subsequent experiments.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(complete=True,summary_sha256=file_digest(out/'summary.json')))+'\n')
    print(f'Report: {out / "report.md"}')


if __name__=='__main__': main()
