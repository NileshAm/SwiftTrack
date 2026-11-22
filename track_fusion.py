"""Track three saved fusion variants sequentially with the same Tracktor checkpoint."""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from Utils import config
from Utils.detections import file_digest


def inputs(suite, fusion):
    plan=json.loads((suite/'plan.json').read_text())
    report=json.loads((fusion/'summary.json').read_text())
    marker=json.loads((fusion/'COMPLETE.json').read_text())
    if not marker.get('complete') or marker['summary_sha256']!=file_digest(fusion/'summary.json'):
        raise ValueError('Fusion completion marker or summary hash is invalid')
    if report['max_frames'] is not None:
        raise ValueError('Use a full-video fusion sweep, not --max-frames outputs')
    if set(report['provenance']['tracktor']['sources'])!=set(plan['settings']['VAL_SEQUENCES']):
        raise ValueError('Fusion validation sequences differ from suite')
    checkpoint=suite/'sources/tracktor/association/model.pt'
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    expected=json.loads((suite/'sources/tracktor/result.json').read_text())['checkpoint_sha256']
    if file_digest(checkpoint)!=expected:
        raise ValueError('Tracktor checkpoint changed since baseline evaluation')
    variants={}
    for score in (.4,.5,.6):
        label=f'score_{score:g}_overlap_0.5'
        paths={seq:(fusion/label/seq/'det.txt').resolve() for seq in plan['settings']['VAL_SEQUENCES']}
        for path in paths.values():
            if not path.is_file(): raise FileNotFoundError(path)
        variants[label]=paths
    identity=dict(checkpoint_sha256=expected,plan_sha256=file_digest(suite/'plan.json'),
                  fusion_summary_sha256=file_digest(fusion/'summary.json'),
                  exports={label:{seq:file_digest(p) for seq,p in paths.items()} for label,paths in variants.items()})
    return plan,checkpoint,variants,identity


def child(suite,fusion,attempt,label):
    from detector_suite import configure_source
    plan,checkpoint,variants,identity=inputs(suite,fusion)
    configure_source(suite,'tracktor',plan)
    config.CHECKPOINT_PATH=checkpoint
    config.TRACK_OUTPUT_DIR=attempt/'tracking/val'
    config.EVALUATION_OUTPUT_DIR=attempt/'evaluation/val'
    config.TRACKING_VARIANT='fusion_inference_transfer'
    import tracking
    import evaluate
    summaries=tracking.main(detection_overrides=variants[label])
    evaluate.main()
    info=json.loads((config.EVALUATION_OUTPUT_DIR/'run_info.json').read_text())
    result=dict(variant=label,metrics=info['metrics']['MOT17'],
                fps=sum(r['frames'] for r in summaries)/sum(r['seconds'] for r in summaries),
                identity=identity,protocol='Tracktor-trained checkpoint transferred to fused validation detections')
    (attempt/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir',type=Path,default=Path('outputs/detector_suite_v1'))
    parser.add_argument('--fusion-dir',type=Path,help='Default: latest completed full-video fusion sweep')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--child',choices=['score_0.4_overlap_0.5','score_0.5_overlap_0.5','score_0.6_overlap_0.5'],help=argparse.SUPPRESS)
    parser.add_argument('--attempt',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    suite=args.suite_dir.resolve()
    fusion=args.fusion_dir
    if fusion is None:
        choices=[]
        for path in (suite/'fusion').iterdir():
            if (path/'COMPLETE.json').is_file() and (path/'summary.json').is_file():
                if json.loads((path/'summary.json').read_text()).get('max_frames') is None:
                    choices.append(path)
        if not choices: raise FileNotFoundError('Run python detector_fusion.py on full videos first')
        fusion=max(choices,key=lambda p:p.stat().st_mtime)
    fusion=fusion.resolve()
    plan,checkpoint,variants,identity=inputs(suite,fusion)
    if args.child:
        child(suite,fusion,args.attempt.resolve(),args.child); return
    print(f'Fusion: {fusion}\nCheckpoint: {checkpoint}\nVariants: {list(variants)}',flush=True)
    if args.check_only:
        print('Inputs validated. No GPU processing launched.'); return
    from filelock import FileLock
    from detector_suite import run_command
    root=fusion/'tracking_comparison'
    root.mkdir(exist_ok=True)
    results=[]
    with FileLock(str(root/'run.lock'),timeout=0):
        for label in variants:
            parent=root/label
            parent.mkdir(exist_ok=True)
            existing=[]
            for path in parent.glob('*/result.json'):
                saved=json.loads(path.read_text())
                if saved.get('identity')==identity: existing.append((path,saved))
            if existing:
                results.append(max(existing,key=lambda item:item[0].stat().st_mtime)[1]); continue
            attempt=parent/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            attempt.mkdir()
            run_command([sys.executable,Path(__file__).resolve(),'--suite-dir',suite,'--fusion-dir',fusion,
                         '--child',label,'--attempt',attempt],attempt/'run.log')
            results.append(json.loads((attempt/'result.json').read_text()))
        baseline=json.loads((suite/'sources/tracktor/result.json').read_text())
        rows=[dict(variant='tracktor_baseline',**baseline['metrics'])]+[
              dict(variant=r['variant'],**r['metrics']) for r in results]
        with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (root/'comparison.json').write_text(json.dumps(dict(results=results,baseline=baseline),indent=2)+'\n')
        lines=['# Fusion tracking comparison','',
            'Same frozen appearance, graph checkpoint and tracking settings. Fusion is applied at inference. '
            'Validation-tuned results, not an independent final benchmark. FPS excludes detector inference.', '',
            '| Configuration | HOTA | IDF1 | MOTA | IDSW | FN | FP |','|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append('| '+r['variant']+' | '+' | '.join(str(r[k]) for k in ['HOTA','IDF1','MOTA','IDSW','CLR_FN','CLR_FP'])+' |')
        (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(f'Comparison: {root / "comparison.md"}')


if __name__=='__main__': main()
