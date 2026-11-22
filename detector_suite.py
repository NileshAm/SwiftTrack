"""Unattended A-E detector experiment suite. Run: python detector_suite.py

Downloads assets, trains RT-DETR, exports detections, trains matched OSNet
association models, evaluates A-D, then tests high/low recovery for the HOTA winner.
Restart the identical command to resume. See guide 30 for protocol and limitations.
"""
import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from Utils import config
from Utils.detections import file_digest

SCORES = [.05, .10, .15, .20, .225, .25, .275, .30, .35, .40, .50, .60, .70]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, default=str) + '\n')
    pending.replace(path)


@contextmanager
def argv(values):
    previous = sys.argv
    sys.argv = [previous[0]] + list(map(str, values))
    try:
        yield
    finally:
        sys.argv = previous


def worker_python(root):
    return root / 'worker_env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def run_command(command, log, cwd=None):
    """Argument-list subprocess; stream output to terminal and persistent log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND: ' + repr(list(map(str, command))) + '\n')
        stream.flush()
        environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        process = subprocess.Popen(list(map(str, command)), cwd=cwd or config.PROJECT_ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                # Windows PowerShell commonly uses cp1252; optional detector
                # libraries may emit Unicode progress-bar glyphs. Keep the
                # complete UTF-8 line in the log while making console output safe.
                try:
                    print(line, end='', flush=True)
                except UnicodeEncodeError:
                    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
                    safe = line.encode(encoding, errors='replace').decode(encoding, errors='replace')
                    print(safe, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        if code:
            raise RuntimeError(f'Subprocess exited {code}; see {log}')


def install_workers(root):
    """Reuse base CUDA torch but install optional packages only inside a local venv."""
    import torch, torchvision
    python = worker_python(root)
    if not python.exists():
        run_command([sys.executable, '-m', 'venv', '--system-site-packages', root/'worker_env'], root/'logs/install.log')
    constraints = root/'worker_constraints.txt'
    constraints.write_text(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n'
                           'numpy==1.26.4\ntransformers==4.49.0\ntimm==1.0.15\n'
                           'peft==0.14.0\naccelerate==1.2.1\nsupervision==0.25.1\n'
                           'open_clip_torch==2.32.0\nopencv-python-headless==4.10.0.84\n')
    run_command([python, '-m', 'pip', 'install', '--prefer-binary', '--timeout', '60', '--retries', '2',
                 '-c', constraints, 'ultralytics==8.4.216', 'rfdetr==1.3.0', 'numpy==1.26.4',
                 'transformers==4.49.0', 'timm==1.0.15', 'peft==0.14.0', 'accelerate==1.2.1',
                 'supervision==0.25.1'], root/'logs/install.log')
    run_command([python, '-c', 'import torch; from ultralytics import YOLO; from rfdetr import RFDETRSmall,RFDETRMedium; print(torch.__version__,torch.cuda.is_available())'], root/'logs/install.log')


def source_settings(plan):
    for key, value in plan['settings'].items():
        setattr(config, key, Path(value) if key in ('DATASET_ROOT', 'TEST_DATASET_ROOT', 'TRACKTOR_CACHE_DIR') else value)
    config.TEST_SEQUENCES = []
    config.SAVE_VIDEO = config.SAVE_TEST_VIDEO = False
    config.RUN_SPLIT_TRACKING = True
    config.TRACK_SPLIT = 'val'


def job_for(root, name, plan):
    family = 'yolo26' if name.startswith('yolo') else 'rfdetr'
    return dict(family=family, size=name[-1], settings=plan['settings'],
                cache=str(root/'caches'/name), weights=str(root/'weights'/name),
                image_size=640, floor=.05, recipe=str(root/'sources'/name/'recipe.json'))


def select_threshold(report):
    """Detector F1 selection with explicit precision/recall goals; not tracking optimization."""
    rows = [r for r in report['results'] if r['sequence']=='COMBINED' and r['iou']==.5
            and isinstance(r['score'], (int, float)) and r['f1'] is not None]
    eligible = [r for r in rows if r['precision'] >= .70 and r['recall'] >= .52]
    selected = max(eligible or rows, key=lambda r: (r['f1'], r['recall']))
    return dict(selected=selected, eligible=bool(eligible),
                rule='max F1 with precision>=.70 and recall>=.52; explicit best-F1 fallback')


def configure_source(root, name, plan, *, for_training=False):
    from encoder_experiment import configure
    configure('reid', plan['association_epochs'])
    source_settings(plan)
    folder = root/'sources'/name
    config.RUN_NAME = name
    config.OUTPUT_DIR = folder/'association'
    config.CHECKPOINT_PATH = config.OUTPUT_DIR/'model.pt'
    config.RESUME_TRAINING = True
    config.DETECTION_MIN_SCORE = None
    if name == 'tracktor':
        config.DETECTION_BACKEND = 'tracktor'
    elif name == 'rtdetr':
        config.DETECTION_BACKEND = 'rtdetrv2'
        config.DETECTION_CACHE_DIR = root/'caches'/name
        config.DETECTOR_WEIGHTS_DIR = root/'rtdetr_training/best'
        config.DETECTOR_IMAGE_SIZE = 640
        config.DETECTOR_INPUT_MODE = 'square'
        config.DETECTOR_MIN_SCORE = .05
    else:
        config.DETECTION_BACKEND = 'cached'
        config.DETECTION_CACHE_DIR = root/'caches'/name
        config.CACHED_DETECTOR_RECIPE = json.loads((folder/'recipe.json').read_text())
    if for_training:
        if name != 'tracktor':
            operating = json.loads((folder/'operating_point.json').read_text())
            config.DETECTION_MIN_SCORE = operating['selected']['score']
    config.TRACKING_VARIANT = 'single_pass'
    config.TRACK_OUTPUT_DIR = folder/'tracking/val'
    config.EVALUATION_OUTPUT_DIR = folder/'evaluation/val'
    config.TEST_OUTPUT_DIR = folder/'tracking/test'


def task(root, action, name, plan):
    source_settings(plan)
    folder = root/'sources'/name
    folder.mkdir(parents=True, exist_ok=True)
    if action == 'rtdetr_train':
        spec = importlib.util.spec_from_file_location('suite_rtdetr_trainer', config.PROJECT_ROOT/'detector-train.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(['--output-dir', str(root/'rtdetr_training'), '--epochs', str(plan['detector_epochs']),
                     '--image-size', '640', '--batch-size', '1', '--accumulation', '4',
                     '--lr', '5e-5', '--backbone-lr', '5e-6', '--weight-decay', '1e-5',
                     '--train-fraction', '.4', '--selection-mode', 'sweep', '--scores', *map(str,SCORES),
                     '--min-precision', '.70', '--min-recall', '.52'])
    elif action == 'cache':
        if name == 'rtdetr':
            configure_source(root, name, plan)
            import detect
            with argv(['--backend','rtdetrv2','--split','all']):
                detect.main()
        else:
            job = root/'jobs'/f'{name}.json'
            write_json(job, job_for(root,name,plan))
            run_command([worker_python(root),config.PROJECT_ROOT/'detector_worker.py','--job',job], root/'logs'/f'{name}_worker.log')
    elif action == 'detector_eval':
        if name == 'tracktor':
            configure_source(root,name,plan)
        else:
            configure_source(root,name,plan)
        report_dir = folder/'detector_evaluation'
        if report_dir.exists():
            report_dir.rename(folder/f'detector_evaluation_partial_{time.time_ns()}')
        import evaluate_detector
        values = ['--backend',config.DETECTION_BACKEND,'--output-dir',str(report_dir),'--scores',*map(str,SCORES)]
        if name != 'tracktor':
            values += ['--cache-dir',str(config.DETECTION_CACHE_DIR)]
        with argv(values):
            evaluate_detector.main()
        report = json.loads((report_dir/'summary.json').read_text())
        write_json(folder/'operating_point.json', select_threshold(report))
    elif action == 'association':
        configure_source(root,name,plan,for_training=True)
        import train
        train.main()
    elif action in ('track', 'recover'):
        configure_source(root,name,plan,for_training=True)
        if action == 'recover':
            high = config.DETECTION_MIN_SCORE
            if high is None or high <= .05:
                raise ValueError('Selected source has no low-score interval for recovery')
            config.TRACKING_VARIANT = 'high_low_recovery'
            config.TRACK_OUTPUT_DIR = folder/'recovery/tracking/val'
            config.EVALUATION_OUTPUT_DIR = folder/'recovery/evaluation/val'
        # Restart interrupted tracking in archived folders; never delete old outputs.
        for directory in [config.TRACK_OUTPUT_DIR.parent, config.EVALUATION_OUTPUT_DIR.parent]:
            if directory.exists():
                directory.rename(directory.with_name(directory.name+f'_partial_{time.time_ns()}'))
        import tracking, evaluate
        summaries = tracking.main(recovery_high=high if action=='recover' else None,
                                  recovery_low=.05 if action=='recover' else None)
        metrics = evaluate.main()['MOT17']
        detector_fps = None
        if name != 'tracktor':
            cache_summaries = [
                json.loads((config.DETECTION_CACHE_DIR / config.DATASET_SPLIT / sequence /
                            'summary.json').read_text())
                for sequence in config.VAL_SEQUENCES
            ]
            detector_fps = (sum(row['identity']['frames'] for row in cache_summaries) /
                            sum(row['seconds'] for row in cache_summaries))
        result = dict(source=name, mode=action, metrics=metrics,
                      fps=sum(r['frames'] for r in summaries)/sum(r['seconds'] for r in summaries),
                      detector_fps=detector_fps,
                      checkpoint_sha256=file_digest(config.CHECKPOINT_PATH))
        write_json(folder/('recovery_result.json' if action=='recover' else 'result.json'),result)
    else:
        raise ValueError(action)


def report(root, plan, states):
    lines=['# Sequential detector suite', '',
           'Validation-only selection; not an independent final benchmark. Same OSNet association budget per source.',
           'RT-DETR is MOT-fine-tuned; YOLO/RF-DETR are pretrained. This is not an architecture-only comparison.',
           'Detector confidence chosen by F1 with precision/recall goals; not necessarily optimal for tracking.',
           'Detection cache FPS includes frame reads and export; tracking FPS is end-to-end frame-loop throughput.', '',
           '| Source | Mode | HOTA | IDF1 | MOTA | IDSW | Recall | Detection cache FPS | Tracking FPS |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    results=[]
    for name in plan['sources']:
        for filename in ['result.json','recovery_result.json']:
            path=root/'sources'/name/filename
            if path.exists():
                row=json.loads(path.read_text()); results.append(row); m=row['metrics']
                detector_fps = 'n/a' if row.get('detector_fps') is None else f"{row['detector_fps']:.2f}"
                lines.append(f"| {name} | {row['mode']} | {m['HOTA']} | {m['IDF1']} | {m['MOTA']} | {m['IDSW']} | {m['CLR_Re']} | {detector_fps} | {row['fps']:.2f} |")
    lines += ['', '## Detector operating points (IoU 0.5)', '',
              '| Source | Threshold | Precision | Recall | F1 | Meets selection goals |',
              '| --- | --- | --- | --- | --- | --- |']
    for name in plan['sources']:
        point=root/'sources'/name/'operating_point.json'
        if point.exists():
            op=json.loads(point.read_text()); r=op['selected']
            lines.append(f"| {name} | {r['score']} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.2%} | {op['eligible']} |")
            lines.append(f"\n[{name} detector report](sources/{name}/detector_evaluation/report.md)\n")
    lines += ['', '## Stage status', '']
    lines += [f"- {k}: {v['status']}" + (f" — {v['error']}" if v.get('error') else '') for k,v in states.items()]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(root/'comparison.json',results)
    metric_fields=['HOTA','IDF1','AssA','MOTA','IDSW','CLR_Re','CLR_Pr']
    fields=['source','mode',*metric_fields,'detection_cache_fps','tracking_fps']
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields); writer.writeheader()
        for row in results:
            writer.writerow(dict(source=row['source'],mode=row['mode'],
                                 detection_cache_fps=row.get('detector_fps'),tracking_fps=row['fps'],
                                 **{k:row['metrics'][k] for k in metric_fields}))
    if results:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(12,5)); FigureCanvasAgg(fig)
        axes=fig.subplots(1,2)
        labels=[r['source']+('/recovery' if r['mode']=='recover' else '') for r in results]
        for metric in ['HOTA','IDF1','MOTA']:
            axes[0].plot(labels,[float(r['metrics'][metric]) for r in results],marker='o',label=metric)
        axes[0].set(ylabel='Validation score (%)',title='Higher is better'); axes[0].legend()
        axes[1].bar(labels,[float(r['metrics']['IDSW']) for r in results])
        axes[1].set(ylabel='Identity switches',title='Lower is better')
        for ax in axes: ax.tick_params(axis='x',rotation=45)
        fig.tight_layout(); fig.savefig(root/'comparison.png',dpi=140)


def stage(root, key, states, command, artifacts, plan):
    """Resume completed stages only while their recorded artifact hashes still match."""
    previous=states.get(key,{})
    if previous.get('status')=='complete' and all(Path(p).is_file() and file_digest(p)==h for p,h in previous.get('artifacts',{}).items()) and previous.get('artifacts'):
        print(f'SKIP complete: {key}',flush=True)
        return True
    states[key]={'status':'running','started':datetime.now().isoformat()}
    write_json(root/'state.json',states)
    try:
        run_command(command,root/'logs'/f'{key}.log')
        for path in artifacts:
            if not path.is_file():
                raise RuntimeError(f'{key} did not produce its required artifact: {path}')
        states[key]={'status':'complete','artifacts':{str(p):file_digest(p) for p in artifacts}}
        ok=True
    except Exception as error:
        states[key]={'status':'failed','error':str(error)}
        ok=False
    write_json(root/'state.json',states)
    report(root,plan,states)
    return ok


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=config.PROJECT_ROOT/'outputs/detector_suite_v1')
    parser.add_argument('--association-epochs',type=int,default=4)
    parser.add_argument('--detector-epochs',type=int,default=3)
    parser.add_argument('--sizes',choices=['s','sm'],default='s',help='s: small models (default); sm: also test medium models')
    parser.add_argument('--check-only',action='store_true',help='Prepare dependencies/assets and smoke-check models, without training')
    parser.add_argument('--task',choices=['rtdetr_train','cache','detector_eval','association','track','recover'],help=argparse.SUPPRESS)
    parser.add_argument('--source',help=argparse.SUPPRESS)
    args=parser.parse_args()
    root=args.run_dir.resolve()
    if args.task:
        task(root,args.task,args.source,json.loads((root/'plan.json').read_text()))
        return
    if min(args.association_epochs,args.detector_epochs)<1:
        parser.error('epoch budgets must be positive')
    if os.name == 'nt':
        import atexit, ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        atexit.register(lambda: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000))
    # Base setup is intentionally separate from optional worker dependencies.
    from prepare_ablation import preflight,download_verified,TRACKTOR_URL,REID_URL,REID_SHA256
    config.TEST_SEQUENCES=[]
    config.DETECTION_BACKEND='tracktor'
    preflight(['reid', 'detr'])
    root.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    import atexit
    run_lock=FileLock(str(root/'suite.lock'))
    run_lock.acquire(timeout=0)
    atexit.register(run_lock.release)
    free=shutil.disk_usage(root).free
    if free < 8*1024**3:
        raise RuntimeError('At least 8 GiB free disk space is required for models, caches and results')
    sources=['tracktor','rtdetr']+[f'yolo26_{s}' for s in args.sizes]+[f'rfdetr_{s}' for s in args.sizes]
    import torch
    keys=['DATASET_ROOT','DATASET_SPLIT','TEST_DATASET_ROOT','TEST_DATASET_SPLIT','TRAIN_SEQUENCES','VAL_SEQUENCES',
          'DEVICE','SEED','ASSOCIATION_DISTANCE_PX','MATCH_THRESHOLD','NEW_TRACK_MIN_SCORE','LOOKBACK_FRAMES',
          'MAX_MISSED_FRAMES','USE_MOTION_PREDICTION','USE_APPEARANCE_HISTORY','APPEARANCE_HISTORY_SIZE',
          'TRACK_CONFIRM_MIN_HITS','TRACK_CONFIRM_WINDOW','TRAIN_BATCH_SIZE','VAL_BATCH_SIZE','LEARNING_RATE']
    plan=json.loads(json.dumps(dict(version=1,sources=sources,association_epochs=args.association_epochs,
                  detector_epochs=args.detector_epochs,settings={k:getattr(config,k) for k in keys},
                  torch=str(torch.__version__),scores=SCORES),default=str))
    plan['settings']['TRACKTOR_CACHE_DIR'] = str(root/'caches/tracktor')
    path=root/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Run settings changed. Choose a fresh --run-dir to preserve comparable results.')
    write_json(path,plan)
    source_settings(plan)
    download_verified(TRACKTOR_URL,config.TRACKTOR_ARCHIVE,config.TRACKTOR_ARCHIVE_SHA256)
    download_verified(REID_URL,config.REID_WEIGHTS_PATH,REID_SHA256)
    from zipfile import ZipFile
    from import_tracktor import import_sequence
    with ZipFile(config.TRACKTOR_ARCHIVE) as archive:
        for name in dict.fromkeys(config.TRAIN_SEQUENCES+config.VAL_SEQUENCES):
            import_sequence(archive,config.DATASET_ROOT/config.DATASET_SPLIT/name)
    states=json.loads((root/'state.json').read_text()) if (root/'state.json').exists() else {}
    if not stage(root,'core_smoke',states,[sys.executable,__file__,'--run-dir',root,'--task-helper','core'],[root/'core_ready.json'],plan):
        raise SystemExit('Core detector/encoder check failed before training. Inspect logs/core_smoke.log.')
    # Prepare extras before long training so missing APIs/downloads are found now.
    install_marker=root/'worker_ready.json'
    if not stage(root,'install_workers',states,[sys.executable,__file__,'--run-dir',root,'--task-helper','install'],[install_marker,worker_python(root)],plan):
        raise SystemExit('Worker setup failed before training. See logs/install.log and install_workers.log.')
    ready=[]
    for name in sources[2:]:
        job=root/'jobs'/f'{name}.json'; write_json(job,job_for(root,name,plan))
        marker=root/'sources'/name/'smoke_ready.json'
        command=[sys.executable,__file__,'--run-dir',root,'--task-helper','smoke','--source',name]
        if stage(root,f'{name}_smoke',states,command,[marker],plan): ready.append(name)
    if args.check_only:
        print(f'Preflight finished. See {root / "comparison.md"}. No training launched.')
        if any(v['status']!='complete' for v in states.values()):
            raise SystemExit('Some preflight checks failed; inspect comparison.md before leaving a run unattended.')
        return
    def child(action,name):
        return [sys.executable,__file__,'--run-dir',root,'--task',action,'--source',name]
    # Keep display awake during the unattended experiment; restore on exit.
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for name in sources:
            if name not in ('tracktor','rtdetr') and name not in ready: continue
            folder=root/'sources'/name
            if name=='rtdetr' and not stage(root,'rtdetr_train',states,child('rtdetr_train',name),[root/'rtdetr_training/best/training_manifest.json',root/'rtdetr_training/best/model.safetensors'],plan): continue
            if name!='tracktor':
                artifacts=[root/'caches'/name/config.DATASET_SPLIT/seq/'summary.json' for seq in config.TRAIN_SEQUENCES+config.VAL_SEQUENCES]
                if not stage(root,f'{name}_cache',states,child('cache',name),artifacts,plan): continue
            if not stage(root,f'{name}_detector_eval',states,child('detector_eval',name),[folder/'detector_evaluation/summary.json',folder/'operating_point.json'],plan): continue
            if not stage(root,f'{name}_association',states,child('association',name),[folder/'association/model.pt',folder/'association/last.pt'],plan): continue
            stage(root,f'{name}_track',states,child('track',name),[folder/'result.json'],plan)
        candidates=[]
        for name in sources:
            path=root/'sources'/name/'result.json'
            if name!='tracktor' and states.get(f'{name}_track',{}).get('status')=='complete' and path.exists():
                operating=json.loads((root/'sources'/name/'operating_point.json').read_text())
                if operating['selected']['score']>.05:
                    candidates.append(json.loads(path.read_text()))
        if candidates:
            best=max(candidates,key=lambda r:(float(r['metrics']['HOTA']),float(r['metrics']['IDF1'])))
            write_json(root/'recovery_selection.json',dict(source=best['source'],criterion='highest validation HOTA, IDF1 tie-break; same validation reused'))
            stage(root,'E_recovery',states,child('recover',best['source']),[root/'sources'/best['source']/'recovery_result.json'],plan)
        else:
            states['E_recovery']={'status':'blocked','error':'No successful confidence-scored detector baseline with a low-score interval'}
        write_json(root/'state.json',states); report(root,plan,states)
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print(f'Finished. Read {root / "comparison.md"}',flush=True)
    if any(v['status']!='complete' for v in states.values()):
        raise SystemExit('Some stages failed or were blocked; completed experiments are preserved. Rerun to retry.')


def helper():
    # Helpers run in clean subprocesses; no GPU context leaks between models.
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--task-helper',required=True)
    parser.add_argument('--source')
    args=parser.parse_args()
    root=args.run_dir.resolve(); plan=json.loads((root/'plan.json').read_text())
    source_settings(plan)
    if args.task_helper=='install':
        install_workers(root)
        write_json(root/'worker_ready.json',dict(python=str(worker_python(root)),ready=True))
    elif args.task_helper=='core':
        from encoder_experiment import configure, smoke
        configure('reid')
        source_settings(plan)
        smoke()
        config.DETECTION_BACKEND='rtdetrv2'
        config.DETECTOR_WEIGHTS_DIR=None
        config.DETECTOR_IMAGE_SIZE=640
        config.DETECTOR_INPUT_MODE='square'
        from detect import PersonDetector
        from Utils.data import load_image_frames, read_image
        detector=PersonDetector()
        folder=config.DATASET_ROOT/config.DATASET_SPLIT/config.VAL_SEQUENCES[0]
        boxes,scores=detector.predict(read_image(next(iter(load_image_frames(folder).values()))))
        print(f'RT-DETR 640 smoke PASS: {len(boxes)} person boxes')
        write_json(root/'core_ready.json',dict(ready=True))
    else:
        job=root/'jobs'/f'{args.source}.json'
        run_command([worker_python(root),config.PROJECT_ROOT/'detector_worker.py','--job',job,'--smoke'],root/'logs'/f'{args.source}_smoke_worker.log')
        write_json(root/'sources'/args.source/'smoke_ready.json',dict(ready=True))


if __name__=='__main__':
    if '--task-helper' in sys.argv: helper()
    else: main()
