"""Baseline comparison: two detectors, three encoders, with/without GCN.

Tracktor and YOLOX are the default detection sources. --detections gt retains
the earlier GT-box control. GT IDs are used only by TrackEval after tracking.
No-GCN bypasses graph attention but keeps the same checkpoint's Sinkhorn matcher.
The fine-tuned OSNet arm transfers the frozen-OSNet graph checkpoint; this
project has no association checkpoint trained on the fine-tuned OSNet features.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile

import torch
from tqdm.auto import tqdm

from Utils import config, mot_metrics as diagnostic
from Utils.data import load_ground_truth, load_image_frames, load_sequence, read_image
from Utils.detections import file_digest, validate_tracktor
from Utils.features import extract_features, load_encoder
from Utils.outputs import tracking_context
from Utils.yolox_osnet import cache_for_sequence as yolox_cache_for_sequence
from model import AssociationModel, MODEL_VERSION
import tracking


ROOT=config.PROJECT_ROOT
DEFAULT_ASSOCIATION={
    'frozen_osnet':ROOT/'outputs/encoder_comparison_reid_v1/model.pt',
    'finetuned_osnet':ROOT/'outputs/encoder_comparison_reid_v1/model.pt',
    'densenet':ROOT/'outputs/encoder_comparison_densenet_v1/model.pt',
}
DEFAULT_FINETUNED_OSNET=ROOT/'outputs/osnet_finetune/lr_1e-05/best.pt'
VARIANTS=('frozen_osnet','finetuned_osnet','densenet')


def configure(encoder):
    name,width={'densenet':('densenet121',1024),'reid':('osnet_reid',512)}[encoder]
    config.ENCODER_NAME=name
    config.FEATURE_DIM=width
    config.FREEZE_ENCODER=True
    config.USE_PRETRAINED_WEIGHTS=True


def gt_cache(sequence_path,folder):
    """Write person boxes without IDs, scores from the original MOT GT file."""
    folder.mkdir(parents=True)
    path=folder/'det.txt'
    truth=load_ground_truth(sequence_path)
    with path.open('x') as stream:
        for frame,record in truth.items():
            for x1,y1,x2,y2 in record['boxes'][record['valid_mask']]:
                stream.write(f'{frame},-1,{x1+1:.6f},{y1+1:.6f},{x2-x1:.6f},{y2-y1:.6f},1,-1,-1,-1\n')
    return path


def load_weights(variant,association_path,osnet_path,device,detection_min_score,new_track_min_score):
    configure('densenet' if variant=='densenet' else 'reid')
    config.DEVICE=device
    config.DETECTION_BACKEND='supplied'
    config.DETECTION_MIN_SCORE=detection_min_score
    config.NEW_TRACK_MIN_SCORE=new_track_min_score
    config.BOX_ORIGIN=1
    checkpoint=torch.load(association_path,map_location='cpu',weights_only=True)
    if checkpoint.get('model_version')!=MODEL_VERSION:
        raise ValueError(f'{association_path}: incompatible graph model version')
    saved_settings=checkpoint['settings']
    if saved_settings['ENCODER_NAME']!=config.ENCODER_NAME or saved_settings['FEATURE_DIM']!=config.FEATURE_DIM:
        raise ValueError(f'{association_path}: checkpoint uses a different encoder or feature width')
    for key in ('CROP_SIZE','GRAPH_HIDDEN_DIMS','ASSOCIATION_DISTANCE_PX',
                'SINKHORN_ITERATIONS','SINKHORN_SCORE_SCALE','SINKHORN_UNMATCHED_LOGIT'):
        if key in saved_settings:
            setattr(config,key,saved_settings[key])
    config.MATCH_THRESHOLD=saved_settings['MATCH_THRESHOLD']
    encoder=load_encoder(device)
    if encoder.feature_settings!=checkpoint['feature_settings']:
        raise ValueError(f'{association_path}: encoder weights or preprocessing differ from the checkpoint')
    transfer=False
    if variant=='finetuned_osnet':
        saved=torch.load(osnet_path,map_location='cpu',weights_only=True)
        recipe=saved.get('recipe',{})
        if (recipe.get('version')!='mot_osnet_triplet_v1' or
                recipe.get('base_features',{}).get('weights_sha256')!=encoder.feature_settings['weights_sha256']):
            raise ValueError(f'{osnet_path}: incompatible fine-tuned OSNet weights')
        encoder.network.load_state_dict(saved['encoder_state_dict'],strict=True)
        encoder.feature_settings={**encoder.feature_settings,
                                  'fine_tuned_checkpoint_sha256':file_digest(osnet_path),
                                  'fine_tuned_epoch':saved['epoch']}
        transfer=True
    encoder.eval()
    model=AssociationModel(**checkpoint['model_settings']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval()
    return model,encoder,checkpoint,transfer


def save_table(root,rows,settings):
    fields=['detector','encoder','graph','status','HOTA','DetA','AssA','IDF1','MOTA','IDSW',
            'consistent_pct','single_id_people','people','checkpoint_epoch','transfer','error']
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    lines=['# Tracker and detector baseline comparison','',
           'Validation MOT17-02-SDP and MOT17-09-SDP. GT person IDs are never supplied to the tracker.',
           f"Detection read score: {settings['detection_min_score']}; new-track score: {settings['new_track_min_score']}.",
           'Tracktor detection scores are placeholders (1.0); the score filter only has meaning for YOLOX.',
           'No-GCN bypasses attention only; both arms use the same learned Sinkhorn matcher and checkpoint.',
           'Fine-tuned OSNet uses the graph trained on frozen OSNet features (transfer); no dedicated fine-tuned graph checkpoint exists.',
           'These are repeatedly used validation sequences; treat the table as an internal baseline, not an independent test score.','',
           '| Detector | Encoder | Graph | Status | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Consistent ID |',
           '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| '+' | '.join(str(row.get(key,'')) for key in
                    ('detector','encoder','graph','status','HOTA','DetA','AssA','IDF1','MOTA','IDSW','consistent_pct'))+' |')
        if row.get('error'):
            lines.append(f"\n{row['detector']} {row['encoder']} {row['graph']}: {row['error']}")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def resolve_sources(names,sequences,args,root):
    """Validate each cached detector file before any long tracking arm starts."""
    resolved={}
    config.TRACKTOR_CACHE_DIR=args.tracktor_cache_dir
    for detector in names:
        entries={}
        for sequence in sequences:
            sequence_path=config.DATASET_ROOT/config.DATASET_SPLIT/sequence
            if detector=='tracktor':
                path=validate_tracktor(sequence_path)
                summary=json.loads(path.with_name('summary.json').read_text())
                identity={'backend':'tracktor','source':summary['identity']['source'],
                          'score_note':'Tracktor confidence is a placeholder 1.0'}
                digest=summary['sha256']
            elif detector=='yolox':
                path,summary=yolox_cache_for_sequence(args.yolox_cache_dir,sequence_path)
                identity=summary['identity']['detector']
                if args.detection_min_score<identity['min_score']:
                    raise ValueError(f'{sequence}: read score is below the YOLOX cache floor')
                digest=summary['sha256']
            else:
                path=gt_cache(sequence_path,root/'gt_detection_cache'/sequence)
                identity={'backend':'ground_truth_boxes','score_note':'IDs excluded; score 1.0'}
                digest=file_digest(path)
            entries[sequence]={'path':path.resolve(),'sha256':digest,'identity':identity}
        resolved[detector]=entries
    return resolved


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder',choices=('all',*VARIANTS),default='all')
    parser.add_argument('--graph',choices=('both','on','off'),default='both')
    parser.add_argument('--detections',choices=('both','tracktor','yolox','gt'),default='both')
    parser.add_argument('--tracktor-cache-dir',type=Path,default=config.TRACKTOR_CACHE_DIR)
    parser.add_argument('--yolox-cache-dir',type=Path,default=ROOT/'detections/yolox_x_mot20_val_v1')
    parser.add_argument('--detection-min-score',type=float,default=0.5)
    parser.add_argument('--new-track-min-score',type=float,default=0.7)
    parser.add_argument('--frozen-osnet-association',type=Path,default=DEFAULT_ASSOCIATION['frozen_osnet'])
    parser.add_argument('--finetuned-osnet-association',type=Path,default=DEFAULT_ASSOCIATION['finetuned_osnet'])
    parser.add_argument('--densenet-association',type=Path,default=DEFAULT_ASSOCIATION['densenet'])
    parser.add_argument('--finetuned-osnet-weights',type=Path,default=DEFAULT_FINETUNED_OSNET)
    parser.add_argument('--device',default=config.DEVICE)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--smoke-test',action='store_true',help='Check each selected arm on one frame; no TrackEval')
    args=parser.parse_args(argv)
    if not 0<=args.detection_min_score<=1 or not 0<=args.new_track_min_score<=1:
        parser.error('Detection and new-track scores must be in [0,1]')
    variants=VARIANTS if args.encoder=='all' else (args.encoder,)
    graphs=(True,False) if args.graph=='both' else (args.graph=='on',)
    detectors=('tracktor','yolox') if args.detections=='both' else (args.detections,)
    association_paths={'frozen_osnet':args.frozen_osnet_association,
                       'finetuned_osnet':args.finetuned_osnet_association,
                       'densenet':args.densenet_association}
    for variant in variants:
        for path in [association_paths[variant]]+([args.finetuned_osnet_weights] if variant=='finetuned_osnet' else []):
            if not path.is_file():parser.error(f'Missing weights: {path}')
    if torch.device(args.device).type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; choose --device cpu')
    sequences=tuple(diagnostic.SEQUENCES)
    if any(sequence not in config.VAL_SEQUENCES for sequence in sequences):
        parser.error(f'This diagnostic requires {sequences} in config.VAL_SEQUENCES')
    if args.smoke_test:
        import numpy as np
        with tempfile.TemporaryDirectory(prefix='tracker-baseline-smoke-') as temporary:
            sources=resolve_sources(detectors,sequences,args,Path(temporary))
            sequence=sequences[0]
            path=config.DATASET_ROOT/config.DATASET_SPLIT/sequence
            for detector in detectors:
                entry=sources[detector][sequence]
                frame=load_sequence(path,entry['path'],(1,1),box_origin=1,
                                    min_score=args.detection_min_score)[1]
                image=read_image(frame)
                boxes=frame['boxes'][:3]
                for variant in variants:
                    model,encoder,_,_=load_weights(variant,association_paths[variant],
                        args.finetuned_osnet_weights,args.device,args.detection_min_score,
                        args.new_track_min_score)
                    features,valid=extract_features(image,boxes,encoder,show_progress=False)
                    for use_graph in graphs:
                        results=tracking.associate_frame(model,boxes[valid.numpy()],boxes[valid.numpy()],
                            features[valid],features[valid],np.ones(int(valid.sum())),
                            image.shape[:2],use_graph=use_graph)
                        print(f'PASS {detector} {variant} graph={use_graph}: {len(results[0])} matches')
                    del model,encoder
        return
    root=args.output_dir or ROOT/'outputs/tracker_detector_baselines'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if root.exists():raise FileExistsError(f'Choose a new output directory: {root}')
    root.mkdir(parents=True)
    sources=resolve_sources(detectors,sequences,args,root)
    settings={'sequences':sequences,'detectors':detectors,'encoders':variants,
              'graph_modes':['with_gcn' if x else 'without_gcn' for x in graphs],
              'detection_min_score':args.detection_min_score,
              'new_track_min_score':args.new_track_min_score,
              'sources':{detector:{seq:{**entry,'path':str(entry['path'])} for seq,entry in entries.items()}
                         for detector,entries in sources.items()}}
    (root/'baseline_manifest.json').write_text(json.dumps(settings,indent=2,default=str)+'\n')
    rows=[]
    with tqdm(total=len(detectors)*len(variants)*len(graphs),desc='Baseline comparisons',unit='arm') as parent:
        for detector in detectors:
            for variant in variants:
                model,encoder,checkpoint,transfer=load_weights(
                    variant,association_paths[variant],args.finetuned_osnet_weights,args.device,
                    args.detection_min_score,args.new_track_min_score)
                for use_graph in graphs:
                    graph='with_gcn' if use_graph else 'without_gcn'
                    folder=root/detector/variant/graph
                    config.OUTPUT_DIR=folder
                    config.TRACK_OUTPUT_DIR=folder/'tracking/val'
                    config.EVALUATION_OUTPUT_DIR=folder/'evaluation/val'
                    config.SAVE_VIDEO=False
                    row={'detector':detector,'encoder':variant,'graph':graph,'status':'running',
                         'checkpoint_epoch':checkpoint['epoch'],'transfer':transfer}
                    rows.append(row);save_table(root,rows,settings)
                    folder.mkdir(parents=True,exist_ok=True)
                    (folder/'manifest.json').write_text(json.dumps({
                        'detector':detector,'detection_files':settings['sources'][detector],
                        'ids_passed_to_tracker':False,'encoder':encoder.feature_settings,
                        'association_checkpoint':str(association_paths[variant].resolve()),
                        'association_sha256':file_digest(association_paths[variant]),
                        'checkpoint_epoch':checkpoint['epoch'],'use_graph':use_graph,
                        'transfer_from_frozen_osnet_association':transfer,
                        'match_threshold':config.MATCH_THRESHOLD,
                        'detection_min_score':args.detection_min_score,
                        'new_track_min_score':args.new_track_min_score,
                        'device':args.device},indent=2,default=str)+'\n')
                    total=sum(diagnostic.SEQUENCES[sequence] for sequence in sequences)
                    with tqdm(total=total,desc=f'{detector} {variant} {graph}',unit='frame') as overall:
                        for sequence in sequences:
                            config.DETECTION_FILE=sources[detector][sequence]['path']
                            tracking.run_sequence(sequence,model,encoder,split='val',overall=overall,use_graph=use_graph)
                            inputs=folder/'evaluation/val/inputs/MOT17'
                            gt_source=config.DATASET_ROOT/config.DATASET_SPLIT/sequence/'gt/gt.txt'
                            gt_target=inputs/'gt'/sequence/'gt/gt.txt'
                            gt_target.parent.mkdir(parents=True,exist_ok=True)
                            shutil.copyfile(gt_source,gt_target)
                            track_target=inputs/'trackers/DragonTrack/data'/f'{sequence}.txt'
                            track_target.parent.mkdir(parents=True,exist_ok=True)
                            shutil.copyfile(config.TRACK_OUTPUT_DIR/sequence/'tracks.txt',track_target)
                    measured=diagnostic.load(folder)
                    scores=diagnostic.score(measured)
                    association=diagnostic.association_only(measured)['overall']
                    row.update({key:scores[key] for key in ('HOTA','DetA','AssA','IDF1','MOTA','IDSW')})
                    row.update(consistent_pct=association['consistent_pct'],
                               single_id_people=association['single_id_people'],people=association['people'],
                               status='complete')
                    (folder/'report.json').write_text(json.dumps({
                        'metrics':scores,'association_only':association,
                        'per_sequence':{sequence:diagnostic.score({sequence:data}) for sequence,data in measured.items()}
                        },indent=2)+'\n')
                    save_table(root,rows,settings)
                    parent.update(1)
                del model,encoder
                if torch.cuda.is_available():torch.cuda.empty_cache()
    print(f'Comparison: {root/"comparison.md"}')
    return root


if __name__=='__main__':main()
