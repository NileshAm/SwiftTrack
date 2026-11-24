"""Evaluate frozen OSNet + cosine tracking with validation GT boxes as detections.

Only box coordinates are supplied to the tracker; person IDs are never supplied.
This is an analysis control on labeled validation data, not a deployable score.
"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil

from tqdm.auto import tqdm
import torch

from Utils import config
from Utils.detections import file_digest
from Utils.features import load_encoder
from Utils.outputs import tracking_context


def load_script(filename, name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_gt_detection_cache(data, folder):
    folder.mkdir(parents=True)
    path=folder/'det.txt'
    with path.open('x') as stream:
        for frame,boxes in enumerate(data['gt_dets'],start=1):
            for x,y,w,h in boxes:
                stream.write(f'{frame},-1,{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,-1,-1,-1\n')
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--similarity',type=float,default=.7)
    parser.add_argument('--center-distance',type=float,default=1.5)
    parser.add_argument('--device',default=config.DEVICE)
    args=parser.parse_args(argv)
    if not 0<=args.similarity<=1 or args.center_distance<=0:
        parser.error('Similarity must be in [0,1] and center distance must be positive')
    if torch.device(args.device).type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; choose --device cpu')
    out=args.output_dir or config.PROJECT_ROOT/'outputs/tracker_gt_boxes'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if out.exists():
        raise FileExistsError(f'Choose a fresh output directory: {out}')
    diagnostic=load_script('diagnose-tracking.py','diagnostic')
    tracker=load_script('track-tracktor-osnet.py','tracktor_cosine')
    # Use the same GT preprocessing and validation sequences as the fixed-box report.
    source=diagnostic.load(diagnostic.DEFAULT_TRACKTOR)
    args.output_dir=out
    args.detection_min_score=0.0
    args.new_track_min_score=0.7
    args.cache_dir=out/'gt_detection_cache'
    args.save_video=False
    tracker.configure(args)
    config.TRACKING_VARIANT='gt_boxes_osnet_cosine'
    config.DETECTION_BACKEND='supplied'
    encoder=load_encoder()
    expected=sum(data['num_timesteps'] for data in source.values())
    with tqdm(total=expected,desc='GT-box tracker control',unit='frame') as overall:
        for sequence,data in source.items():
            context=tracking_context(sequence,'val')
            cache=write_gt_detection_cache(data,out/'gt_detection_cache'/sequence)
            summary={'identity':{'detector':{'backend':'ground_truth_boxes','min_score':0.0,
                     'source':'TrackEval preprocessed validation pedestrian GT; IDs excluded'}},
                     'sha256':file_digest(cache)}
            tracker.run_sequence(sequence,'val',encoder,args,cache,summary,overall)
            evaluation_root=out/'evaluation/val/inputs/MOT17'
            gt_source=diagnostic.DEFAULT_TRACKTOR/'evaluation/val/inputs/MOT17/gt'/sequence/'gt/gt.txt'
            gt_target=evaluation_root/'gt'/sequence/'gt/gt.txt'
            gt_target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(gt_source,gt_target)
            track_target=evaluation_root/'trackers/DragonTrack/data'/f'{sequence}.txt'
            track_target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(context['output_dir']/'tracks.txt',track_target)
    measured=diagnostic.load(out)
    scores=diagnostic.score(measured)
    association=diagnostic.association_only(measured)
    report={'experiment':'validation GT boxes as detections, IDs hidden from tracker',
            'sequences':list(source),'settings':{'similarity':args.similarity,
            'center_distance':args.center_distance,'new_track_min_score':args.new_track_min_score,
            'encoder':'frozen OSNet; direct cosine, Kalman motion, appearance history'},
            'metrics':scores,'association_on_matched_boxes':association}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    values=association['overall']
    (out/'report.md').write_text('\n'.join([
        '# Tracker with ground-truth boxes', '',
        'Validation person boxes were supplied without identity labels. This measures the existing OSNet/cosine tracker with detection errors removed. It is an analysis control, not a deployable result.',
        '',f"HOTA **{scores['HOTA']}**, DetA **{scores['DetA']}**, AssA **{scores['AssA']}**, IDF1 **{scores['IDF1']}**, MOTA **{scores['MOTA']}**, identity switches **{scores['IDSW']}**.",
        '',f"Of {values['matched_boxes']} matched boxes, {values['consistent_boxes']} ({values['consistent_pct']}%) retain one globally consistent identity; {values['single_id_people']}/{values['people']} people receive only one ID.",
        '',f"Identity changes: {values['identity_changes']}; after a gap: {values['identity_changes_after_gap']}.",
        '', 'See `report.json` for settings and per-sequence counts.', ''])+'\n')
    print(f'GT-box tracker report: {out/"report.md"}')
    return out


if __name__=='__main__':main()
