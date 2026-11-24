"""Track author-linked Tracktor boxes with frozen OSNet and direct cosine.

This uses the exact matching and track-state code of track-yolox-osnet.py.
Run evaluate-tracktor-osnet.py separately for full validation metrics.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from Utils import config
from Utils.data import load_sequence, read_image
from Utils.features import load_encoder, extract_features
from Utils.outputs import tracking_context
from Utils.splits import split_manifest, validate_video_splits
from Utils.track_filter import export_confirmed
from Utils.track_state import tracker_settings, predict_track_boxes, update_tracks, expire_tracks
from Utils.yolox_osnet import cosine_matches
from Utils.detections import validate_tracktor, file_digest
from tracking import draw_tracks, make_evaluation_video


def cache_for_sequence(cache_dir, sequence_path):
    config.TRACKTOR_CACHE_DIR = Path(cache_dir)
    path = validate_tracktor(sequence_path)
    metadata = json.loads(path.with_name('summary.json').read_text())
    return path, {'identity': {'detector': {'backend': 'tracktor',
        'min_score': 0.0, 'source': metadata['identity']['source']}},
        'sha256': file_digest(path)}


def configure(args):
    config.ENCODER_NAME = 'osnet_reid'
    config.FEATURE_DIM = 512
    config.FREEZE_ENCODER = config.USE_PRETRAINED_WEIGHTS = True
    config.DEVICE = args.device
    config.DETECTION_MIN_SCORE = args.detection_min_score
    config.NEW_TRACK_MIN_SCORE = args.new_track_min_score
    config.TRACKING_VARIANT = 'tracktor_osnet_cosine'
    config.DETECTION_BACKEND = 'tracktor'
    config.TRACKTOR_CACHE_DIR = args.cache_dir
    config.TRACK_SPLIT = 'val'
    config.SAVE_VIDEO = args.save_video
    config.SAVE_TEST_VIDEO = args.save_video
    config.SAVE_PLOTS = False
    config.OUTPUT_DIR = args.output_dir
    config.TRACK_OUTPUT_DIR = args.output_dir / 'tracking' / 'val'
    config.TEST_OUTPUT_DIR = args.output_dir / 'tracking' / 'test'
    config.EVALUATION_OUTPUT_DIR = args.output_dir / 'evaluation' / 'val'


@torch.no_grad()
def run_sequence(sequence, split, encoder, args, cache_path, cache_summary, overall):
    context = tracking_context(sequence, split)
    folder = context['output_dir']
    if folder.exists():
        raise FileExistsError(f'Output already exists: {folder}. Choose a fresh --output-dir.')
    folder.mkdir(parents=True)
    frames = load_sequence(context['sequence_path'], cache_path, context['frame_range'],
                           box_origin=1, min_score=args.detection_min_score)
    tracks, next_id, raw_rows, confirmed = [], 1, 0, set()
    started = time.perf_counter()
    pending = folder / '.tracks.pending.txt'
    with pending.open('x') as stream, tqdm(frames.values(), total=len(frames),
            desc=f'OSNet {sequence}', unit='frame', position=1, leave=False,
            disable=not config.SHOW_PROGRESS) as bar:
        for frame in bar:
            image = read_image(frame)
            features, valid = extract_features(image, frame['boxes'], encoder, show_progress=False)
            keep = valid.numpy()
            current = dict(frame, boxes=frame['boxes'][keep], scores=frame['scores'][keep],
                           row_ids=frame['row_ids'][keep])
            features = features[valid]
            tracks = expire_tracks(tracks, frame['frame_id'])
            predicted = predict_track_boxes(tracks, frame['frame_id'])
            track_features = torch.stack([track['feature'] for track in tracks]) if tracks else torch.empty((0, 512))
            matches, unmatched_tracks, unmatched_detections = cosine_matches(
                track_features, features, predicted, current['boxes'],
                min_similarity=args.similarity, center_distance=args.center_distance)
            tracks, next_id, observations = update_tracks(
                tracks, matches, unmatched_tracks, unmatched_detections, current, features, next_id)
            for observation in observations:
                if observation['confirmed']:
                    confirmed.add(observation['id'])
                x1, y1, x2, y2 = observation['box']
                stream.write(f"{frame['frame_id']},{observation['id']},{x1+1:.3f},{y1+1:.3f},"
                             f"{x2-x1:.3f},{y2-y1:.3f},{observation['score']:.6f},-1,-1,-1\n")
            raw_rows += len(observations)
            overall.update(1)
            bar.set_postfix(detections=len(features), matched=len(matches), ids=next_id-1, refresh=False)
    rows = export_confirmed(pending, folder / 'tracks.txt', confirmed)
    pending.unlink()
    Image.fromarray(draw_tracks(image, [x for x in observations if x['id'] in confirmed])).save(folder / 'preview.jpg')
    seconds = time.perf_counter()-started
    summary = dict(sequence=sequence, split=split, complete=True, frames=len(frames),
        frame_range=context['frame_range'], dataset_root=str(context['dataset_root'].resolve()),
        dataset_split=context['dataset_split'], data_split=split_manifest(),
        mot_rows=rows, raw_mot_rows=raw_rows, ids_created=next_id-1,
        seconds=seconds, fps=len(frames)/seconds, save_video=context['save_video'],
        detection_file=str(cache_path), detections=cache_summary['identity']['detector'],
        detection_sha256=cache_summary['sha256'], encoder=encoder.feature_settings,
        method='frozen_osnet_cosine_no_gcn', similarity_threshold=args.similarity,
        center_distance_height_multiplier=args.center_distance,
        new_track_min_score=args.new_track_min_score,
        read_detection_min_score=args.detection_min_score,
        tracker_settings=tracker_settings())
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, default=str)+'\n')
    tqdm.write(f"{sequence}: {rows} observations, {next_id-1} IDs, {summary['fps']:.2f} FPS")
    if context['save_video']:
        make_evaluation_video(sequence, split, overall=overall)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, default=config.TRACKTOR_CACHE_DIR)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--split', choices=('val','test'), default='val')
    p.add_argument('--evaluate', action='store_true', help='Run TrackEval after full validation tracking; separate evaluation script is also available')
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--smoke-test', action='store_true', help='Check two frames and direct OSNet matching; save no tracks')
    p.add_argument('--similarity', type=float, default=.7, help='Minimum OSNet cosine similarity')
    p.add_argument('--center-distance', type=float, default=1.5, help='Motion gate in person heights; minimum 25px')
    p.add_argument('--detection-min-score', type=float, default=.1)
    p.add_argument('--new-track-min-score', type=float, default=.5)
    p.add_argument('--device', default=config.DEVICE)
    args=p.parse_args(argv)
    if not 0 <= args.similarity <= 1 or not 0 <= args.detection_min_score <= 1 or not 0 <= args.new_track_min_score <= 1 or args.center_distance <= 0:
        p.error('Invalid threshold; probabilities must be in [0,1] and center distance > 0')
    if args.split == 'test' and args.evaluate:
        p.error('--evaluate requires --split val')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a small check')
    args.output_dir = args.output_dir or config.PROJECT_ROOT/'outputs'/'tracktor_osnet_cosine'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    configure(args)
    validate_video_splits()
    tracker_settings()
    names = config.VAL_SEQUENCES if args.split == 'val' else config.TEST_SEQUENCES
    if not names:
        raise ValueError(f'No {args.split} videos configured')
    jobs=[]
    for name in (names[:1] if args.smoke_test else names):
        context=tracking_context(name,args.split)
        cache,summary=cache_for_sequence(args.cache_dir,context['sequence_path'])
        if args.detection_min_score < summary['identity']['detector']['min_score']:
            raise ValueError('Detection read score is below the cache floor')
        if not args.smoke_test and context['output_dir'].exists():
            raise FileExistsError(f'Output already exists: {context["output_dir"]}')
        jobs.append((name,context,cache,summary))
    if any(job[3]['identity']['detector'] != jobs[0][3]['identity']['detector'] for job in jobs[1:]):
        raise ValueError('Selected videos have different detector recipes')
    encoder=load_encoder()
    if args.smoke_test:
        name,context,cache,_=jobs[0]
        first=context['frame_range'][0]
        frame=load_sequence(context['sequence_path'],cache,(first,first),box_origin=1,
                            min_score=args.detection_min_score)[first]
        features,valid=extract_features(read_image(frame),frame['boxes'],encoder,show_progress=False)
        second=first+1
        if second <= context['frame_range'][1]:
            later=load_sequence(context['sequence_path'],cache,(second,second),box_origin=1,
                                min_score=args.detection_min_score)[second]
            later_features,later_valid=extract_features(read_image(later),later['boxes'],encoder,show_progress=False)
            matches,_,_=cosine_matches(features[valid],later_features[later_valid],
                frame['boxes'][valid.numpy()],later['boxes'][later_valid.numpy()],
                min_similarity=args.similarity,center_distance=args.center_distance)
            print(f'PASS: Tracktor cache + frozen OSNet, first frame {int(valid.sum())} usable, next frame {int(later_valid.sum())} usable, {len(matches)} direct matches; feature shape {tuple(features.shape)}')
        else:
            print(f'PASS: Tracktor cache + frozen OSNet, {int(valid.sum())} usable detections, feature shape {tuple(features.shape)}')
        return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'settings.json').write_text(json.dumps(dict(
        cache_dir=str(args.cache_dir.resolve()), split=args.split, similarity=args.similarity,
        center_distance=args.center_distance, detection_min_score=args.detection_min_score,
        new_track_min_score=args.new_track_min_score, device=args.device,
        encoder=encoder.feature_settings, detector=jobs[0][3]['identity']['detector']),indent=2,default=str)+'\n')
    total=sum(c['frame_range'][1]-c['frame_range'][0]+1 for _,c,_,_ in jobs)
    if args.save_video: total *= 2
    with tqdm(total=total,desc='Tracktor + OSNet overall',unit='frame',disable=not config.SHOW_PROGRESS) as overall:
        for name,_,cache,summary in jobs:
            run_sequence(name,args.split,encoder,args,cache,summary,overall)
    if args.evaluate:
        import evaluate
        evaluate.main()
    print(f'Results: {args.output_dir}')
    return args.output_dir


if __name__ == '__main__':
    main()
