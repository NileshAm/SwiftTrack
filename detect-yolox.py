"""Export pretrained CrowdHuman + MOT20 YOLOX-X detections; --smoke-test checks one frame."""

import argparse
import csv
import fcntl
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
import torch
from tqdm.auto import tqdm

from Utils import config
from Utils.data import _sequence_info, load_image_frames, read_image
from Utils.yolox import YOLOXDetector
from Utils.detections import file_digest
from Utils.splits import validate_video_splits, video_key


def identity(sequence_path, recipe):
    info = _sequence_info(sequence_path)
    return dict(sequence_path=str(sequence_path.resolve()), frames=info.getint('seqLength'),
                width=info.getint('imWidth'), height=info.getint('imHeight'), detector=recipe)


def validate_existing(path, expected):
    saved = json.loads(path.with_name('summary.json').read_text())
    if (saved.get('complete') is not True or saved.get('identity') != expected
            or not path.is_file() or saved.get('sha256') != file_digest(path)):
        raise ValueError(f'Changed/incompatible YOLOX cache: {path}. Choose a fresh --cache-dir.')


def preview(image, boxes, scores, path):
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    for box, score in zip(boxes, scores, strict=True):
        draw.rectangle(box.tolist(), outline='lime', width=2)
        draw.text(tuple(map(float, box[:2])), f'person {score:.2f}', fill='lime')
    picture.save(path)


def run_sequence(sequence_path, path, detector, overall):
    """Atomic full-video export; interrupted videos restart, verified complete videos skip."""
    expected = identity(sequence_path, detector.recipe)
    frames = load_image_frames(sequence_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start, count = time.perf_counter(), 0
    pending = path.with_name('det.txt.partial')
    with pending.open('w', newline='') as stream, tqdm(
            frames.values(), total=len(frames), desc=sequence_path.name, unit='frame',
            position=1, leave=False, disable=not config.SHOW_PROGRESS) as bar:
        writer = csv.writer(stream)
        for frame in bar:
            image = read_image(frame)
            boxes, scores = detector.predict(image)
            for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
                writer.writerow([frame['frame_id'], -1, f'{x1+1:.6f}', f'{y1+1:.6f}',
                                 f'{x2-x1:.6f}', f'{y2-y1:.6f}', f'{score:.8f}', -1, -1, -1])
            if frame['frame_id'] == 1:
                preview(image, boxes, scores, path.with_name('preview.jpg'))
            count += len(boxes)
            bar.set_postfix(people=len(boxes), saved=count, refresh=False)
            overall.update(1)
    pending.replace(path)
    seconds = time.perf_counter() - start
    summary = dict(complete=True, identity=expected, sha256=file_digest(path), mot_rows=count,
                   seconds=seconds, fps=len(frames)/seconds, person_class_id=detector.person_id,
                   device=str(next(detector.model.parameters()).device),
                   timing='image reads, inference, export and preview; excludes model loading')
    pending_summary = path.with_name('summary.json.partial')
    pending_summary.write_text(json.dumps(summary, indent=2)+'\n')
    pending_summary.replace(path.with_name('summary.json'))
    tqdm.write(f'Saved {count} YOLOX person boxes: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train','val','trainval','test','all'), default='val')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=896)
    parser.add_argument('--nms-iou', type=float, default=.7)
    parser.add_argument('--min-score', type=float, default=.10, help='Cache floor; sweep higher scores afterward')
    parser.add_argument('--cache-dir', type=Path, help='Default: separate directory per resize/score/NMS')
    parser.add_argument('--weights-dir', type=Path, default=config.PROJECT_ROOT/'outputs'/'yolox_weights')
    parser.add_argument('--checkpoint', type=Path, help='Official checkpoint or best.pth exported by detector-train-yolox.py')
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--smoke-test', action='store_true', help='Only first frame of first selected video; creates no detection cache')
    args = parser.parse_args(argv)
    if not 0 <= args.min_score <= 1 or min(args.width,args.height) < 1:
        parser.error('sizes/counts must be positive; min-score must be in [0,1]')
    if not 0 < args.nms_iou <= 1:
        parser.error('nms-iou must be in (0,1]')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu for a smoke check only.')
    validate_video_splits()
    groups = {'train': (config.DATASET_ROOT, config.DATASET_SPLIT, config.TRAIN_SEQUENCES),
              'val': (config.DATASET_ROOT, config.DATASET_SPLIT, config.VAL_SEQUENCES),
              'test': (config.TEST_DATASET_ROOT, config.TEST_DATASET_SPLIT, config.TEST_SEQUENCES)}
    sequences, seen = [], set()
    for split, (root, dataset_split, names) in groups.items():
        if args.split not in ('all', split) and not (args.split == 'trainval' and split in ('train', 'val')):
            continue
        for name in names:
            video_key(name)
            path = (Path(root)/dataset_split/name).resolve()
            if path not in seen:
                seen.add(path)
                sequences.append(path)
    if not sequences:
        raise ValueError('No sequences selected')
    detector = YOLOXDetector(args)
    tqdm.write(f'Loaded {detector.recipe["model"]}: {args.width}x{args.height} aspect-preserving input, '
               f'confidence floor {args.min_score:g}, device {args.device}')
    if args.smoke_test:
        frame = load_image_frames(sequences[0], (1,1))[1]
        image = read_image(frame)
        boxes, scores = detector.predict(image)
        out = config.PROJECT_ROOT/'outputs'/'yolox_smoke'/str(time.time_ns())
        out.mkdir(parents=True)
        preview(image, boxes, scores, out/'preview.jpg')
        (out/'result.json').write_text(json.dumps(dict(recipe=detector.recipe, frame=str(frame['image_path']),
                                                     boxes=boxes.tolist(), scores=scores.tolist()), indent=2)+'\n')
        tqdm.write(f'YOLOX smoke check: {len(boxes)} people. Preview: {out / "preview.jpg"}')
        return out
    cache = args.cache_dir or config.PROJECT_ROOT/'detections'/f'yolox_x_crowdhuman_mot20_{args.width}x{args.height}_score{args.min_score:g}_nms{args.nms_iou:g}'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/'.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another YOLOX export is using this cache') from None
        jobs, destinations = [], set()
        for sequence in sequences:
            path = cache/sequence.parent.name/sequence.name/'det.txt'
            if path.resolve() in destinations:
                raise ValueError('Selected datasets share a cache destination; use separate runs')
            destinations.add(path.resolve())
            expected = identity(sequence, detector.recipe)
            if path.with_name('summary.json').exists():
                validate_existing(path, expected)
                tqdm.write(f'Skipping complete YOLOX cache: {sequence.name}')
            else:
                jobs.append((sequence,path,expected['frames']))
        with tqdm(total=sum(n for _,_,n in jobs), desc='YOLOX detection overall', unit='frame',
                  smoothing=0, disable=not config.SHOW_PROGRESS) as overall:
            for i,(sequence,path,_) in enumerate(jobs,1):
                overall.set_postfix(video=f'{i}/{len(jobs)}', sequence=sequence.name)
                run_sequence(sequence,path,detector,overall)
    tqdm.write(f'YOLOX cache: {cache}. Evaluate with --backend yolox --cache-dir pointing here.')
    return cache


if __name__ == '__main__':
    main()
