"""Compare cached Iter-E2EDET and YOLOX boxes on small MOT17 people."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from Utils import config
from Utils.data import load_ground_truth, load_sequence, match_detections_to_gt
from Utils.itere2edet import cache_for_sequence as iter_cache
from Utils.yolox_osnet import cache_for_sequence as yolox_cache


def inspected_frames(sequence, iter_path, yolox_path, score):
    itr = load_sequence(sequence, iter_path, box_origin=1, min_score=score)
    yolo = load_sequence(sequence, yolox_path, box_origin=1, min_score=score)
    gt = load_ground_truth(sequence)
    for frame_id, item in itr.items():
        truth = gt[frame_id]
        small = truth['valid_mask'] & ((truth['boxes'][:, 3] - truth['boxes'][:, 1]) < 50)
        if not small.any():
            continue
        iter_ids = set(match_detections_to_gt(item['boxes'], truth)['ids'])
        yolo_ids = set(match_detections_to_gt(yolo[frame_id]['boxes'], truth)['ids'])
        missed = [int(identity) for identity in truth['ids'][small] if identity not in iter_ids]
        recovered = [identity for identity in missed if identity in yolo_ids]
        yield dict(sequence=sequence.name, frame=frame_id, small_gt=int(small.sum()),
                   iter_missed=len(missed), yolox_recovered=len(recovered)), item, yolo[frame_id], truth, recovered


def render(item, yolo, truth, recovered, target, score):
    image = Image.open(item['image_path']).convert('RGB')
    panels = []
    for title, record, color in [('Iter-E2EDET', item, '#00e678'), ('YOLOX-X', yolo, '#4ba3ff')]:
        panel = image.copy()
        draw = ImageDraw.Draw(panel)
        for box in record['boxes']:
            draw.rectangle(tuple(map(float, box)), outline=color, width=2)
        for box, identity, valid in zip(truth['boxes'], truth['ids'], truth['valid_mask']):
            if valid and int(identity) in recovered:
                draw.rectangle(tuple(map(float, box)), outline='#ff3b30', width=4)
                draw.text((float(box[0]), max(0., float(box[1]) - 13)), f'GT {identity}', fill='#ff3b30')
        draw.text((15, 15), f'{title} | score >= {score} | red: Iter missed, YOLOX matched', fill='yellow')
        panels.append(panel)
    combined = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)))
    offset = 0
    for panel in panels:
        combined.paste(panel, (offset, 0))
        offset += panel.width
    combined.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iter-checkpoint', type=Path, required=True)
    parser.add_argument('--iter-cache-dir', type=Path, default=config.PROJECT_ROOT/'detections/iter_e2edet_crowdhuman_v1')
    parser.add_argument('--yolox-cache-dir', type=Path, default=config.PROJECT_ROOT/'detections/yolox_x_mot20_val_v1')
    parser.add_argument('--score', type=float, default=0.3)
    parser.add_argument('--images', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=config.PROJECT_ROOT/'outputs/iter_e2edet_diagnosis')
    args = parser.parse_args()
    if not 0 <= args.score <= 1 or args.images < 1:
        parser.error('Score must be in [0,1] and --images must be positive')
    candidates = []
    for name in config.VAL_SEQUENCES:
        sequence = config.DATASET_ROOT/config.DATASET_SPLIT/name
        iter_path, iter_summary = iter_cache(args.iter_cache_dir, sequence, args.iter_checkpoint)
        yolo_path, yolo_summary = yolox_cache(args.yolox_cache_dir, sequence)
        if args.score < max(iter_summary['identity']['detector']['min_score'],
                            yolo_summary['identity']['detector']['min_score']):
            parser.error(f'{name}: requested score is below a cache floor')
        candidates.extend(inspected_frames(sequence, iter_path, yolo_path, args.score))
    candidates.sort(key=lambda item: (item[0]['yolox_recovered'], item[0]['iter_missed']), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'small_person_frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence', 'frame', 'small_gt', 'iter_missed', 'yolox_recovered'])
        writer.writeheader()
        writer.writerows(item[0] for item in candidates)
    for row, itr, yolo, truth, recovered in candidates[:args.images]:
        target = args.output_dir/f"{row['sequence']}_{row['frame']:06d}.jpg"
        render(itr, yolo, truth, recovered, target, args.score)
        print(target)
    print(f"Frame rankings: {args.output_dir/'small_person_frames.csv'}")


if __name__ == '__main__':
    main()
