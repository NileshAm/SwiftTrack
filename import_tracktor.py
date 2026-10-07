"""Import authors-linked Tracktor boxes: python import_tracktor.py [--archive ZIP].

No images/GT/model inference. Discard upstream track IDs; map missing (-1)
confidence to a neutral weight of 1, explicitly recorded in cache metadata.
"""
import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from tqdm.auto import tqdm
from Utils import config
from Utils.data import _sequence_info
from Utils.detections import file_digest, tracktor_path, tracktor_identity, validate_tracktor


def import_sequence(archive, sequence_path):
    sequence_path = Path(sequence_path)
    path = tracktor_path(sequence_path)
    if path.exists() or path.with_name('summary.json').exists():
        return validate_tracktor(sequence_path)
    identity = tracktor_identity(sequence_path)
    member = f'MOT17Labels/{sequence_path.parent.name}/{sequence_path.name}/det/tracktor_prepr_det.txt'
    with archive.open(member) as stream:
        reader = csv.reader(io.TextIOWrapper(stream))
        rows = []
        for number, row in enumerate(reader, 1):
            values = np.asarray(row, dtype=np.float64)
            if len(values) != 10 or not np.isfinite(values).all():
                raise ValueError(f'{member}:{number}: expected 10 finite MOT columns')
            frame = values[0]
            if frame != int(frame) or not 1 <= frame <= identity['frames'] or min(values[4:6]) <= 0:
                raise ValueError(f'{member}:{number}: invalid frame or box')
            score = values[6]
            if score != -1 and not 0 <= score <= 1:
                raise ValueError(f'{member}:{number}: unsupported confidence {score}')
            # Upstream loader explicitly subtracts 1 from x/y: this is MOT origin 1.
            values[1] = -1
            values[6] = 1.0 if score == -1 else score
            rows.append(values)
    rows.sort(key=lambda row: row[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)
    summary = {'complete': True, 'identity': identity, 'source_member': member,
               'rows': len(rows), 'sha256': file_digest(path)}
    path.with_name('summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=config.TRACKTOR_ARCHIVE)
    args = parser.parse_args()
    if file_digest(args.archive) != config.TRACKTOR_ARCHIVE_SHA256:
        raise ValueError('Archive checksum differs from the inspected authors-linked archive')
    jobs = [config.DATASET_ROOT/config.DATASET_SPLIT/name
            for name in dict.fromkeys(config.TRAIN_SEQUENCES+config.VAL_SEQUENCES)]
    jobs += [config.TEST_DATASET_ROOT/config.TEST_DATASET_SPLIT/name for name in config.TEST_SEQUENCES]
    jobs = list(dict.fromkeys(jobs))
    with ZipFile(args.archive) as archive:
        for sequence in tqdm(jobs, desc='Import Tracktor detections', unit='video',
                             disable=not config.SHOW_PROGRESS, dynamic_ncols=True):
            import_sequence(archive, sequence)
    print(f'Imported/verified {len(jobs)} videos in {config.TRACKTOR_CACHE_DIR}')


if __name__ == '__main__':
    main()
