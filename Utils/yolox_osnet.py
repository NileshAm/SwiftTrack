"""Direct OSNet cosine association and verification of pretrained YOLOX caches."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from Utils.data import _sequence_info
from Utils.detections import file_digest

# Published CrowdHuman + MOT20 YOLOX-X checkpoint, not the MOT fine-tune export.
PRETRAINED_SHA256 = '9ce38a6048470fd2597e1aff6959c302e7297dd9cae1ce3b9db1af615ca6dac4'


def cache_for_sequence(cache_dir, sequence_path):
    sequence_path = Path(sequence_path)
    path = Path(cache_dir) / sequence_path.parent.name / sequence_path.name / 'det.txt'
    summary_path = path.with_name('summary.json')
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f'Missing complete YOLOX cache: {path}. Run detect-yolox.py first.')
    summary = json.loads(summary_path.read_text())
    info = _sequence_info(sequence_path)
    identity = summary.get('identity', {})
    detector = identity.get('detector', {})
    if (summary.get('complete') is not True
            or identity.get('sequence_path') != str(sequence_path.resolve())
            or identity.get('frames') != info.getint('seqLength')
            or identity.get('width') != info.getint('imWidth')
            or identity.get('height') != info.getint('imHeight')
            or detector.get('backend') != 'yolox'
            or detector.get('checkpoint_sha256') != PRETRAINED_SHA256
            or detector.get('box_origin') != 1
            or 'fine_tuning' in detector
            or summary.get('sha256') != file_digest(path)):
        raise ValueError(f'Incomplete, changed, or non-pretrained YOLOX cache: {path}')
    return path.resolve(), summary


def cache_for_checkpoint(cache_dir, sequence_path, checkpoint_sha256):
    """Validate a YOLOX cache exported with the selected checkpoint."""
    sequence_path = Path(sequence_path)
    path = Path(cache_dir) / sequence_path.parent.name / sequence_path.name / 'det.txt'
    summary_path = path.with_name('summary.json')
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f'Missing complete YOLOX cache: {path}')
    summary = json.loads(summary_path.read_text())
    info = _sequence_info(sequence_path)
    identity = summary.get('identity', {})
    detector = identity.get('detector', {})
    if (summary.get('complete') is not True
            or identity.get('sequence_path') != str(sequence_path.resolve())
            or identity.get('frames') != info.getint('seqLength')
            or identity.get('width') != info.getint('imWidth')
            or identity.get('height') != info.getint('imHeight')
            or detector.get('backend') != 'yolox'
            or detector.get('checkpoint_sha256') != checkpoint_sha256
            or detector.get('box_origin') != 1
            or summary.get('sha256') != file_digest(path)):
        raise ValueError(f'Changed, incomplete, or wrong-checkpoint YOLOX cache: {path}')
    return path.resolve(), summary


def cosine_matches(track_features, detection_features, predicted_boxes, detection_boxes,
                   *, min_similarity=0.7, center_distance=1.5):
    """One-to-one matching on OSNet cosine; gate by predicted center distance.

    center_distance is a multiplier of the larger person's height, with a 25px
    minimum. It limits implausible jumps without using a graph or learned scores.
    """
    t, d = len(track_features), len(detection_features)
    if not t or not d:
        return [], list(range(t)), list(range(d))
    if not 0 <= min_similarity <= 1 or center_distance <= 0:
        raise ValueError('Invalid OSNet association thresholds')
    a = torch.nn.functional.normalize(torch.as_tensor(track_features, dtype=torch.float32), dim=1)
    b = torch.nn.functional.normalize(torch.as_tensor(detection_features, dtype=torch.float32), dim=1)
    similarity = (a @ b.T).numpy()
    if not np.isfinite(similarity).all():
        raise ValueError('Nonfinite OSNet similarity')
    previous = np.asarray(predicted_boxes, dtype=np.float32).reshape(t, 4)
    current = np.asarray(detection_boxes, dtype=np.float32).reshape(d, 4)
    centers_a = (previous[:, :2] + previous[:, 2:]) / 2
    centers_b = (current[:, :2] + current[:, 2:]) / 2
    distance = np.linalg.norm(centers_a[:, None] - centers_b[None], axis=2)
    height = np.maximum(previous[:, 3] - previous[:, 1], 1)
    other_height = np.maximum(current[:, 3] - current[:, 1], 1)
    gate = np.maximum(25., center_distance * np.maximum(height[:, None], other_height[None]))
    allowed = (similarity >= min_similarity) & (distance <= gate)
    # A forbidden pair cannot displace a valid match in the assignment.
    rows, cols = linear_sum_assignment(np.where(allowed, 1-similarity, 1e6))
    matches = [(int(i), int(j)) for i, j in zip(rows, cols) if allowed[i, j]]
    used_t, used_d = {i for i, _ in matches}, {j for _, j in matches}
    return matches, [i for i in range(t) if i not in used_t], [j for j in range(d) if j not in used_d]
