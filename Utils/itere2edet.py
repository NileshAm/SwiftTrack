"""Validate completed Iter-E2EDET detection caches before tracking."""

import json
from pathlib import Path

from Utils.data import _sequence_info
from Utils.detections import file_digest


def cache_for_sequence(cache_dir, sequence_path, checkpoint):
    sequence_path = Path(sequence_path)
    checkpoint = Path(checkpoint)
    path = Path(cache_dir) / sequence_path.parent.name / sequence_path.name / "det.txt"
    summary_path = path.with_name("summary.json")
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing Iter-E2EDET cache: {path}. Run detect-itere2edet.py first.")
    summary = json.loads(summary_path.read_text())
    identity = summary.get("identity", {})
    recipe = identity.get("detector", {})
    info = _sequence_info(sequence_path)
    if (summary.get("complete") is not True
            or identity.get("sequence_path") != str(sequence_path.resolve())
            or identity.get("frames") != info.getint("seqLength")
            or identity.get("width") != info.getint("imWidth")
            or identity.get("height") != info.getint("imHeight")
            or recipe.get("format") != "mot_person_v1"
            or recipe.get("backend") != "iter_e2edet"
            or recipe.get("box_origin") != 1
            or recipe.get("checkpoint_sha256") != file_digest(checkpoint)
            or summary.get("sha256") != file_digest(path)):
        raise ValueError(f"Changed or incompatible Iter-E2EDET cache: {path}")
    return path.resolve(), summary
