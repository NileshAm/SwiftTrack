"""Evaluate a completed direct-cosine Tracktor + OSNet validation run.

Example: python evaluate-tracktor-osnet.py outputs/tracktor_osnet_cosine/RUN_FOLDER
"""
import argparse
import json
from pathlib import Path

from Utils import config
from Utils.detections import file_digest, validate_tracktor
from Utils.outputs import tracking_context
from Utils.splits import split_manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path,help='Folder printed by track-tracktor-osnet.py')
    args=parser.parse_args(argv)
    root=args.run_dir.resolve()
    settings_path=root/'settings.json'
    if not settings_path.is_file():
        parser.error(f'Missing tracking settings: {settings_path}')
    settings=json.loads(settings_path.read_text())
    if settings.get('split')!='val' or settings.get('detector',{}).get('backend')!='tracktor':
        parser.error('This is not a Tracktor + OSNet validation run')
    config.TRACKTOR_CACHE_DIR=Path(settings['cache_dir'])
    config.DETECTION_BACKEND='tracktor'
    config.DETECTION_MIN_SCORE=settings['detection_min_score']
    config.TRACKING_VARIANT='tracktor_osnet_cosine'
    config.OUTPUT_DIR=root
    config.TRACK_OUTPUT_DIR=root/'tracking'/'val'
    config.EVALUATION_OUTPUT_DIR=root/'evaluation'/'val'
    config.TRACK_SPLIT='val'
    config.SAVE_PLOTS=False
    for sequence in config.VAL_SEQUENCES:
        context=tracking_context(sequence,'val')
        cache=validate_tracktor(context['sequence_path'])
        summary_path=context['output_dir']/'summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(f'Track {sequence} first: {summary_path}')
        summary=json.loads(summary_path.read_text())
        if (summary.get('detection_file')!=str(cache)
                or summary.get('detection_sha256')!=file_digest(cache)
                or summary.get('method')!='frozen_osnet_cosine_no_gcn'):
            raise ValueError(f'{sequence}: tracking output is not from the current Tracktor cache and direct matcher')
    import evaluate
    return evaluate.main()


if __name__=='__main__':
    main()
