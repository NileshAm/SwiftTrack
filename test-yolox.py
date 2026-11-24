"""One-frame YOLOX integration check: inference, preview and GT evaluation.

This is a functionality check, not a full validation benchmark.
Run python test-yolox.py in dragon-track before exporting full videos.
"""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np

from Utils import config
from Utils.data import load_ground_truth
from evaluate_detector import metrics, score_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Optional existing official MOT20 weights')
    parser.add_argument('--device', default=config.DEVICE)
    args = parser.parse_args(argv)
    detection = runpy.run_path(str(Path(__file__).with_name('detect-yolox.py')))
    options = ['--split','val','--smoke-test','--device',args.device]
    if args.checkpoint:
        options += ['--checkpoint',str(args.checkpoint)]
    output = detection['main'](options)
    result = json.loads((output/'result.json').read_text())
    image = Path(result['frame'])
    sequence = image.parent.parent
    truth = load_ground_truth(sequence, config.GT_FILE, box_origin=config.BOX_ORIGIN)[int(image.stem)]
    boxes = np.asarray(result['boxes'], dtype=np.float32).reshape(-1,4)
    scores = np.asarray(result['scores'], dtype=np.float32)
    rows = [dict(iou=iou, score=score, **metrics(score_frame(boxes[scores>=score],truth,iou)))
            for iou in (.5,.75) for score in (.1,.2,.3,.4,.5,.6,.7)]
    report = dict(scope='one-frame functionality check only; not full validation accuracy',
                  frame=str(image), recipe=result['recipe'], results=rows)
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'One-frame evaluation: {output / "evaluation.json"}')
    return output


if __name__ == '__main__':
    main()
