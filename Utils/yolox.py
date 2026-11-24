"""Official CrowdHuman + MOT20 YOLOX-X, detector only, MMDetection 3.3."""
from hashlib import sha256
from pathlib import Path
import fcntl
import urllib.request
import warnings

import numpy as np
from tqdm.auto import tqdm
from Utils.detections import file_digest

URL = ('https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/'
       'bytetrack_yolox_x_crowdhuman_mot20-private_20220506_101040-9ce38a60.pth')
CONFIG = 'bytetrack_yolox_x_8xb4-80e_crowdhuman-mot20train_test-mot20test.py'


def checkpoint_path(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / URL.rsplit('/', 1)[-1]
    with (directory/'.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            pending = path.with_suffix('.pth.partial')
            with urllib.request.urlopen(URL, timeout=60) as response, pending.open('wb') as stream:
                with tqdm(total=int(response.headers.get('Content-Length', 0)) or None,
                          unit='B', unit_scale=True, desc='YOLOX weights') as bar:
                    while chunk := response.read(1024*1024):
                        stream.write(chunk)
                        bar.update(len(chunk))
            if not file_digest(pending).startswith('9ce38a60'):
                raise ValueError('Official YOLOX checkpoint checksum mismatch')
            pending.replace(path)
        if not file_digest(path).startswith('9ce38a60'):
            raise ValueError(f'Official YOLOX checkpoint checksum mismatch: {path}')
    return path


class YOLOXDetector:
    def __init__(self, args, *, trainable=False):
        import torch
        import mmdet
        import mmcv
        from mmengine.config import Config
        from mmdet.apis import init_detector, inference_detector
        if mmdet.__version__ != '3.3.0':
            raise RuntimeError('Use the existing dragon-track MMDetection 3.3.0 installation')
        source = Path(mmdet.__file__).parent/'.mim/configs/bytetrack'/CONFIG
        original = Config.fromfile(source)
        detector = original.model.detector.copy()
        detector.pop('init_cfg', None)
        detector.backbone.pop('init_cfg', None)
        # Official tracking preprocessing: raw BGR, no RGB conversion or normalization.
        detector.data_preprocessor = dict(type='DetDataPreprocessor', pad_size_divisor=32)
        detector.test_cfg.score_thr = args.min_score
        detector.test_cfg.nms.iou_threshold = args.nms_iou
        cfg = Config(dict(model=detector, default_scope='mmdet', test_dataloader=dict(dataset=dict(type='CocoDataset', metainfo=dict(classes=('person',)), pipeline=[
            dict(type='LoadImageFromNDArray'),
            dict(type='Resize', scale=(args.width, args.height), keep_ratio=True),
            dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.,114.,114.))),
            dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor')),
        ]))))
        path = args.checkpoint or checkpoint_path(args.weights_dir)
        digest = file_digest(path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        state = saved['state_dict']
        # The official file wraps detector weights in a ByteTrack module.
        if digest.startswith('9ce38a60'):
            state = {k.removeprefix('detector.'): v for k,v in state.items() if k.startswith('detector.')}
            if not state:
                raise ValueError('Expected official ByteTrack detector.* state keys')
        elif saved.get('meta', {}).get('yolox_training', {}).get('version') != 'mot_yolox_finetune_v1':
            raise ValueError('Use the official MOT20 checkpoint or a local export from detector-train-yolox.py')
        with warnings.catch_warnings(record=True) as notices:
            self.model = init_detector(cfg, checkpoint=None, device='cpu')
        for notice in notices:
            if 'checkpoint is None, use COCO classes by default.' not in str(notice.message):
                warnings.warn(str(notice.message))
        self.model.load_state_dict(state, strict=True)
        self.model.dataset_meta = dict(classes=('person',))
        self.model.to(args.device).eval()
        if not trainable:
            self.model.requires_grad_(False)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.inference, self.person_id, self.min_score = inference_detector, 0, args.min_score
        self.recipe = dict(format='mot_person_v1', backend='yolox', model='YOLOX-X-CrowdHuman-MOT20',
            training_data='CrowdHuman train+val and MOT20 train; upstream validation on MOT17 train',
            checkpoint_sha256=digest, checkpoint_url=URL, config_sha256=sha256(cfg.pretty_text.encode()).hexdigest(),
            resize_width=args.width, resize_height=args.height, keep_ratio=True, pad_divisor=32,
            pad_value=114, color='BGR', min_score=args.min_score, nms_iou=args.nms_iou,
            box_origin=1, precision='float32_tf32_disabled', mmdet=mmdet.__version__,
            mmcv=mmcv.__version__, torch=str(torch.__version__))
        if not digest.startswith('9ce38a60'):
            self.recipe['fine_tuning'] = saved['meta']['yolox_training']

    def predict(self, image):
        import torch
        with torch.inference_mode():
            result = self.inference(self.model, np.ascontiguousarray(image[..., ::-1])).pred_instances
        selected = (result.labels == 0) & (result.scores >= self.min_score)
        boxes = result.bboxes[selected].cpu().numpy().copy()
        scores = result.scores[selected].cpu().numpy().copy()
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite YOLOX predictions')
        h,w = image.shape[:2]
        boxes[:,0::2] = boxes[:,0::2].clip(0,w)
        boxes[:,1::2] = boxes[:,1::2].clip(0,h)
        valid = (boxes[:,2:]-boxes[:,:2] > 1e-5).all(axis=1)
        return boxes[valid], scores[valid]
