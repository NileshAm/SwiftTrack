"""Separate MMDetection DDQ inference; imports stay lazy for RT-DETR users."""

from hashlib import sha256
from pathlib import Path
import urllib.request
import warnings

import numpy as np
from tqdm.auto import tqdm

from Utils.detections import file_digest

MODELS = {
    '4scale': ('ddq-detr-4scale_r50_8xb2-12e_coco.py',
               'https://download.openmmlab.com/mmdetection/v3.0/ddq/'
               'ddq-detr-4scale_r50_8xb2-12e_coco/'
               'ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth', '42528127'),
    '5scale': ('ddq-detr-5scale_r50_8xb2-12e_coco.py',
               'https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_5scale_coco_1x.pth', None),
}


def download_checkpoint(variant, directory):
    """Download the official COCO weights atomically, checking the published hash when present."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _, url, expected = MODELS[variant]
    path = directory / url.rsplit('/', 1)[-1]
    with (directory / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            pending = path.with_suffix('.pth.partial')
            with urllib.request.urlopen(url, timeout=60) as response, pending.open('wb') as stream:
                total = int(response.headers.get('Content-Length', 0)) or None
                with tqdm(total=total, unit='B', unit_scale=True, desc='DDQ weights') as bar:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        bar.update(len(chunk))
            if expected and not file_digest(pending).startswith(expected):
                raise ValueError(f'DDQ download checksum mismatch: {pending}')
            pending.replace(path)
        if expected and not file_digest(path).startswith(expected):
            raise ValueError(f'DDQ checkpoint checksum mismatch: {path}')
    return path


def prepare_config(variant, input_mode, image_size, short_side, max_size, max_detections):
    """Use MMDetection's pinned architecture with an image-only inference pipeline."""
    import mmdet
    from mmengine.config import Config
    if mmdet.__version__ != '3.3.0':
        raise RuntimeError('DDQ uses mmdet==3.3.0; see Documentation/Implementation/27-ddq.md')
    path = Path(mmdet.__file__).parent / '.mim' / 'configs' / 'ddq' / MODELS[variant][0]
    if not path.is_file():
        raise FileNotFoundError(f'MMDetection wheel configs missing: {path}')
    cfg = Config.fromfile(path)
    cfg.model.backbone.init_cfg = None  # All weights come from the detector checkpoint.
    cfg.model.test_cfg.max_per_img = max_detections
    resize = (dict(type='Resize', scale=(image_size, image_size), keep_ratio=False)
              if input_mode == 'square' else
              dict(type='Resize', scale=(max_size, short_side), keep_ratio=True))
    # No LoadAnnotations or dataset access during inference.
    cfg.test_dataloader.dataset.pipeline = [
        dict(type='LoadImageFromNDArray'), resize,
        dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
    ]
    return cfg


class DDQDetector:
    """RGB input -> clipped original-resolution person boxes; internal DDQ query selection retained."""

    def __init__(self, args, *, trainable=False):
        import torch
        import mmcv
        import mmengine
        import mmdet
        from mmdet.apis import init_detector, inference_detector
        cfg = prepare_config(args.variant, args.input_mode, args.image_size, args.short_side,
                             args.max_size, args.max_detections)
        checkpoint = args.checkpoint or download_checkpoint(args.variant, args.weights_dir)
        checkpoint = Path(checkpoint).resolve()
        # Only load trusted official/user-specified MMDetection checkpoints.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        with warnings.catch_warnings(record=True) as notices:
            self.model = init_detector(cfg, checkpoint=None, device='cpu')
        for notice in notices:
            if 'checkpoint is None, use COCO classes by default.' not in str(notice.message):
                warnings.warn_explicit(notice.message, notice.category, notice.filename, notice.lineno)
        state = saved.get('state_dict', saved)
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, 'module.')
        # DDQ training attaches this shared head to the decoder during forward().
        # Published checkpoints therefore contain aliases absent before that call.
        # Remove only verified identical duplicates; all real weights remain strict.
        for key in list(state):
            if key.startswith('decoder.aux_reg_branches.'):
                original = key.replace('decoder.', 'bbox_head.', 1)
                if original not in state or not torch.equal(state[key], state[original]):
                    raise ValueError(f'DDQ checkpoint has inconsistent shared weights: {key}')
                del state[key]
        self.model.load_state_dict(state, strict=True)  # Never silently run partly random weights.
        classes = saved.get('meta', {}).get('dataset_meta', {}).get('classes')
        classes = classes or saved.get('meta', {}).get('CLASSES') or self.model.dataset_meta['classes']
        ids = [i for i, name in enumerate(classes) if name.lower() == 'person']
        if len(ids) != 1:
            raise ValueError('DDQ checkpoint must have exactly one person class')
        self.person_id = ids[0]
        self.model.to(args.device).eval()
        if not trainable:
            self.model.requires_grad_(False)
        self.inference = inference_detector
        self.min_score = args.min_score
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.recipe = {
            'format': 'mot_person_v1', 'backend': 'ddq', 'model': f'DDQ-DETR-R50-{args.variant}',
            'checkpoint_sha256': file_digest(checkpoint), 'config_sha256': sha256(cfg.pretty_text.encode()).hexdigest(),
            'input_mode': args.input_mode, 'image_size': args.image_size if args.input_mode == 'square' else None,
            'short_side': args.short_side if args.input_mode == 'aspect' else None,
            'max_size': args.max_size if args.input_mode == 'aspect' else None,
            'min_score': args.min_score, 'max_detections': args.max_detections,
            'class': 'person', 'box_origin': 1, 'precision': 'float32_tf32_disabled',
            'postprocessing': 'official_ddq_distinct_query_selection_topk_clip_no_extra_nms',
            'mmdet': mmdet.__version__, 'mmcv': mmcv.__version__, 'mmengine': mmengine.__version__,
            'torch': str(torch.__version__),
        }

    def predict(self, image):
        import torch
        with torch.inference_mode():
            # MMDetection's ndarray pipeline expects BGR; Utils.data supplies RGB.
            result = self.inference(self.model, np.ascontiguousarray(image[..., ::-1])).pred_instances
        selected = (result.labels == self.person_id) & (result.scores >= self.min_score)
        boxes = result.bboxes[selected].detach().cpu().numpy().copy()
        scores = result.scores[selected].detach().cpu().numpy().copy()
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise RuntimeError('DDQ produced nonfinite boxes/scores')
        height, width = image.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
        valid = (boxes[:, 2:] - boxes[:, :2] > 1e-5).all(axis=1)
        return boxes[valid], scores[valid]
