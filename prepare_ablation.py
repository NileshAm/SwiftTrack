"""Create public assets and train missing checkpoints for a code-only checkout.

Requires MOT17 train videos from config and the existing Python environment.
No teammate outputs needed. Downloads are checksum-verified; training resumes.
python prepare_ablation.py --encoder all --epochs 4
"""
import argparse
import importlib
import json
from pathlib import Path
import platform
from urllib.request import Request, urlopen
from zipfile import ZipFile

from tqdm.auto import tqdm
from encoder_experiment import configure
from Utils import config
from Utils.detections import file_digest

TRACKTOR_URL = 'https://webshare.cvg.cit.tum.de/u/brasoand/mot_neural_solver/prepr_det_files.zip'
REID_URL = ('https://huggingface.co/kaiyangzhou/osnet/resolve/'
            'a5c5cc037c24235cda3b21085b93ad77c9616224/'
            'osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
REID_SHA256 = '48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727'


def download_verified(url, destination, expected_hash):
    """Reuse verified assets; atomically publish downloads; reject changed files."""
    destination = Path(destination)
    if destination.exists():
        if file_digest(destination) != expected_hash:
            raise ValueError(f'Checksum mismatch: {destination}. Move this file aside and retry.')
        print(f'Reusing verified asset: {destination}', flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + '.download')
    request = Request(url, headers={'User-Agent': 'SwiftTrack-ablation-setup/1.0'})
    with urlopen(request, timeout=120) as response, pending.open('wb') as stream:
        total = int(response.headers.get('Content-Length', 0)) or None
        with tqdm(total=total, unit='B', unit_scale=True, desc=destination.name,
                  disable=not config.SHOW_PROGRESS) as bar:
            while block := response.read(1024 * 1024):
                stream.write(block)
                bar.update(len(block))
    if file_digest(pending) != expected_hash:
        raise ValueError(f'Download checksum mismatch: {pending}; destination was not replaced')
    pending.replace(destination)
    return destination


def preflight(encoders):
    """Fail before downloads/training when dependencies or labelled videos are missing."""
    modules = ['torch', 'torchvision', 'numpy', 'scipy', 'PIL', 'tqdm', 'matplotlib', 'trackeval']
    if 'detr' in encoders:
        modules += ['transformers', 'timm']
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as error:
            missing.append(f'{name}: {error}')
    if missing:
        raise RuntimeError('Environment is incomplete. See Documentation/Implementation/29-ablation-from-scratch.md\n'
                           + '\n'.join(missing))
    import torch
    if config.DEVICE == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; activate the GPU-enabled environment or set DEVICE="cpu"')
    from Utils.splits import validate_video_splits
    validate_video_splits()
    from Utils.data import _sequence_info
    for name in dict.fromkeys(config.TRAIN_SEQUENCES + config.VAL_SEQUENCES):
        folder = config.DATASET_ROOT / config.DATASET_SPLIT / name
        info = _sequence_info(folder)
        for relative in [config.GT_FILE, info.get('imDir')]:
            if not (folder / relative).exists():
                raise FileNotFoundError(f'Missing MOT17 input: {folder / relative}')
        images = folder / info.get('imDir')
        if len(list(images.glob('*' + info.get('imExt')))) != info.getint('seqLength'):
            raise ValueError(f'Image count differs from seqinfo.ini: {folder}')


def prepare(encoders, epochs=4, run_tag=None):
    """Prepare selected assets, smoke-check encoders and train a matched epoch budget."""
    run_tag = run_tag or f'scratch_{epochs}ep'
    # Validate the tag before downloads or output creation.
    configure(encoders[0], epochs, run_tag=run_tag)
    config.TEST_SEQUENCES = []
    config.DETECTION_BACKEND = 'tracktor'
    preflight(encoders)
    download_verified(TRACKTOR_URL, config.TRACKTOR_ARCHIVE, config.TRACKTOR_ARCHIVE_SHA256)
    if 'reid' in encoders:
        download_verified(REID_URL, config.REID_WEIGHTS_PATH, REID_SHA256)
    from import_tracktor import import_sequence
    with ZipFile(config.TRACKTOR_ARCHIVE) as archive:
        for name in tqdm(list(dict.fromkeys(config.TRAIN_SEQUENCES + config.VAL_SEQUENCES)),
                         desc='Import validation/training detections', disable=not config.SHOW_PROGRESS):
            import_sequence(archive, config.DATASET_ROOT / config.DATASET_SPLIT / name)
    import torch
    from encoder_experiment import smoke
    import train
    for encoder in encoders:
        configure(encoder, epochs, run_tag=run_tag)
        config.RESUME_TRAINING = True
        if config.CHECKPOINT_PATH.is_file():
            last_path = config.OUTPUT_DIR / 'last.pt'
            latest = torch.load(last_path if last_path.is_file() else config.CHECKPOINT_PATH,
                                map_location='cpu', weights_only=True)
            if latest['epoch'] > epochs:
                raise ValueError(f'{config.OUTPUT_DIR} already exceeds {epochs} epochs; choose another --run-tag')
            del latest
        print(f'\nPreparing {encoder}: {epochs} total epochs, {config.OUTPUT_DIR}', flush=True)
        # Downloads DenseNet/DETR pretrained weights automatically when absent.
        smoke()
        train.main()
        settings = {k: v for k, v in vars(config).items() if k.isupper()}
        (config.OUTPUT_DIR / 'setup_manifest.json').write_text(json.dumps({
            'encoder': encoder, 'target_epochs': epochs, 'run_tag': run_tag,
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'checkpoint_sha256': file_digest(config.CHECKPOINT_PATH),
            'settings': settings,
            'note': 'Locally trained; not guaranteed bitwise identical to another machine.',
        }, indent=2, default=str) + '\n')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return run_tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=['all', 'densenet', 'detr', 'reid'], default='all')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--run-tag')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    encoders = ['densenet', 'detr', 'reid'] if args.encoder == 'all' else [args.encoder]
    tag = prepare(encoders, args.epochs, args.run_tag)
    print(f'\nReady. Run: python gcn_ablation.py --encoder {args.encoder} --run-tag {tag}')


if __name__ == '__main__':
    main()
