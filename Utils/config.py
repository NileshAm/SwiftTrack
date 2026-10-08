"""Settings shared by training, tracking, and evaluation.

From a script in the project root, use: from Utils import config
Residual appearance run; see Documentation/Implementation/22-appearance-residual.md.
Train on TRAIN_SEQUENCES; validate/score full videos in VAL_SEQUENCES.
"""

from pathlib import Path
from torch.cuda import is_available


# PATHS: based on this file's location, so they do not depend on your terminal.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MOT17_ROOT = DATA_ROOT / "MOT17"
MOT20_ROOT = DATA_ROOT / "MOT20"  # Not present locally yet; unused initially.
DATASET_ROOT = MOT17_ROOT
DATASET_SPLIT = "train"  # MOT's labelled folder containing both video lists below.

RUN_NAME = "mot17_appearance_residual_v1"  # Fresh residual model; preserve the tuned v4 checkpoint/results.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / RUN_NAME
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"  # Best validation weights saved by train.py.
TRACKEVAL_ROOT = None  # None uses installed trackeval; alternatively point to its source checkout.


# DATA: edit these lists. Every selected video is used from first to last frame.
# Example 5/2 split, not a prescribed paper split. Detector variants of the same
# video (e.g. MOT17-02-SDP and MOT17-02-FRCNN) must stay in the SAME set.
TRAIN_SEQUENCES = [
    "MOT17-04-SDP", "MOT17-05-SDP", "MOT17-10-SDP", "MOT17-11-SDP", "MOT17-13-SDP"
]
VAL_SEQUENCES = ["MOT17-02-SDP", "MOT17-09-SDP"]

DETECTION_BACKEND = "tracktor"  # Authors-linked preprocessed public boxes; upstream IDs discarded.
TRACKTOR_CACHE_DIR = PROJECT_ROOT / "detections" / "tracktor_public"
TRACKTOR_ARCHIVE = TRACKTOR_CACHE_DIR / "prepr_det_files.zip"
TRACKTOR_ARCHIVE_SHA256 = "d7fe442e544bea2a102188d947b598a2ce63810e2158c709341c6e3dac4d1c7c"
DETECTION_FILE = "det/det.txt"  # Only for "supplied"; relative to each sequence folder.
DETECTION_CACHE_DIR = PROJECT_ROOT / "detections" / "rtdetrv2_s_coco_640_score010"
DETECTOR_MODEL_NAME = "PekingU/rtdetr_v2_r18vd"  # COCO pretrained RT-DETRv2-S, ResNet18.
DETECTOR_REVISION = "5650961749fa93567c0d46fc7f43ea4f9e914107"
DETECTOR_INPUT_MODE = "square"  # "original": keep every pixel, pad bottom/right to a multiple of 32.
DETECTOR_IMAGE_SIZE = 640  # Only for "square" mode; try 960/1280 in separate caches.
DETECTOR_MIN_SCORE = 0.10  # Cache floor; tune on validation, not test videos.
GT_FILE = "gt/gt.txt"  # Training labels/evaluation only, never tracking input.
DETECTION_SOURCE = "Authors-linked Tracktor-preprocessed MOT17 public detections (SDP variant)"
BOX_FORMAT = "xywh"  # Input: left, top, width, height.
BOX_ORIGIN = 1  # MOT coordinates -> subtract 1 from x/y for internal zero-origin boxes.
DETECTION_MIN_SCORE = None  # Optional extra filter when READING boxes; cannot recover boxes below cache floor.
GT_MATCH_IOU = 0.5  # Starting overlap threshold for detection-to-GT labels.


# COMPUTE AND FEATURES: full-image DETR batches on the 8 GB GPU.
DEVICE = "cuda"  # Use "cpu" only for small debugging runs if CUDA is unavailable.
SHOW_PROGRESS = True  # Overall ETA + detailed tqdm bars; False hides both levels.
SAVE_PLOTS = True  # Refresh training PNGs each epoch; save charts after evaluation.
SEED = 42
CROP_BATCH_SIZE = 16  # Only used if explicitly switching back to DenseNet121.
CROP_SIZE = (256, 128)  # DenseNet-only (height, width).
ENCODER_NAME = "detr_resnet50"
USE_PRETRAINED_WEIGHTS = True  # Official DETR weights are already cached in dragon-track.
FREEZE_ENCODER = True  # Restore original pretrained DETR features; train only graph/association weights.
DETR_GRADIENT_CHECKPOINTING = False  # Faster with spare VRAM; True trades compute for less memory.
FEATURE_DIM = 256  # DETR transformer encoder features, pooled inside each supplied box.
DETR_MODEL_NAME = "facebook/detr-resnet-50"
DETR_REVISION = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
DETR_SHORT_SIDE = 800
DETR_MAX_SIZE = 1333


# PAPER: specified architecture/optimizer; unspecified constants are marked below.
GRAPH_HIDDEN_DIMS = (512, 128)
# Both graph layers now preserve own features with a learned residual projection.
ASSOCIATION_DISTANCE_PX = 25  # Paper Sec. 3.7; same original-pixel gate in train and tracking.
SINKHORN_ITERATIONS = 50  # Fixed-marginal normalization, then unmatched residual correction.
SINKHORN_SCORE_SCALE = 5.0  # Released score exponent multiplier; not specified by the paper.
SINKHORN_UNMATCHED_LOGIT = -1.0  # Initial learned unmatched preference; masses are never learned.

# TRAINING: actual full-image DETR batches, with chronological graph scoring.
EPOCHS = 30  # Total target: resume the epoch-20 residual model through epoch 40.
RESUME_TRAINING = True  # Continue this run's last completed epoch; empty run starts fresh.
LEARNING_RATE = 1e-4  # Paper Sec. 4.1, Adam.
TRAIN_BATCH_SIZE = 2  # Images encoded together; one optimizer step per batch. Paper uses 2.
VAL_BATCH_SIZE = 4  # Independent image batching, no gradients or optimizer updates.
LOSS_ALPHA = 0.5  # Eq. 10; paper does not report alpha, so expose the starting choice.
WBCE_POSITIVE_WEIGHT = 10.0  # Released WBCE weights, unspecified in paper.
WBCE_NEGATIVE_WEIGHT = 1.0
CONTRASTIVE_MARGIN = 1.0  # Euclidean distance of unit embeddings (same directions as cosine matching).
NEGATIVE_RATIO = 3  # Keep all positives and at most this many negatives per positive.
HARD_NEGATIVE_FRACTION = 0.5  # Remaining sampled negatives are random; validation uses all.
LOOKBACK_FRAMES = 30  # Maximum observation gap, shared by train/validation/tracking.
MAX_MISSED_FRAMES = LOOKBACK_FRAMES - 1  # 30-frame gap has 29 intervening missing frames.
GRAD_CLIP_NORM = 5.0
SAVE_EVERY_EPOCH = True  # Legacy setting; last.pt now always saves each epoch for safe resume.


# TRACKING: finalize score-based thresholds after defining the association score.
MATCH_THRESHOLD = 0.05  # Released tracker default; validate for this implementation.
NEW_TRACK_MIN_SCORE = 0.5  # Provisional confidence threshold; tune on validation only.
USE_MOTION_PREDICTION = True  # Shared by training, validation loss, and tracking.
USE_APPEARANCE_HISTORY = True  # Shared confidence-weighted memory of reliable features.
APPEARANCE_HISTORY_SIZE = 30  # Accepted observations, not elapsed frames.
APPEARANCE_HISTORY_MIN_SCORE = 0.5  # Lower-score matches keep IDs but do not change memory.
TRACK_CONFIRM_MIN_HITS = 1  # Released tracker: reject IDs observed fewer than 3 times.
TRACK_CONFIRM_WINDOW = 15  # Require those observations within 15 frames.
TRACKING_VARIANT = "paper_motion_history"  # Separate results using the SAME trained checkpoint.
SAVE_VIDEO = False  # Enable for the demo; leave off when measuring speed.
TRACK_SPLIT = "val"  # Whole VAL_SEQUENCES; there is no percentage-based eval remainder.
RUN_SPLIT_TRACKING = True  # False runs only TEST_SEQUENCES, without repeating validation tracking.
TRACK_OUTPUT_DIR = OUTPUT_DIR / "tracking" / TRACKING_VARIANT / TRACK_SPLIT
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation" / TRACKING_VARIANT / TRACK_SPLIT

# tracking.py runs full VAL_SEQUENCES when RUN_SPLIT_TRACKING=True, then these optional
# FULL test videos. No GT is read and test videos never enter train/evaluate.
TEST_SEQUENCES = ["MOT17-01-SDP", "MOT17-03-SDP", "MOT17-06-SDP", "MOT17-07-SDP"]  # Set [] to skip additional tracking/videos.
TEST_DATASET_ROOT = MOT17_ROOT
TEST_DATASET_SPLIT = "test"
TEST_OUTPUT_DIR = OUTPUT_DIR / "tracking" / TRACKING_VARIANT / "test"
SAVE_TEST_VIDEO = True

def test_gpu():
    if (is_available()):
        print("Cuda device detected.")
        return True
    else:
        print("Cuda device not detected.")
    return False
