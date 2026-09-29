"""Settings shared by training, tracking, and evaluation.

From a script in the project root, use: from Utils import config
DETR appearance + graph attention run; see Documentation/Implementation/11-paper-components.md.
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

RUN_NAME = "mot17_SDP_video_split"  # Association training with supplied SDP boxes.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / RUN_NAME
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"  # Best validation weights saved by train.py.
TRACKEVAL_ROOT = None  # None uses installed trackeval; alternatively point to its source checkout.


# DATA: edit these lists. Every selected video is used from first to last frame.
# Example 5/2 split, not a prescribed paper split. Detector variants of the same
# video (e.g. MOT17-02-SDP and MOT17-02-FRCNN) must stay in the SAME set.
TRAIN_SEQUENCES = [
    "MOT17-04-SDP", "MOT17-05-SDP"
]
VAL_SEQUENCES = ["MOT17-02-SDP", "MOT17-09-SDP"]

DETECTION_FILE = "det/det.txt"  # Relative to each sequence folder.
GT_FILE = "gt/gt.txt"  # Training labels/evaluation only, never tracking input.
DETECTION_SOURCE = "MOT17 supplied SDP detections"
BOX_FORMAT = "xywh"  # Input: left, top, width, height.
BOX_ORIGIN = 1  # MOT coordinates -> subtract 1 from x/y for internal zero-origin boxes.
DETECTION_MIN_SCORE = None  # Keep supplied detections initially.
GT_MATCH_IOU = 0.5  # Starting overlap threshold for detection-to-GT labels.


# COMPUTE AND FEATURES: one full-frame DETR pass at a time on the 8 GB GPU.
DEVICE = "cuda"  # Use "cpu" only for small debugging runs if CUDA is unavailable.
SHOW_PROGRESS = True  # Overall ETA + detailed tqdm bars; False hides both levels.
SEED = 42
CROP_BATCH_SIZE = 16  # Only used if explicitly switching back to DenseNet121.
CROP_SIZE = (256, 128)  # DenseNet-only (height, width).
ENCODER_NAME = "detr_resnet50"
USE_PRETRAINED_WEIGHTS = True  # Official DETR weights are already cached in dragon-track.
FREEZE_ENCODER = True
FEATURE_DIM = 256  # DETR transformer encoder features, pooled inside each supplied box.
DETR_MODEL_NAME = "facebook/detr-resnet-50"
DETR_REVISION = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
DETR_SHORT_SIDE = 800
DETR_MAX_SIZE = 1333


# MODEL: paper-inspired attention, relationships, and trainable Sinkhorn.
GRAPH_HIDDEN_DIM = 128
GRAPH_LAYERS = 2
ATTENTION_HEADS = 4  # GRAPH_HIDDEN_DIM must be divisible by this value.
SINKHORN_ITERATIONS = 100  # Alternating scaled row/column normalization.


# TRAINING: FP32, one frame-pair graph per step, fixed learning rate, no scheduler.
EPOCHS = 20  # Training budget; model.pt keeps the lowest validation loss, not necessarily epoch 20.
LEARNING_RATE = 3e-5  # Conservative new candidate: at 1e-4 validation worsened after epoch 3.
BCE_WEIGHT = 1.0
CONTRASTIVE_WEIGHT = 1.0
CONTRASTIVE_MARGIN = 1.0  # Distance margin for L2-normalized graph embeddings.
ASSIGNMENT_WEIGHT = 1.0  # Direct supervision through learnable Sinkhorn, including unmatched IDs.
NEGATIVE_RATIO = 3  # Keep all positives and at most this many negatives per positive.
HARD_NEGATIVE_FRACTION = 0.5  # Remaining sampled negatives are random; validation uses all.
TRAIN_FRAME_GAPS = (1, 2, 4)  # Sample one gap per anchor each epoch, within the training split.
GRAD_CLIP_NORM = 5.0
SAVE_EVERY_EPOCH = True


# TRACKING: finalize score-based thresholds after defining the association score.
MATCH_THRESHOLD = 0.1  # Keep the comparison baseline; tune on validation after retraining.
NEW_TRACK_MIN_SCORE = 0.5  # Provisional confidence threshold; tune on validation only.
MAX_MISSED_FRAMES = 30  # Keep a lost track eligible for matching for this many frames.
SAVE_VIDEO = False  # Enable for the demo; leave off when measuring speed.
TRACK_SPLIT = "val"  # Whole VAL_SEQUENCES; there is no percentage-based eval remainder.
RUN_SPLIT_TRACKING = True  # False runs only TEST_SEQUENCES, without repeating validation tracking.
TRACK_OUTPUT_DIR = OUTPUT_DIR / "tracking" / TRACK_SPLIT
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation" / TRACK_SPLIT

# tracking.py runs full VAL_SEQUENCES when RUN_SPLIT_TRACKING=True, then these optional
# FULL test videos. No GT is read and test videos never enter train/evaluate.
TEST_SEQUENCES = ["MOT17-01-SDP", "MOT17-03-SDP"]  # Set [] to skip additional tracking/videos.
TEST_DATASET_ROOT = MOT17_ROOT
TEST_DATASET_SPLIT = "test"
TEST_OUTPUT_DIR = OUTPUT_DIR / "tracking" / "test"
SAVE_TEST_VIDEO = True

def test_gpu():
    if (is_available()):
        print("Cuda device detected.")
        return True
    else:
        print("Cuda device not detected.")
    return False
