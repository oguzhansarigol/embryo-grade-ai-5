"""DeepEmbryo project configuration: paths, hyperparameters, constants."""
from pathlib import Path

# --- Paths ---
# Project root is always two levels above this file (works in Colab after the
# zip is extracted to /content/code-base/, and locally from the repo root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "EMBRIO GRADE DATASET"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"
GRADCAM_DIR = FIGURE_DIR / "gradcam_samples"

for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR, REPORT_DIR, LOG_DIR, GRADCAM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Dataset ---
CLASSES = ["3AA", "3CC", "4AA", "Cleavage"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# --- Image preprocessing ---
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Training ---
SEED = 42
K_FOLDS = 5
BATCH_SIZE = 16
NUM_WORKERS = 2

# Two-stage training schedule
WARMUP_EPOCHS = 5      # head-only warmup
FINETUNE_EPOCHS = 45   # full network fine-tune
TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS

LR_HEAD_WARMUP = 1e-3
LR_HEAD_FINETUNE = 1e-4
LR_BACKBONE_FINETUNE = 1e-5
WEIGHT_DECAY = 0.05

LABEL_SMOOTHING = 0.1
# MixUp + CutMix were causing 4AA ↔ 3CC confusion on this small fine-grained
# dataset (the model literally learns the 50/50 mixture target). Disabled.
MIXUP_ALPHA = 0.0
CUTMIX_ALPHA = 0.0
DROPOUT = 0.3
DROP_PATH = 0.1

EARLY_STOP_PATIENCE = 10

# --- Inference ---
CONFIDENCE_THRESHOLD = 0.70  # Below this -> manual review warning (doc §5.2)

# --- Model ---
MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
