"""
Transfer Learning for Multi-Label Text Classification.

Default configuration
"""

from typing import Final

from tlmtc.types import BestModelMetric, BestThresholdMetric, LoraBias, OptunaSpace, Threshold

# TARGET
TARGET_NAME = "Target"

# MODEL CHECKPOINTS
PROXY_CHECKPOINT = "microsoft/deberta-v3-xsmall"
CHECKPOINT = "microsoft/deberta-v3-base"

# SPLITTING
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 2469

# WORKFLOW
HYPERPARAMETER_TUNING = True
THRESHOLD_OPTIMIZATION = True
THRESHOLD_TYPE: Final[Threshold] = "label"
TRANSFER_LEARNING = True
SCALE_LEARNING_RATE = False
WRAP_PEFT = True

# HYPERPARAMETERS
SEQUENCE_LENGTH = 128
BEST_MODEL_METRIC: Final[BestModelMetric] = "roc_auc_macro"
BATCH_SIZE = 16
TRAIN_EPOCHS = 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"
BEST_THRESHOLD_METRIC: Final[BestThresholdMetric] = "f1_macro"
TUNING_TRIALS = 10

# HYPERPARAMETER SEARCH SPACE
OPTUNA_SPACE_PEFT: Final[OptunaSpace] = {
    "lr_low": 3e-5,
    "lr_high": 5e-4,
    "batch_sizes": [8, 16, 32, 64],
    "wd_low": 0.0,
    "wd_high": 0.05,
    "schedulers": ["linear", "cosine"],
    "epoch_low": 5,
    "epoch_high": 20,
}
OPTUNA_SPACE_BASE: Final[OptunaSpace] = {
    "lr_low": 1e-5,
    "lr_high": 3e-4,
    "batch_sizes": [8, 16, 32],
    "wd_low": 0.0,
    "wd_high": 0.3,
    "schedulers": ["linear", "cosine", "polynomial"],
    "epoch_low": 5,
    "epoch_high": 30,
}

# PEFT
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_BIAS: Final[LoraBias] = "none"

# HARDWARE
USE_CPU = False
