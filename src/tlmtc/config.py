"""
Transfer Learning for Multi-Label Text Classification.

Default configuration
"""

from pathlib import Path

# PATHS
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH.parent / "data"
RAW_DATA_PATH = DATA_PATH / "raw.csv"
RAW_TEST_DATA_PATH = DATA_PATH / "raw_test.csv"
TRAIN_DATA_PATH = DATA_PATH / "train.parquet"
VAL_DATA_PATH = DATA_PATH / "val.parquet"
TEST_DATA_PATH = DATA_PATH / "test.parquet"
OUTPUT_DATA_PATH = BASE_PATH.parent / "outputs"
OUTPUT_LOGGING_PATH = OUTPUT_DATA_PATH / "logging"

# MODEL CHECKPOINTS
CHECKPOINT = "microsoft/deberta-v3-base"

# SPLITTING
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 2469

# WORKFLOW
HYPERPARAMETER_TUNING = True

# HYPERPARAMETERS
SEQUENCE_LENGTH = 128
BEST_MODEL_METRIC = "roc_auc_macro"
BATCH_SIZE = 16
TRAIN_EPOCHS = 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"

# HARDWARE
USE_CPU = False
