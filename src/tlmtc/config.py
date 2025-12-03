"""
Transfer Learning for Multi-Label Text Classification.

Default configuration
"""

from pathlib import Path

# PATHS
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH.parent / "data"
RAW_DATA_PATH = DATA_PATH / "raw.csv"
TRAIN_DATA_PATH = DATA_PATH / "train.parquet"
VAL_DATA_PATH = DATA_PATH / "val.parquet"
TEST_DATA_PATH = DATA_PATH / "test.parquet"
