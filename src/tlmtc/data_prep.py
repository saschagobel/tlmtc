"""Internal helpers for data preparation and dataset persistence.

Defines helpers used by the data pipeline for preprocessing, splitting, and saving intermediate datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def df_preprocess(
    df_path: str | Path,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Import, preprocess and extract column labels from raw train/test data.

    Args:
        df_path: Path to the raw CSV data file, with required columns "text", "label_*".

    Returns:
        df: Preprocessed DataFrame.
        label_cols: Label column names.
        X: Text samples as a NumPy array.
        y: Label matrix as a NumPy array.
    """
    df = pd.read_csv(df_path).dropna()
    if df.empty:
        raise ValueError("After dropping missing values, no valid samples remain.")

    if "text" not in df.columns:
        raise ValueError("Input data must contain a 'text' column.")

    label_cols = [col for col in df.columns if col.startswith("label_")]
    if len(label_cols) < 2:
        raise ValueError("Expected at least two 'label_*' columns for multi-label classification.")
    for col in label_cols:
        if not df[col].map(lambda x: isinstance(x, (int, float))).all():
            raise TypeError(f"Column '{col}' must contain only integer values.")
    allowed_values = {0, 1, 0.0, 1.0}
    if not set(df[label_cols].stack().unique()).issubset(allowed_values):
        raise ValueError("Label columns must contain only binary values {0, 1, 0.0, 1.0}.")

    X = df["text"].values
    y = df[label_cols].values
    return df, label_cols, X, y


def df_split(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split preprocessed data into stratified train and test sets.

    Args:
        df: Preprocessed data.
        X: Texts.
        y: Label matrix.
        test_size: Proportion of dataset to be used for testing.
        random_seed: Random seed.

    Returns:
        train_data: Train set.
        test_data: Test set.
    """
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    for train_idx, test_idx in msss.split(X, y):
        train_data = df.iloc[train_idx].reset_index(drop=True)
        test_data = df.iloc[test_idx].reset_index(drop=True)
    return train_data, test_data


def df_save(
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save a DataFrame to disk as parquet file.

    Args:
        df: DataFrame to save.
        path: Path where the DataFrame will be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
