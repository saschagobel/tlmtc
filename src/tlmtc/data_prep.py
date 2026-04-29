"""Internal helpers for data preparation and dataset persistence.

Defines helpers used by the data pipeline for preprocessing, splitting, and saving intermediate datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from tlmtc.data_contracts import validate_multilabel_frame


def df_preprocess(
    df_path: str | Path,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Import, validate, preprocess and extract labels from train/test data.

    Args:
        df_path: Path to the CSV data file, with required columns "text", "label_*".

    Returns:
        df: Preprocessed DataFrame.
        label_cols: Label column names.
        X: Text samples as a NumPy array.
        y: Label matrix as a NumPy array.
    """
    df = pd.read_csv(df_path).dropna().reset_index(drop=True)
    df, label_cols = validate_multilabel_frame(df)

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
