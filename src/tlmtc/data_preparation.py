"""Internal helpers for data preparation and dataset persistence.

Defines helpers used by the data pipeline for preprocessing, splitting, and saving intermediate datasets.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import BatchEncoding, PreTrainedTokenizerBase

from tlmtc.data_contracts import TEXT_COL, TEXT_PAIR_COL, InputMode, validate_multilabel_frame


def df_preprocess(
    df_path: Path,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, InputMode]:
    """Import, validate, preprocess and extract labels from train/test data.

    Args:
        df_path: Path to the CSV data file, with required text and label columns,
            and an optional paired-text column.

    Returns:
        df: Preprocessed DataFrame.
        label_cols: Label column names.
        X: Text samples as a NumPy array.
        y: Label matrix as a NumPy array.
        input_mode: Input mode inferred from the validated dataframe columns.
    """
    df = pd.read_csv(df_path).dropna().reset_index(drop=True)
    df, label_cols, input_mode = validate_multilabel_frame(df)

    X = df[TEXT_COL].values
    y = df[label_cols].values
    return df, label_cols, X, y, input_mode


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
    max_split_attempts = 3
    label_cols = [col for col in df.columns if col.startswith("label_")]
    missing_positive_labels: list[str] = []

    for attempt in range(max_split_attempts):
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_seed + attempt,
        )

        train_idx, test_idx = next(splitter.split(X, y))
        train_data = df.iloc[train_idx].reset_index(drop=True)
        test_data = df.iloc[test_idx].reset_index(drop=True)

        missing_positive_labels = [col for col in label_cols if train_data[col].sum() == 0 or test_data[col].sum() == 0]

        if not missing_positive_labels:
            return train_data, test_data

    raise ValueError(
        "Could not create a valid multilabel stratified split after "
        f"{max_split_attempts} attempts. The following labels have no positive examples "
        f"in at least one split partition: {missing_positive_labels}. "
        "Increase the smaller split size, provide more positive examples for rare labels, "
        "or remove/merge labels with insufficient support."
    )


def df_save(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Save a DataFrame to disk as parquet file.

    Args:
        df: DataFrame to save.
        path: Path where the DataFrame will be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def tokenize_batch(
    batch: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    input_mode: InputMode,
    sequence_length: int,
) -> BatchEncoding:
    """Tokenize a single-text or paired-text batch for sequence classification.

    Args:
        batch: Batched Hugging Face Dataset row dictionary.
        tokenizer: Hugging Face tokenizer for the selected checkpoint.
        input_mode: Validated input mode inferred from the data contract.
        sequence_length: Maximum tokenized sequence length.

    Returns:
        Tokenized model inputs.
    """
    if input_mode is InputMode.PAIRED_TEXT:
        return tokenizer(
            batch[TEXT_COL],
            batch[TEXT_PAIR_COL],
            truncation="longest_first",
            padding="max_length",
            max_length=sequence_length,
        )

    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        padding="max_length",
        max_length=sequence_length,
    )
