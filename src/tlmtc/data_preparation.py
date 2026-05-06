"""Data-preparation operations for Hugging Face multi-label text classification."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import BatchEncoding, PreTrainedTokenizerBase

from tlmtc.data_contracts import LABEL_PREFIX, TEXT_COL, TEXT_PAIR_COL, InputMode, validate_multilabel_frame


def df_preprocess(
    df_path: Path,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, InputMode]:
    """Load, clean, and validate a multi-label CSV for stratified splitting.

    Args:
        df_path: Path to a CSV with a required text column, optional paired-text column,
            and binary label columns.

    Returns:
        Preprocessed DataFrame, label column names, text_values, label matrix, and inferred input mode.
    """
    df = pd.read_csv(df_path).dropna().reset_index(drop=True)
    df, label_cols, input_mode = validate_multilabel_frame(df)

    text_values = df[TEXT_COL].values
    label_matrix = df[label_cols].values
    return df, label_cols, text_values, label_matrix, input_mode


def df_split(
    df: pd.DataFrame,
    text_values: np.ndarray,
    label_matrix: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a multilabel-stratified two-way data split.

    Args:
        df: Preprocessed DataFrame to split.
        text_values: Text values used as splitter inputs.
        label_matrix: Multi-label target matrix used for stratification.
        test_size: Fraction of rows assigned to the second split.
        random_seed: Random seed for reproducible splitting.

    Returns:
        Training partition and held-out partition.

    Raises:
        ValueError: If no valid split preserves positive examples for every label.
    """
    max_split_attempts = 3
    label_cols = [col for col in df.columns if col.startswith(LABEL_PREFIX)]
    missing_positive_labels: list[str] = []

    for attempt in range(max_split_attempts):
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_seed + attempt,
        )

        train_idx, test_idx = next(splitter.split(text_values, label_matrix))
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
    """Save a DataFrame as a parquet artifact.

    Args:
        df: DataFrame to persist.
        path: Destination parquet path.
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
        tokenizer: Tokenizer for the selected checkpoint.
        input_mode: Text input mode used to choose single-text or paired-text tokenization.
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
