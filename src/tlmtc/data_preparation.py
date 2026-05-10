"""Data-preparation operations for Hugging Face multi-label text classification training and prediction."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Value
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

from tlmtc.data_contracts import (
    LABEL_PREFIX,
    TEXT_COL,
    TEXT_PAIR_COL,
    InputMode,
    validate_multilabel_frame,
    validate_prediction_frame,
)


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


def read_prediction_csv(
    df_path: Path,
    expected_input_mode: InputMode,
) -> pd.DataFrame:
    """Load and validate an unlabeled prediction CSV.

    Args:
        df_path: Path to an unlabeled CSV with the text columns required by the trained model.
        expected_input_mode: Input mode persisted by the training run.

    Returns:
        Validated prediction DataFrame with original columns preserved.
    """
    df = pd.read_csv(df_path)
    return validate_prediction_frame(df=df, expected_input_mode=expected_input_mode)


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


def create_prediction_dataset(
    df: pd.DataFrame,
    input_mode: InputMode,
) -> Dataset:
    """Create an unlabeled Hugging Face dataset for prediction.

    Args:
        df: Validated prediction DataFrame.
        input_mode: Input mode persisted by the training run.

    Returns:
        Hugging Face dataset containing prediction text inputs.
    """
    feature_spec = {
        TEXT_COL: Value(dtype="string"),
    }
    input_cols = [TEXT_COL]

    if input_mode is InputMode.PAIRED_TEXT:
        feature_spec[TEXT_PAIR_COL] = Value(dtype="string")
        input_cols.append(TEXT_PAIR_COL)

    return Dataset.from_pandas(
        df[input_cols],
        features=Features(feature_spec),
        preserve_index=False,
    )


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


def tokenize_prediction_dataset(
    dataset: Dataset,
    checkpoint: str,
    input_mode: InputMode,
    sequence_length: int,
) -> Dataset:
    """Tokenize prediction inputs.

    Args:
        dataset: Unlabeled Hugging Face prediction dataset.
        checkpoint: Checkpoint used to load the tokenizer.
        input_mode: Input mode persisted by the training run.
        sequence_length: Maximum tokenized sequence length.

    Returns:
        Tokenized prediction dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokenized_dataset = dataset.map(
        lambda batch: tokenize_batch(
            batch=batch,
            tokenizer=tokenizer,
            input_mode=input_mode,
            sequence_length=sequence_length,
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset
