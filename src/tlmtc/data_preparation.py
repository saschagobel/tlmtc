"""Data-preparation operations for Hugging Face multi-label text classification training and prediction."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Value
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

from tlmtc.data_contracts import (
    LABEL_PREFIX,
    SPLIT_GROUP_COL,
    TEXT_COL,
    TEXT_PAIR_COL,
    InputMode,
    validate_multilabel_frame,
    validate_prediction_frame,
)
from tlmtc.runtime_output import emit_progress


def df_preprocess(
    data: Path | pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, InputMode]:
    """Load, clean, and validate raw multi-label training data for stratified splitting.

    Args:
        data: Path to a CSV, or a DataFrame with a required text column,
            optional paired-text column, and binary label columns.

    Returns:
        Preprocessed DataFrame, label column names, text_values, label matrix, and inferred input mode.
    """
    if isinstance(data, Path):
        df = pd.read_csv(data)
    else:
        df = data

    df = df.dropna().reset_index(drop=True)
    df, label_cols, input_mode = validate_multilabel_frame(df)

    text_values = df[TEXT_COL].values
    label_matrix = df[label_cols].values
    return df, label_cols, text_values, label_matrix, input_mode


def read_prediction_data(
    data: Path | pd.DataFrame,
    expected_input_mode: InputMode,
) -> pd.DataFrame:
    """Load and validate unlabeled prediction data.

    Args:
        data: Path to an unlabeled CSV, or a DataFrame with the text columns required by the trained model.
        expected_input_mode: Input mode persisted by the training run.

    Returns:
        Validated prediction DataFrame with original columns preserved.
    """
    if isinstance(data, Path):
        df = pd.read_csv(data)
    else:
        df = data

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
        text_values: Text values used as splitter inputs for row-level splitting.
        label_matrix: Multi-label target matrix used for row-level splitting.
        test_size: Fraction of rows or groups assigned to the second split.
        random_seed: Random seed for reproducible splitting.

    Returns:
        Training partition and held-out partition.

    Raises:
        ValueError: If no valid split preserves positive examples for every label.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    max_split_attempts = 3
    grouped_split = SPLIT_GROUP_COL in df.columns
    label_cols = [col for col in df.columns if col.startswith(LABEL_PREFIX)]
    missing_positive_labels: list[str] = []

    if grouped_split:
        group_labels = df.groupby(SPLIT_GROUP_COL, sort=False)[label_cols].max()
        split_values = group_labels.index.to_numpy()
        split_labels = group_labels.to_numpy()
    else:
        split_values = text_values
        split_labels = label_matrix

    for attempt in range(max_split_attempts):
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_seed + attempt,
        )

        train_idx, test_idx = next(splitter.split(split_values, split_labels))

        if grouped_split:
            train_groups = group_labels.index[train_idx]
            test_groups = group_labels.index[test_idx]

            train_data = df[df[SPLIT_GROUP_COL].isin(train_groups)].reset_index(drop=True)
            test_data = df[df[SPLIT_GROUP_COL].isin(test_groups)].reset_index(drop=True)
        else:
            train_data = df.iloc[train_idx].reset_index(drop=True)
            test_data = df.iloc[test_idx].reset_index(drop=True)

        missing_positive_labels = [col for col in label_cols if train_data[col].sum() == 0 or test_data[col].sum() == 0]

        if not missing_positive_labels:
            if grouped_split:
                _emit_grouped_split_drift(
                    train_data=train_data,
                    test_data=test_data,
                    test_size=test_size,
                )
            return train_data, test_data

    split_kind = "grouped multilabel stratified" if grouped_split else "multilabel stratified"
    grouped_hint = (
        " Grouped splitting is constrained by the number and label composition of split groups."
        if grouped_split
        else ""
    )

    raise ValueError(
        f"Could not create a valid {split_kind} split after "
        f"{max_split_attempts} attempts. The following labels have no positive examples "
        f"in at least one split partition: {missing_positive_labels}."
        f"{grouped_hint} "
        "Increase the smaller split size, provide more positive examples for rare labels, "
        "or remove/merge labels with insufficient support."
    )


def _emit_grouped_split_drift(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    test_size: float,
) -> None:
    """Emit a progress message when grouped splitting drifts from the requested row fraction."""
    achieved_test_size = len(test_data) / (len(train_data) + len(test_data))

    if abs(achieved_test_size - test_size) <= 0.05:
        return

    emit_progress(
        "Grouped splitting produced a held-out row fraction of "
        f"{achieved_test_size:.3f}, requested {test_size:.3f}. "
        f"Exact row-level split sizes are not always possible because rows sharing "
        f"the same '{SPLIT_GROUP_COL}' value must stay in the same split."
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
    tokenizer_dir: Path,
    input_mode: InputMode,
    sequence_length: int,
    trust_remote_code: bool,
    inference_backend: Literal["torch", "onnx"] = "torch",
) -> Dataset:
    """Tokenize prediction inputs.

    Args:
        dataset: Unlabeled Hugging Face prediction dataset.
        tokenizer_dir: Directory containing the persisted tokenizer artifacts.
        input_mode: Input mode persisted by the training run.
        sequence_length: Maximum tokenized sequence length.
        trust_remote_code: Whether Hugging Face tokenizer loading may execute custom remote code.
        inference_backend: Runtime backend used for prediction.

    Returns:
        Tokenized prediction dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

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
    tokenized_dataset.set_format("numpy" if inference_backend == "onnx" else "torch")
    return tokenized_dataset
