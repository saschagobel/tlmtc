"""Tests for data preparation utility functions."""

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import torch
from datasets import Dataset, DatasetDict

from tlmtc.data_pipeline import DataPipeline


@pytest.fixture
def sample_raw_csv(tmp_path: Path):
    """Create a small synthetic multi-label dataset and write it to raw.csv."""
    df = pd.DataFrame(
        {
            "text": ["hello world ooqz", "foo bar", "hello", "bar world qooz"],
            "label_a": [1, 0, 1, 0],
            "label_b": [0, 1, 1, 0],
        }
    )
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_raw_test_csv(tmp_path: Path):
    """Create a small synthetic multi-label test dataset and write it to raw_test.csv."""
    df = pd.DataFrame(
        {
            "text": ["ooqz world", "foo Hello", "hello bar"],
            "label_a": [1, 1, 0],
            "label_b": [1, 0, 0],
        }
    )
    csv_path = tmp_path / "raw_test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def pipeline_instance_factory(tmp_path: Path):
    """Create a factory fixture for creating DataPipeline instances with configurable inputs."""

    def _factory(
        raw_csv: Path,
        raw_test_csv: Path,
        hyperparameter_tuning: bool,
    ):
        """Instantiate a DataPipeline with temporary directories for outputs."""
        train_path = tmp_path / "train.parquet"
        val_path = tmp_path / "val.parquet"
        test_path = tmp_path / "test.parquet"

        return DataPipeline(
            raw_data_path=raw_csv,
            raw_test_data_path=raw_test_csv,
            train_data_path=train_path,
            val_data_path=val_path,
            test_data_path=test_path,
            hyperparameter_tuning=hyperparameter_tuning,
            validation_size=0.15,
            test_size=0.15,
            random_seed=42,
            checkpoint="tests/data/tiny_tokenizer",
            sequence_length=16,
        )

    return _factory


@pytest.mark.parametrize(
    "hyperparameter_tuning, use_raw_test",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_split_data_handles_all_split_configurations(
    tmp_path: Path,
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
    hyperparameter_tuning: bool,
    use_raw_test: bool,
):
    """Validate split_data behavior for all combinations of tuning mode and presence of raw test data."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv if use_raw_test else "",
        hyperparameter_tuning=hyperparameter_tuning,
    )

    dp.split_data()

    assert dp.train_data_path.exists()
    assert dp.test_data_path.exists()

    if hyperparameter_tuning:
        assert dp.val_data_path.exists()
    else:
        assert not dp.val_data_path.exists()

    train_df = pd.read_parquet(dp.train_data_path)
    test_df = pd.read_parquet(dp.test_data_path)
    val_df = pd.read_parquet(dp.val_data_path) if hyperparameter_tuning else None

    assert len(train_df) > 0
    assert len(test_df) > 0

    if hyperparameter_tuning and use_raw_test:
        assert len(test_df) == len(pd.read_csv(sample_raw_test_csv))
        assert len(train_df) + len(val_df) == len(pd.read_csv(sample_raw_csv))

    elif hyperparameter_tuning and not use_raw_test:
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(pd.read_csv(sample_raw_csv))

    elif not hyperparameter_tuning and use_raw_test:
        assert len(train_df) == len(pd.read_csv(sample_raw_csv))
        assert len(test_df) == len(pd.read_csv(sample_raw_test_csv))

    else:
        total = len(train_df) + len(test_df)
        assert total == len(pd.read_csv(sample_raw_csv))


def test_split_data_raises_when_raw_data_missing(tmp_path: Path, pipeline_instance_factory: Callable):
    """Ensure split_data raises FileNotFoundError when the raw_data_path is missing."""
    missing_raw = tmp_path / "does_not_exist.csv"

    dp = pipeline_instance_factory(
        raw_csv=missing_raw,
        raw_test_csv="",
        hyperparameter_tuning=True,
    )

    with pytest.raises(FileNotFoundError):
        dp.split_data()


def test_split_data_raises_when_label_columns_mismatch(
    tmp_path: Path,
    sample_raw_csv: Path,
    pipeline_instance_factory: Callable,
):
    """Ensure split_data raises a ValueError when train and test label columns differ."""
    df_test = pd.DataFrame(
        {
            "text": ["x", "y"],
            "label_x": [1, 0],
            "label_b": [0, 1],
        }
    )
    mismatched_test_path = tmp_path / "raw_test_mismatch.csv"
    df_test.to_csv(mismatched_test_path, index=False)

    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=mismatched_test_path,
        hyperparameter_tuning=True,
    )

    with pytest.raises(ValueError, match="Mismatch between train/test label columns"):
        dp.split_data()


@pytest.mark.parametrize("hyperparameter_tuning", [True, False])
def test_get_multi_hot_vectors_transforms_label_columns_across_splits(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
    hyperparameter_tuning: bool,
):
    """Verify that get_multi_hot_vectors converts label_* columns into multi-hot 'labels' lists across all splits."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=hyperparameter_tuning,
    )

    dp.split_data().get_multi_hot_vectors()

    expected_splits = ["train_data", "test_data"]
    if hyperparameter_tuning:
        expected_splits.insert(1, "val_data")

    for split_attr in expected_splits:
        df = getattr(dp, split_attr)

        assert list(df.columns) == ["text", "labels"]

        labels = df["labels"].iloc[0]
        assert isinstance(labels, list)
        assert len(labels) == 2
        assert set(labels).issubset({0, 1})


@pytest.mark.parametrize(
    "hyperparameter_tuning, expected_error",
    [
        (True, "Validation data not found"),
        (False, "Train/test data not found"),
    ],
)
def test_get_multi_hot_vectors_raises_if_called_before_split_data(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
    hyperparameter_tuning: bool,
    expected_error: str,
):
    """Ensure get_multi_hot_vectors raises an error when called before split_data, in both tuning modes."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=hyperparameter_tuning,
    )

    with pytest.raises(RuntimeError, match=expected_error):
        dp.get_multi_hot_vectors()


@pytest.mark.parametrize("hyperparameter_tuning", [True, False])
def test_create_hf_dataset_constructs_correct_splits_and_schema(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
    hyperparameter_tuning: bool,
):
    """Verify that create_hf_dataset builds the expected DatasetDict splits and schema under both tuning modes."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=hyperparameter_tuning,
    )

    dp.split_data().get_multi_hot_vectors()
    dp.create_hf_dataset()

    assert isinstance(dp.hf_dataset, DatasetDict)

    expected_keys = ["train", "test", "validation"] if hyperparameter_tuning else ["train", "test"]

    assert list(dp.hf_dataset.keys()) == expected_keys

    train_split = dp.hf_dataset["train"]
    assert isinstance(train_split, Dataset)

    assert "text" in train_split.features
    assert "labels" in train_split.features

    labels_feature = train_split.features["labels"]
    assert labels_feature.feature.dtype == "int64"


def test_create_hf_dataset_raises_when_validation_split_missing(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
):
    """Ensure create_hf_dataset raises an error when tuning is enabled but no validation split exists."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=True,
    )

    dp.split_data()
    dp.val_data = None

    with pytest.raises(RuntimeError, match="Validation data not found"):
        dp.create_hf_dataset()


def test_create_hf_dataset_raises_if_called_before_split_data(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
):
    """Ensure create_hf_dataset raises an error when invoked before split_data initializes the splits."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=False,
    )

    with pytest.raises(RuntimeError, match="Train/test data not found"):
        dp.create_hf_dataset()


def test_create_hf_dataset_raises_if_labels_column_missing(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
):
    """Ensure create_hf_dataset raises an error when called before labels are generated by get_multi_hot_vectors."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=True,
    )

    dp.split_data()

    with pytest.raises(RuntimeError, match="Missing 'labels'"):
        dp.create_hf_dataset()


def test_tokenize_data_raises_if_called_before_create_hf_dataset(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
):
    """Ensure tokenize_data raises an error when called before create_hf_dataset initializes the HF dataset."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=True,
    )

    dp.split_data().get_multi_hot_vectors()

    with pytest.raises(RuntimeError, match="Hugging Face DatasetDict not found"):
        dp.tokenize_data()


@pytest.mark.parametrize("hyperparameter_tuning", [True, False])
def test_tokenize_data_produces_correct_tokenized_structure_across_splits(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
    hyperparameter_tuning: bool,
):
    """Verify that tokenize_data produces a correctly structured DatasetDict with the expected splits and fields."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=hyperparameter_tuning,
    )

    dp.split_data().get_multi_hot_vectors().create_hf_dataset()
    dp.tokenize_data()

    assert isinstance(dp.tokenized_dataset, DatasetDict)

    expected_keys = ["train", "test"]
    if hyperparameter_tuning:
        expected_keys.insert(2, "validation")

    assert list(dp.tokenized_dataset.keys()) == expected_keys

    example = dp.tokenized_dataset["train"][0]

    assert "input_ids" in example
    assert "attention_mask" in example
    assert "labels" in example

    assert isinstance(example["labels"], torch.Tensor)
    assert example["labels"].dtype == torch.float32

    assert len(example["input_ids"]) == dp.sequence_length
    assert len(example["attention_mask"]) == dp.sequence_length


def test_tokenize_data_preserves_split_sizes(
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
):
    """Ensure that tokenization preserves the number of samples in each dataset split."""
    dp = pipeline_instance_factory(
        raw_csv=sample_raw_csv,
        raw_test_csv=sample_raw_test_csv,
        hyperparameter_tuning=True,
    )

    dp.split_data().get_multi_hot_vectors().create_hf_dataset()
    original_counts = {k: len(v) for k, v in dp.hf_dataset.items()}

    dp.tokenize_data()
    new_counts = {k: len(v) for k, v in dp.tokenized_dataset.items()}

    assert original_counts == new_counts
