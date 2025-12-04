"""Tests for data preparation utility functions."""

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

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
def test_split_data(
    tmp_path: Path,
    sample_raw_csv: Path,
    sample_raw_test_csv: Path,
    pipeline_instance_factory: Callable,
    hyperparameter_tuning: bool,
    use_raw_test: bool,
):
    """Test all four behavioral branches of DataPipeline.split_data()."""
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

    # --- Label column checks (sanity for stratification) ---
    assert "label_a" in train_df.columns
    assert "label_b" in train_df.columns


def test_split_data_raises_missing_raw_data(tmp_path: Path, pipeline_instance_factory: Callable):
    """Test raise when raw_data_path does not exist."""
    missing_raw = tmp_path / "does_not_exist.csv"

    dp = pipeline_instance_factory(
        raw_csv=missing_raw,
        raw_test_csv="",
        hyperparameter_tuning=True,
    )

    with pytest.raises(FileNotFoundError):
        dp.split_data()


def test_split_data_raises_mismatched_label_columns(
    tmp_path: Path,
    sample_raw_csv: Path,
    pipeline_instance_factory: Callable,
):
    """Test raise when train/test label columns differ."""
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
