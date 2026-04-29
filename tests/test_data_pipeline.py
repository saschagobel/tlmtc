"""Tests for the DataPipeline class."""

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import torch
from datasets import Dataset, DatasetDict

from tlmtc.data_pipeline import DataPipeline
from tlmtc.paths import RunPaths
from tlmtc.settings import ModelSettings, SplitSettings


@pytest.fixture
def sample_raw_csv(tmp_path: Path):
    """Create a small synthetic multi-label dataset and write it to raw.csv."""
    df = pd.DataFrame(
        {
            "text": [
                "hello world ooqz",
                "foo bar",
                "hello",
                "bar world qooz",
                "alpha beta",
                "gamma delta",
                "epsilon zeta",
                "eta theta",
                "iota kappa",
                "lambda mu",
                "nu xi",
                "omicron pi",
            ],
            "label_a": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "label_b": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
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
            "text": ["ooqz world", "foo Hello", "hello bar", "bar baz"],
            "label_a": [1, 1, 0, 0],
            "label_b": [1, 0, 1, 0],
        }
    )
    csv_path = tmp_path / "raw_test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def split_settings() -> SplitSettings:
    """Return deterministic split configuration for test runs."""
    return SplitSettings(validation_size=0.25, test_size=0.25, random_seed=42)


@pytest.fixture
def model_settings() -> ModelSettings:
    """Return minimal model configuration for tokenizer-based test pipeline."""
    return ModelSettings(
        target_name="test-target",
        proxy_checkpoint="unused-here",
        checkpoint="tests/data/tiny_tokenizer",
        sequence_length=16,
    )


@pytest.fixture
def pipeline_instance_factory(
    tmp_path: Path,
    split_settings: SplitSettings,
    model_settings: ModelSettings,
) -> Callable[..., DataPipeline]:
    """Factory for DataPipeline instances using in-test RunPaths."""

    def _factory(*, raw_csv: Path, raw_test_csv: Path | None) -> DataPipeline:
        run_id = "test-run"
        run_dir = tmp_path / "tlmtc_outputs" / run_id
        data_dir = run_dir / "data"
        logs_dir = run_dir / "logs"
        model_dir = run_dir / "model"

        paths = RunPaths(
            work_dir=tmp_path,
            run_dir=run_dir,
            run_id=run_id,
            raw_data_path=raw_csv,
            raw_test_data_path=raw_test_csv,
            data_dir=data_dir,
            logs_dir=logs_dir,
            model_dir=model_dir,
            train_data_path=data_dir / "train.parquet",
            val_data_path=data_dir / "val.parquet",
            test_data_path=data_dir / "test.parquet",
        ).ensure_dirs()

        return DataPipeline(paths=paths, split=split_settings, model=model_settings)

    return _factory


class TestSplitData:
    """Test suite for the DataPipeline.split_data method."""

    @pytest.mark.parametrize("use_raw_test", [False, True])
    def test_splits_data_correctly_across_all_configurations(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
        use_raw_test: bool,
    ):
        """Ensure split_data produces correct train/val/test partitions under all tuning and raw-test scenarios."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv if use_raw_test else None,
        )

        dp.split_data()

        assert dp.paths.train_data_path.exists()
        assert dp.paths.val_data_path.exists()
        assert dp.paths.test_data_path.exists()

        train_df = pd.read_parquet(dp.paths.train_data_path)
        val_df = pd.read_parquet(dp.paths.val_data_path)
        test_df = pd.read_parquet(dp.paths.test_data_path)

        assert len(train_df) > 0
        assert len(test_df) > 0

        if use_raw_test:
            assert len(test_df) == len(pd.read_csv(sample_raw_test_csv))
            assert len(train_df) + len(val_df) == len(pd.read_csv(sample_raw_csv))
        else:
            assert len(train_df) + len(val_df) + len(test_df) == len(pd.read_csv(sample_raw_csv))

    def test_raises_error_when_raw_data_missing(self, tmp_path: Path, pipeline_instance_factory: Callable):
        """Ensure split_data raises FileNotFoundError when raw_data_path does not exist."""
        missing_raw = tmp_path / "does_not_exist.csv"

        dp = pipeline_instance_factory(
            raw_csv=missing_raw,
            raw_test_csv=None,
        )

        with pytest.raises(FileNotFoundError):
            dp.split_data()

    def test_raises_error_when_explicit_raw_test_data_missing(
        self,
        tmp_path: Path,
        sample_raw_csv: Path,
        pipeline_instance_factory: Callable[..., DataPipeline],
    ) -> None:
        missing_raw_test = tmp_path / "missing_raw_test.csv"

        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=missing_raw_test,
        )

        with pytest.raises(FileNotFoundError, match="Raw test data not found"):
            dp.split_data()

    def test_raises_error_on_mismatched_label_columns(
        self,
        tmp_path: Path,
        sample_raw_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure split_data raises ValueError when train and test label columns differ."""
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
        )

        with pytest.raises(ValueError, match="Mismatch between train/test label columns"):
            dp.split_data()


class TestGetMultiHot:
    """Test suite for the DataPipeline.get_multi_hot_vectors method."""

    def test_converts_label_columns_to_multi_hot_vectors_across_splits(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure get_multi_hot_vectors converts label_* columns into multi-hot vectors for all dataset splits."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors()

        expected_splits = ["train_data", "val_data", "test_data"]

        for split_attr in expected_splits:
            df = getattr(dp, split_attr)

            assert list(df.columns) == ["text", "labels"]

            labels = df["labels"].iloc[0]
            assert isinstance(labels, list)
            assert len(labels) == 2

    def test_raises_error_if_called_before_split_data(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure get_multi_hot_vectors raises RuntimeError when invoked before split_data initializes splits."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        with pytest.raises(RuntimeError, match="Train/val/test data not found"):
            dp.get_multi_hot_vectors()


class TestCreateHFDataset:
    """Test suite for the DataPipeline.create_hf_dataset method."""

    def test_constructs_correct_datasetdict_splits_and_schema(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure create_hf_dataset builds the expected DatasetDict with correct splits and feature schema."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors()
        dp.create_hf_dataset()

        assert isinstance(dp.hf_dataset, DatasetDict)

        assert set(dp.hf_dataset.keys()) == {"train", "validation", "test"}

        train_split = dp.hf_dataset["train"]
        assert isinstance(train_split, Dataset)

        assert "text" in train_split.features
        assert "labels" in train_split.features

        labels_feature = train_split.features["labels"]
        assert labels_feature.feature.dtype == "int64"

    def test_raises_error_if_called_before_split_data(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure create_hf_dataset raises RuntimeError when train/test DataFrames are not yet initialized."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        with pytest.raises(RuntimeError, match="Train/val/test data not found"):
            dp.create_hf_dataset()

    def test_raises_error_when_labels_column_missing(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure create_hf_dataset raises RuntimeError when called before multi-hot labels are generated."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        dp.split_data()

        with pytest.raises(RuntimeError, match="Missing 'labels'"):
            dp.create_hf_dataset()


class TestTokenizeData:
    """Test suite for the DataPipeline.tokenize_data method."""

    def test_raises_error_if_called_before_hf_dataset_created(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure tokenize_data raises RuntimeError when invoked before create_hf_dataset."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors()

        with pytest.raises(RuntimeError, match="Hugging Face DatasetDict not found"):
            dp.tokenize_data()

    def test_produces_correct_tokenized_structure_across_splits(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure properly tokenized DatasetDict with tensors for input_ids, attention_mask, and labels."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors().create_hf_dataset()
        dp.tokenize_data()

        assert isinstance(dp.tokenized_dataset, DatasetDict)

        assert set(dp.tokenized_dataset.keys()) == {"train", "validation", "test"}

        example = dp.tokenized_dataset["train"][0]

        assert "input_ids" in example
        assert "attention_mask" in example
        assert "labels" in example

        assert isinstance(example["labels"], torch.Tensor)
        assert example["labels"].dtype == torch.float32

        assert len(example["input_ids"]) == dp.model.sequence_length
        assert len(example["attention_mask"]) == dp.model.sequence_length

    def test_preserves_original_split_sizes(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable,
    ):
        """Ensure tokenize_data preserves the number of examples in each split."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors().create_hf_dataset()
        original_counts = {k: len(v) for k, v in dp.hf_dataset.items()}

        dp.tokenize_data()
        new_counts = {k: len(v) for k, v in dp.tokenized_dataset.items()}

        assert original_counts == new_counts
