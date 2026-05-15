"""Tests for the DataPipeline class."""

from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from datasets import Dataset, DatasetDict

from tlmtc.data_contracts import TEXT_PAIR_COL, DataContractError, InputMode
from tlmtc.data_pipeline import DataPipeline
from tlmtc.paths import resolve_paths
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
def sample_paired_raw_csv(tmp_path: Path) -> Path:
    """Create a small synthetic paired-text multi-label dataset and write it to raw.csv."""
    df = pd.DataFrame(
        {
            "text": [
                "query hello world",
                "query foo bar",
                "query hello",
                "query bar world",
                "query alpha beta",
                "query gamma delta",
                "query epsilon zeta",
                "query eta theta",
                "query iota kappa",
                "query lambda mu",
                "query nu xi",
                "query omicron pi",
            ],
            TEXT_PAIR_COL: [
                "answer hello world",
                "answer foo bar",
                "answer hello",
                "answer bar world",
                "answer alpha beta",
                "answer gamma delta",
                "answer epsilon zeta",
                "answer eta theta",
                "answer iota kappa",
                "answer lambda mu",
                "answer nu xi",
                "answer omicron pi",
            ],
            "label_a": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "label_b": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        }
    )
    csv_path = tmp_path / "paired_raw.csv"
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
def sample_paired_raw_test_csv(tmp_path: Path) -> Path:
    """Create a small synthetic paired-text multi-label test dataset and write it to raw_test.csv."""
    df = pd.DataFrame(
        {
            "text": ["query ooqz world", "query foo hello", "query hello bar", "query bar baz"],
            TEXT_PAIR_COL: ["answer ooqz world", "answer foo hello", "answer hello bar", "answer bar baz"],
            "label_a": [1, 1, 0, 0],
            "label_b": [1, 0, 1, 0],
        }
    )
    csv_path = tmp_path / "paired_raw_test.csv"
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
        paths = resolve_paths(
            raw_csv=raw_csv,
            raw_test_csv=raw_test_csv,
            work_dir=tmp_path,
            run_id="test-run",
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

        assert dp.input_mode is InputMode.SINGLE_TEXT
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
    ) -> None:
        """Ensure split_data rejects train/test files with different label columns."""
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

        with pytest.raises(DataContractError, match="Label column mismatch between raw_csv and raw_test_csv"):
            dp.split_data()

    def test_sets_paired_text_input_mode_when_text_pair_is_present(
        self,
        sample_paired_raw_csv: Path,
        sample_paired_raw_test_csv: Path,
        pipeline_instance_factory: Callable[..., DataPipeline],
    ) -> None:
        """Ensure split_data records paired-text mode when text_pair is present."""
        dp = pipeline_instance_factory(
            raw_csv=sample_paired_raw_csv,
            raw_test_csv=sample_paired_raw_test_csv,
        )

        dp.split_data()

        assert dp.input_mode is InputMode.PAIRED_TEXT

        train_df = pd.read_parquet(dp.paths.train_data_path)
        val_df = pd.read_parquet(dp.paths.val_data_path)
        test_df = pd.read_parquet(dp.paths.test_data_path)

        assert TEXT_PAIR_COL in train_df.columns
        assert TEXT_PAIR_COL in val_df.columns
        assert TEXT_PAIR_COL in test_df.columns

    def test_raises_error_on_mismatched_input_modes_between_raw_and_raw_test(
        self,
        sample_raw_csv: Path,
        sample_paired_raw_test_csv: Path,
        pipeline_instance_factory: Callable[..., DataPipeline],
    ) -> None:
        """Ensure split_data rejects single-text train data with paired-text test data."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_paired_raw_test_csv,
        )

        with pytest.raises(DataContractError, match="Input mode mismatch between raw_csv and raw_test_csv"):
            dp.split_data()

    def test_validates_persisted_split_input_modes_when_reusing_cached_splits(
        self,
        sample_raw_csv: Path,
        pipeline_instance_factory: Callable[..., DataPipeline],
    ) -> None:
        """Ensure cached persisted splits are revalidated before reuse."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=None,
        )
        dp.split_data()

        test_df = pd.read_parquet(dp.paths.test_data_path)
        test_df[TEXT_PAIR_COL] = ["paired text"] * len(test_df)
        test_df.to_parquet(dp.paths.test_data_path, index=False)

        resumed_dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=None,
        )

        with pytest.raises(DataContractError, match="Input mode mismatch between persisted splits"):
            resumed_dp.split_data()


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

    def test_preserves_text_pair_column_for_paired_text_inputs(
        self,
        sample_paired_raw_csv: Path,
        sample_paired_raw_test_csv: Path,
        pipeline_instance_factory: Callable[..., DataPipeline],
    ) -> None:
        """Ensure paired-text inputs keep text_pair when labels are converted to multi-hot vectors."""
        dp = pipeline_instance_factory(
            raw_csv=sample_paired_raw_csv,
            raw_test_csv=sample_paired_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors()

        for split_attr in ("train_data", "val_data", "test_data"):
            df = getattr(dp, split_attr)

            assert list(df.columns) == ["text", TEXT_PAIR_COL, "labels"]

            labels = df["labels"].iloc[0]
            assert isinstance(labels, list)
            assert len(labels) == 2


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
        assert TEXT_PAIR_COL not in train_split.features
        assert "labels" in train_split.features

        assert train_split.features["text"].dtype == "string"
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

    def test_constructs_paired_text_dataset_schema_when_text_pair_is_present(
        self,
        sample_paired_raw_csv: Path,
        sample_paired_raw_test_csv: Path,
        pipeline_instance_factory: Callable[..., DataPipeline],
    ) -> None:
        """Ensure create_hf_dataset includes text_pair in paired-text mode."""
        dp = pipeline_instance_factory(
            raw_csv=sample_paired_raw_csv,
            raw_test_csv=sample_paired_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors().create_hf_dataset()

        assert isinstance(dp.hf_dataset, DatasetDict)

        train_split = dp.hf_dataset["train"]

        assert "text" in train_split.features
        assert TEXT_PAIR_COL in train_split.features
        assert "labels" in train_split.features

        assert train_split.features["text"].dtype == "string"
        assert train_split.features[TEXT_PAIR_COL].dtype == "string"

        labels_feature = train_split.features["labels"]
        assert labels_feature.feature.dtype == "int64"

        example = train_split[0]
        assert isinstance(example["text"], str)
        assert isinstance(example[TEXT_PAIR_COL], str)
        assert isinstance(example["labels"], list)


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

    def test_raises_error_if_input_mode_is_missing(
        self,
        sample_raw_csv: Path,
        sample_raw_test_csv: Path,
        pipeline_instance_factory: Callable[..., DataPipeline],
    ) -> None:
        """Ensure tokenize_data raises RuntimeError when input mode has not been inferred."""
        dp = pipeline_instance_factory(
            raw_csv=sample_raw_csv,
            raw_test_csv=sample_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors().create_hf_dataset()
        dp.input_mode = None

        with pytest.raises(RuntimeError, match="Input mode not found"):
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
        assert dp.input_mode is InputMode.SINGLE_TEXT
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

    def test_delegates_paired_text_tokenization_with_paired_input_mode(
        self,
        sample_paired_raw_csv: Path,
        sample_paired_raw_test_csv: Path,
        pipeline_instance_factory: Callable[..., DataPipeline],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure paired-text tokenization delegates with text_pair and paired input mode."""

        def fake_tokenize_batch(
            *,
            batch,
            tokenizer,
            input_mode,
            sequence_length,
        ):
            return {
                "input_ids": [[1] * sequence_length for _ in batch["text"]],
                "attention_mask": [[1] * sequence_length for _ in batch["text"]],
            }

        tokenizer = object()
        tokenizer_loader = Mock(return_value=tokenizer)
        tokenize_batch_spy = Mock(side_effect=fake_tokenize_batch)

        monkeypatch.setattr(
            "tlmtc.data_pipeline.AutoTokenizer.from_pretrained",
            tokenizer_loader,
        )
        monkeypatch.setattr(
            "tlmtc.data_pipeline.tokenize_batch",
            tokenize_batch_spy,
        )

        dp = pipeline_instance_factory(
            raw_csv=sample_paired_raw_csv,
            raw_test_csv=sample_paired_raw_test_csv,
        )

        dp.split_data().get_multi_hot_vectors().create_hf_dataset()
        dp.tokenize_data()

        tokenizer_loader.assert_called_once_with(
            dp.model.checkpoint,
            trust_remote_code=False,
        )

        assert tokenize_batch_spy.call_count > 0

        for call in tokenize_batch_spy.call_args_list:
            kwargs = call.kwargs

            assert kwargs["tokenizer"] is tokenizer
            assert kwargs["input_mode"] is InputMode.PAIRED_TEXT
            assert kwargs["sequence_length"] == dp.model.sequence_length
            assert TEXT_PAIR_COL in kwargs["batch"]

        assert isinstance(dp.tokenized_dataset, DatasetDict)

        example = dp.tokenized_dataset["train"][0]
        assert "input_ids" in example
        assert "attention_mask" in example
        assert "labels" in example
        assert isinstance(example["labels"], torch.Tensor)
        assert example["labels"].dtype == torch.float32
