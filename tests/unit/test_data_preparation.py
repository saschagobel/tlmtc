"""Tests for data preparation helpers."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from tlmtc.data_contracts import SPLIT_GROUP_COL, TEXT_COL, TEXT_PAIR_COL, DataContractError, InputMode
from tlmtc.data_preparation import (
    df_preprocess,
    df_save,
    df_split,
    read_prediction_data,
    tokenize_batch,
    tokenize_prediction_dataset,
)


class RecordingTokenizer:
    """Minimal tokenizer stand-in that records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.output = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((args, kwargs))
        return self.output


class TestDfPreprocess:
    """Test suite for the _df_preprocess utility function."""

    def test_rejects_csv_with_missing_required_values(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": ["a", None, "c"],
                "label_1": [1, 0, 0],
                "label_2": [0, 1, 1],
            }
        )
        df_in.to_csv(csv_path, index=False)

        with pytest.raises(DataContractError, match="multilabel data contract"):
            df_preprocess(csv_path)

    def test_rejects_dataframe_with_missing_required_values(self) -> None:
        df_in = pd.DataFrame(
            {
                "text": ["a", None, "c"],
                "label_1": [1, 0, 0],
                "label_2": [0, 1, 1],
            }
        )

        with pytest.raises(DataContractError, match="multilabel data contract"):
            df_preprocess(df_in)

    def test_accepts_missing_values_in_extra_columns_without_dropping_rows(self) -> None:
        df_in = pd.DataFrame(
            {
                "text": ["a", "b", "c"],
                "label_1": [1, 0, 0],
                "label_2": [0, 1, 1],
                "provenance": ["source-a", None, "source-c"],
            },
            index=[10, 20, 30],
        )

        df, label_cols, X, y, input_mode = df_preprocess(df_in)

        assert label_cols == ["label_1", "label_2"]
        assert input_mode is InputMode.SINGLE_TEXT
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "text": ["a", "b", "c"],
                    "label_1": [1, 0, 0],
                    "label_2": [0, 1, 1],
                },
                index=[10, 20, 30],
            ),
        )
        np.testing.assert_array_equal(X, np.array(["a", "b", "c"]))
        np.testing.assert_array_equal(y, np.array([[1, 0], [0, 1], [0, 1]]))

    def test_preserves_optional_text_pair_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": ["query a", "query b"],
                "text_pair": ["context a", "context b"],
                "label_1": [1, 0],
                "label_2": [0, 1],
            }
        )
        df_in.to_csv(csv_path, index=False)

        df, label_cols, X, y, input_mode = df_preprocess(csv_path)

        assert label_cols == ["label_1", "label_2"]
        assert input_mode is InputMode.PAIRED_TEXT
        pd.testing.assert_frame_equal(df, df_in)
        np.testing.assert_array_equal(X, np.array(["query a", "query b"]))
        np.testing.assert_array_equal(y, np.array([[1, 0], [0, 1]]))

    def test_preserves_optional_split_group_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                SPLIT_GROUP_COL: ["g1", "g1", "g2", "g2"],
                "label_1": [1, 0, 1, 0],
                "label_2": [0, 1, 0, 1],
            }
        )
        df_in.to_csv(csv_path, index=False)

        df, label_cols, X, y, input_mode = df_preprocess(csv_path)

        assert label_cols == ["label_1", "label_2"]
        assert input_mode is InputMode.SINGLE_TEXT
        pd.testing.assert_frame_equal(df, df_in)
        np.testing.assert_array_equal(X, np.array(["a", "b", "c", "d"]))
        np.testing.assert_array_equal(y, np.array([[1, 0], [0, 1], [1, 0], [0, 1]]))

class TestReadPredictionData:
    """Test suite for reading unlabeled prediction data."""

    def test_reads_csv_and_validates_prediction_frame(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        csv_path = tmp_path / "prediction.csv"
        df_in = pd.DataFrame(
            {
                TEXT_COL: ["first text", "second text"],
                "external_id": ["a", "b"],
            }
        )
        df_validated = df_in.assign(validated=True)
        df_in.to_csv(csv_path, index=False)

        def validate_frame(
            df: pd.DataFrame,
            expected_input_mode: InputMode,
        ) -> pd.DataFrame:
            pd.testing.assert_frame_equal(df, df_in)
            assert expected_input_mode is InputMode.SINGLE_TEXT
            return df_validated

        monkeypatch.setattr(
            "tlmtc.data_preparation.validate_prediction_frame",
            validate_frame,
        )

        result = read_prediction_data(
            data=csv_path,
            expected_input_mode=InputMode.SINGLE_TEXT,
        )

        pd.testing.assert_frame_equal(result, df_validated)

    def test_accepts_dataframe_and_validates_prediction_frame(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        df_in = pd.DataFrame(
            {
                TEXT_COL: ["first text", "second text"],
                "external_id": ["a", "b"],
            }
        )
        df_validated = df_in.assign(validated=True)

        def validate_frame(
            df: pd.DataFrame,
            expected_input_mode: InputMode,
        ) -> pd.DataFrame:
            pd.testing.assert_frame_equal(df, df_in)
            assert expected_input_mode is InputMode.SINGLE_TEXT
            return df_validated

        monkeypatch.setattr(
            "tlmtc.data_preparation.validate_prediction_frame",
            validate_frame,
        )

        result = read_prediction_data(
            data=df_in,
            expected_input_mode=InputMode.SINGLE_TEXT,
        )

        pd.testing.assert_frame_equal(result, df_validated)


class TestDfSplit:
    """Test suite for the _df_split utility function."""

    @staticmethod
    def _balanced_split_frame(n: int = 20) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(n)],
                "label_a": ([1, 0] * ((n + 1) // 2))[:n],
                "label_b": ([0, 1, 1, 0] * ((n + 3) // 4))[:n],
            }
        )

    def test_produces_expected_train_and_test_shapes(self) -> None:
        df = self._balanced_split_frame(n=20)
        text_values = df["text"].values
        label_matrix = df[["label_a", "label_b"]].values

        test_size = 0.3
        train, test = df_split(
            df=df, text_values=text_values, label_matrix=label_matrix, test_size=test_size, random_seed=42
        )

        assert len(train) + len(test) == len(df)

        expected_test = int(round(len(df) * test_size))
        assert abs(len(test) - expected_test) <= 1
        assert abs(len(train) - (len(df) - expected_test)) <= 1

        assert list(train.columns) == list(df.columns)
        assert list(test.columns) == list(df.columns)

    def test_split_is_reproducible_with_same_seed(self) -> None:
        df = self._balanced_split_frame(n=20)
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        train1, test1 = df_split(df, X, y, test_size=0.25, random_seed=123)
        train2, test2 = df_split(df, X, y, test_size=0.25, random_seed=123)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_train_and_test_have_no_overlap(self) -> None:
        df = self._balanced_split_frame(n=20)
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        train, test = df_split(df, X, y, test_size=0.2, random_seed=99)

        train_ids = set(train["text"])
        test_ids = set(test["text"])

        assert train_ids.isdisjoint(test_ids)

    def test_stratifies_labels_across_splits(self) -> None:
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(30)],
                "label_a": [0] * 10 + [1] * 20,
                "label_b": [1] * 15 + [0] * 15,
            }
        )
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        train, test = df_split(df, X, y, test_size=0.2, random_seed=7)

        for label in ["label_a", "label_b"]:
            assert train[label].nunique() == 2
            assert test[label].nunique() == 2

    @pytest.mark.parametrize(
        "rare_label_values",
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1]],
    )
    def test_raises_error_when_split_lacks_class_support(self, rare_label_values: list[int]) -> None:
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(8)],
                "label_a": [1, 1, 1, 1, 0, 0, 0, 0],
                "label_b": rare_label_values,
            }
        )
        text_values = df["text"].values
        label_matrix = df[["label_a", "label_b"]].values

        with pytest.raises(ValueError, match="Could not create a valid multilabel stratified split"):
            df_split(df=df, text_values=text_values, label_matrix=label_matrix, test_size=0.25, random_seed=42)

    def test_keeps_split_groups_disjoint_when_group_column_is_present(self) -> None:
        groups = [f"group_{i}" for i in range(20) for _ in range(2)]
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(40)],
                SPLIT_GROUP_COL: groups,
                "label_a": ([1, 0] * 20),
                "label_b": ([0, 1, 1, 0] * 10),
            }
        )
        text_values = df["text"].values
        label_matrix = df[["label_a", "label_b"]].values

        train, test = df_split(
            df=df,
            text_values=text_values,
            label_matrix=label_matrix,
            test_size=0.25,
            random_seed=42,
        )

        assert len(train) + len(test) == len(df)
        assert set(train[SPLIT_GROUP_COL]).isdisjoint(set(test[SPLIT_GROUP_COL]))

        for group in df[SPLIT_GROUP_COL].unique():
            group_in_train = group in set(train[SPLIT_GROUP_COL])
            group_in_test = group in set(test[SPLIT_GROUP_COL])
            assert group_in_train != group_in_test

    def test_emits_progress_when_grouped_split_row_fraction_drifts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages: list[str] = []

        monkeypatch.setattr(
            "tlmtc.data_preparation.emit_progress",
            messages.append,
        )

        df = pd.DataFrame(
            {
                "text": [f"large {i}" for i in range(18)] + ["small 1", "small 2"],
                SPLIT_GROUP_COL: ["large"] * 18 + ["small"] * 2,
                "label_a": [1, 0] * 10,
                "label_b": [0, 1] * 10,
            }
        )
        text_values = df["text"].values
        label_matrix = df[["label_a", "label_b"]].values

        df_split(
            df=df,
            text_values=text_values,
            label_matrix=label_matrix,
            test_size=0.5,
            random_seed=42,
        )

        assert len(messages) == 1
        assert messages[0].startswith("Grouped splitting produced a held-out row fraction of ")
        assert "requested 0.500" in messages[0]
        assert f"rows sharing the same '{SPLIT_GROUP_COL}' value must stay in the same split" in messages[0]


class TestDfSave:
    """Test suite for the _df_save utility function."""

    def test_creates_parquet_file_on_disk(self, tmp_path: Path) -> None:
        out_path = tmp_path / "data" / "train.parquet"
        df = pd.DataFrame({"text": ["a", "b"], "label_x": [1, 0]})

        df_save(df, out_path)

        assert out_path.exists()

    def test_preserves_dataframe_content_on_roundtrip(self, tmp_path: Path) -> None:
        out_path = tmp_path / "data" / "train.parquet"
        df = pd.DataFrame(
            {
                "text": ["hello", "world"],
                "label_a": [1, 0],
                "label_b": [0, 1],
            }
        )

        df_save(df, out_path)
        loaded = pd.read_parquet(out_path)

        # exact equality check
        pd.testing.assert_frame_equal(df, loaded)

    def test_creates_parent_directories_if_missing(self, tmp_path: Path) -> None:
        out_path = tmp_path / "data" / "train.parquet"
        df = pd.DataFrame({"x": [1, 2, 3]})

        assert not out_path.parent.exists()

        df_save(df, out_path)

        assert out_path.exists()
        assert out_path.parent.exists()


class TestTokenizeBatch:
    """Test suite for the tokenize_batch utility function."""

    def test_tokenizes_single_text_with_single_sequence_truncation(self) -> None:
        tokenizer = RecordingTokenizer()
        batch = {
            TEXT_COL: ["first text", "second text"],
        }

        result = tokenize_batch(
            batch=batch,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            input_mode=InputMode.SINGLE_TEXT,
            sequence_length=32,
        )

        assert result == tokenizer.output
        assert len(tokenizer.calls) == 1

        args, kwargs = tokenizer.calls[0]
        assert args == (batch[TEXT_COL],)
        assert kwargs == {
            "truncation": True,
            "padding": "max_length",
            "max_length": 32,
        }

    def test_tokenizes_paired_text_with_longest_first_truncation(self) -> None:
        tokenizer = RecordingTokenizer()
        batch = {
            TEXT_COL: ["query a", "query b"],
            TEXT_PAIR_COL: ["context a", "context b"],
        }

        result = tokenize_batch(
            batch=batch,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            input_mode=InputMode.PAIRED_TEXT,
            sequence_length=64,
        )

        assert result == tokenizer.output
        assert len(tokenizer.calls) == 1

        args, kwargs = tokenizer.calls[0]
        assert args == (batch[TEXT_COL], batch[TEXT_PAIR_COL])
        assert kwargs == {
            "truncation": "longest_first",
            "padding": "max_length",
            "max_length": 64,
        }


class TestTokenizePredictionDataset:
    """Test suite for tokenizing unlabeled prediction datasets."""

    def test_loads_persisted_tokenizer_and_tokenizes_single_text_dataset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tokenizer = RecordingTokenizer()
        tokenizer_dir = tmp_path / "model"

        def from_pretrained(
            persisted_tokenizer_dir: Path,
            *,
            trust_remote_code: bool,
            local_files_only: bool,
        ) -> RecordingTokenizer:
            assert persisted_tokenizer_dir == tokenizer_dir
            assert trust_remote_code is False
            assert local_files_only is True
            return tokenizer

        monkeypatch.setattr("tlmtc.data_preparation.AutoTokenizer.from_pretrained", from_pretrained)

        dataset = Dataset.from_dict(
            {
                TEXT_COL: ["first text"],
            }
        )

        tokenized = tokenize_prediction_dataset(
            dataset=dataset,
            tokenizer_dir=tokenizer_dir,
            input_mode=InputMode.SINGLE_TEXT,
            sequence_length=32,
            trust_remote_code=False,
        )

        assert len(tokenizer.calls) == 1

        args, kwargs = tokenizer.calls[0]
        assert args == (["first text"],)
        assert kwargs == {
            "truncation": True,
            "padding": "max_length",
            "max_length": 32,
        }

        assert TEXT_COL not in tokenized.column_names
        assert "input_ids" in tokenized.column_names
        assert "attention_mask" in tokenized.column_names
        assert tokenized[0]["input_ids"].tolist() == [1, 2, 3]
        assert tokenized[0]["attention_mask"].tolist() == [1, 1, 1]

    def test_loads_persisted_tokenizer_and_tokenizes_paired_text_dataset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tokenizer = RecordingTokenizer()
        tokenizer_dir = tmp_path / "model"

        def from_pretrained(
            persisted_tokenizer_dir: Path,
            *,
            trust_remote_code: bool,
            local_files_only: bool,
        ) -> RecordingTokenizer:
            assert persisted_tokenizer_dir == tokenizer_dir
            assert trust_remote_code is True
            assert local_files_only is True
            return tokenizer

        monkeypatch.setattr("tlmtc.data_preparation.AutoTokenizer.from_pretrained", from_pretrained)

        dataset = Dataset.from_dict(
            {
                TEXT_COL: ["query a"],
                TEXT_PAIR_COL: ["context a"],
            }
        )

        tokenized = tokenize_prediction_dataset(
            dataset=dataset,
            tokenizer_dir=tokenizer_dir,
            input_mode=InputMode.PAIRED_TEXT,
            sequence_length=64,
            trust_remote_code=True,
        )

        assert len(tokenizer.calls) == 1

        args, kwargs = tokenizer.calls[0]
        assert args == (["query a"], ["context a"])
        assert kwargs == {
            "truncation": "longest_first",
            "padding": "max_length",
            "max_length": 64,
        }

        assert TEXT_COL not in tokenized.column_names
        assert TEXT_PAIR_COL not in tokenized.column_names
        assert "input_ids" in tokenized.column_names
        assert "attention_mask" in tokenized.column_names
        assert tokenized[0]["input_ids"].tolist() == [1, 2, 3]
        assert tokenized[0]["attention_mask"].tolist() == [1, 1, 1]

    def test_uses_numpy_format_for_onnx_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tokenizer = RecordingTokenizer()
        monkeypatch.setattr("tlmtc.data_preparation.AutoTokenizer.from_pretrained", lambda *_, **__: tokenizer)

        tokenized = tokenize_prediction_dataset(
            dataset=Dataset.from_dict({TEXT_COL: ["first text"]}),
            tokenizer_dir=Path("model"),
            input_mode=InputMode.SINGLE_TEXT,
            sequence_length=32,
            trust_remote_code=False,
            inference_backend="onnx",
        )

        assert isinstance(tokenized[0]["input_ids"], np.ndarray)
        assert tokenized[0]["input_ids"].tolist() == [1, 2, 3]
