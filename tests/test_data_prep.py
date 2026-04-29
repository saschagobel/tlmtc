"""Tests for data preparation helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlmtc.data_contracts import DataContractError
from tlmtc.data_prep import df_preprocess, df_save, df_split


class TestDfPreprocess:
    """Test suite for the _df_preprocess utility function."""

    def test_reads_csv_drops_missing_rows_and_extracts_arrays(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": ["a", None, "c"],
                "label_1": [1, 0, 0],
                "label_2": [0, 1, 1],
            }
        )
        df_in.to_csv(csv_path, index=False)

        df, label_cols, X, y = df_preprocess(csv_path)

        assert label_cols == ["label_1", "label_2"]
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "text": ["a", "c"],
                    "label_1": [1, 0],
                    "label_2": [0, 1],
                }
            ),
        )
        np.testing.assert_array_equal(X, np.array(["a", "c"]))
        np.testing.assert_array_equal(y, np.array([[1, 0], [0, 1]]))

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

        df, label_cols, X, y = df_preprocess(csv_path)

        assert label_cols == ["label_1", "label_2"]
        pd.testing.assert_frame_equal(df, df_in)
        np.testing.assert_array_equal(X, np.array(["query a", "query b"]))
        np.testing.assert_array_equal(y, np.array([[1, 0], [0, 1]]))

    def test_raises_contract_error_when_no_rows_remain_after_dropping_missing_values(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": [None],
                "label_1": [1],
                "label_2": [0],
            }
        )
        df_in.to_csv(csv_path, index=False)

        with pytest.raises(DataContractError, match="multilabel data contract"):
            df_preprocess(csv_path)


class TestDfSplit:
    """Test suite for the _df_split utility function."""

    def test_produces_expected_train_and_test_shapes(self) -> None:
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(10)],
                "label_a": np.random.randint(0, 2, 10),
                "label_b": np.random.randint(0, 2, 10),
            }
        )
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        test_size = 0.3
        train, test = df_split(df=df, X=X, y=y, test_size=test_size, random_seed=42)

        assert len(train) + len(test) == len(df)

        expected_test = int(round(len(df) * test_size))
        assert abs(len(test) - expected_test) <= 1
        assert abs(len(train) - (len(df) - expected_test)) <= 1

        assert list(train.columns) == list(df.columns)
        assert list(test.columns) == list(df.columns)

    def test_split_is_reproducible_with_same_seed(self) -> None:
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(12)],
                "label_a": np.random.randint(0, 2, 12),
                "label_b": np.random.randint(0, 2, 12),
            }
        )
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        train1, test1 = df_split(df, X, y, test_size=0.25, random_seed=123)
        train2, test2 = df_split(df, X, y, test_size=0.25, random_seed=123)

        # Identical splits
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_train_and_test_have_no_overlap(self) -> None:
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(20)],
                "label_a": np.random.randint(0, 2, 20),
                "label_b": np.random.randint(0, 2, 20),
            }
        )
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        train, test = df_split(df, X, y, test_size=0.2, random_seed=99)

        # Unique ids
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
