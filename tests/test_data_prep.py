"""Tests for data preparation helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlmtc.data_prep import _df_preprocess, _df_save, _df_split


class TestDfPreprocess:
    """Test suite for the _df_preprocess utility function."""

    def test_returns_processed_data_given_valid_input(self, tmp_path: Path):
        """Ensure valid input is processed and parsed into X, y, and label columns."""
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": ["a", None, "c"],
                "label_1": [1, 0, 1],
                "label_2": [0, 1, None],
            }
        )
        df_in.to_csv(csv_path, index=False)

        df, label_cols, X, y = _df_preprocess(csv_path)

        assert len(df) == 1  # Should drop rows with NA
        assert set(label_cols) == {"label_1", "label_2"}
        assert np.array_equal(X, np.array(["a"]))
        assert y.shape == (1, 2)
        assert np.array_equal(y, df[label_cols].values)

    def test_raises_error_given_only_invalid_rows(self, tmp_path: Path):
        """Ensure an error is raised when all rows become invalid after dropping NAs."""
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": [None],
                "label_1": [1],
                "label_2": [0],
            }
        )
        df_in.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="no valid samples"):
            _df_preprocess(csv_path)

    def test_raises_error_when_text_column_missing(self, tmp_path: Path):
        """Ensure a ValueError is raised when the 'text' column is absent."""
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "label_1": [1],
                "label_2": [0],
            }
        )
        df_in.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="text"):
            _df_preprocess(csv_path)

    def test_raises_error_when_less_than_two_label_columns(self, tmp_path: Path):
        """Ensure at least two label_* columns are required for multi-label classification."""
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame({"text": ["a"], "label_1": [1]})
        df_in.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="at least two"):
            _df_preprocess(csv_path)

    def test_raises_error_when_label_column_has_non_numeric_values(self, tmp_path: Path):
        """Ensure label columns must contain only integer-like values."""
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": ["a"],
                "label_1": [1],
                "label_2": ["b"],
            }
        )
        df_in.to_csv(csv_path, index=False)

        with pytest.raises(TypeError):
            _df_preprocess(csv_path)

    def test_raises_error_when_label_values_are_outside_binary_range(self, tmp_path: Path):
        """Ensure label columns contain only binary values {0,1}."""
        csv_path = tmp_path / "data.csv"
        df_in = pd.DataFrame(
            {
                "text": ["a"],
                "label_1": [1],
                "label_2": [2],
            }
        )
        df_in.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="binary values"):
            _df_preprocess(csv_path)


class TestDfSplit:
    """Test suite for the _df_split utility function."""

    def test_produces_expected_train_and_test_shapes(self):
        """Ensure the split returns correctly sized train/test DataFrames with original columns."""
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
        train, test = _df_split(df=df, X=X, y=y, test_size=test_size, random_seed=42)

        assert len(train) + len(test) == len(df)

        expected_test = int(round(len(df) * test_size))
        assert abs(len(test) - expected_test) <= 1
        assert abs(len(train) - (len(df) - expected_test)) <= 1

        assert list(train.columns) == list(df.columns)
        assert list(test.columns) == list(df.columns)

    def test_split_is_reproducible_with_same_seed(self):
        """Ensure using the same random seed yields identical train/test splits."""
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(12)],
                "label_a": np.random.randint(0, 2, 12),
                "label_b": np.random.randint(0, 2, 12),
            }
        )
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        train1, test1 = _df_split(df, X, y, test_size=0.25, random_seed=123)
        train2, test2 = _df_split(df, X, y, test_size=0.25, random_seed=123)

        # Identical splits
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_train_and_test_have_no_overlap(self):
        """Ensure the train and test sets do not share any sample identifiers."""
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(20)],
                "label_a": np.random.randint(0, 2, 20),
                "label_b": np.random.randint(0, 2, 20),
            }
        )
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        train, test = _df_split(df, X, y, test_size=0.2, random_seed=99)

        # Unique ids
        train_ids = set(train["text"])
        test_ids = set(test["text"])

        assert train_ids.isdisjoint(test_ids)

    def test_stratifies_labels_across_splits(self):
        """Ensure both train and test sets contain all label categories for each label column."""
        df = pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(30)],
                "label_a": [0] * 10 + [1] * 20,
                "label_b": [1] * 15 + [0] * 15,
            }
        )
        X = df["text"].values
        y = df[["label_a", "label_b"]].values

        train, test = _df_split(df, X, y, test_size=0.2, random_seed=7)

        for label in ["label_a", "label_b"]:
            assert train[label].nunique() == 2
            assert test[label].nunique() == 2


class TestDfSave:
    """Test suite for the _df_save utility function."""

    def test_creates_parquet_file_on_disk(self, tmp_path: Path):
        """Ensure saving a DataFrame results in a parquet file at the target location."""
        out_path = tmp_path / "data" / "train.parquet"
        df = pd.DataFrame({"text": ["a", "b"], "label_x": [1, 0]})

        _df_save(df, out_path)

        assert out_path.exists()

    def test_preserves_dataframe_content_on_roundtrip(self, tmp_path: Path):
        """Ensure saving and reloading a DataFrame preserves all columns and values exactly."""
        out_path = tmp_path / "data" / "train.parquet"
        df = pd.DataFrame(
            {
                "text": ["hello", "world"],
                "label_a": [1, 0],
                "label_b": [0, 1],
            }
        )

        _df_save(df, out_path)
        loaded = pd.read_parquet(out_path)

        # exact equality check
        pd.testing.assert_frame_equal(df, loaded)

    def test_creates_parent_directories_if_missing(self, tmp_path: Path):
        """Ensure parent directories are automatically created when saving a parquet file."""
        out_path = tmp_path / "data" / "train.parquet"
        df = pd.DataFrame({"x": [1, 2, 3]})

        assert not out_path.parent.exists()

        _df_save(df, out_path)

        assert out_path.exists()
        assert out_path.parent.exists()
