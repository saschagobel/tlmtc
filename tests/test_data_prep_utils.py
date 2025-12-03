"""Tests for data preparation utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlmtc.utils import _df_preprocess, _df_save, _df_split


def test_df_preprocess_valid_input(tmp_path: Path):
    """Test success with valid input."""
    csv_path = tmp_path / "data.csv"
    df_in = pd.DataFrame({
        "text": ["a", None, "c"],
        "label_1": [1, 0, 1],
        "label_2": [0, 1, None],
    })
    df_in.to_csv(csv_path, index=False)

    df, label_cols, X, y = _df_preprocess(csv_path)

    assert len(df) == 1 # Should drop rows with NA
    assert set(label_cols) == {"label_1", "label_2"}
    assert np.array_equal(X, np.array(["a"]))
    assert y.shape == (1, 2)
    assert np.array_equal(y, df[label_cols].values)


def test_df_preprocess_empty_data(tmp_path: Path):
    """Test failure with empty data frame."""
    csv_path = tmp_path / "data.csv"
    df_in = pd.DataFrame({
        "text": [None],
        "label_1": [1],
        "label_2": [0],
    })
    df_in.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="no valid samples"):
        _df_preprocess(csv_path)


def test_df_preprocess_missing_text_column(tmp_path: Path):
    """Test failure with missing text column."""
    csv_path = tmp_path / "data.csv"
    df_in = pd.DataFrame({
        "label_1": [1],
        "label_2": [0],
    })
    df_in.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="text"):
        _df_preprocess(csv_path)


def test_df_preprocess_not_enough_label_columns(tmp_path: Path):
    """Test failure with only one label column."""
    csv_path = tmp_path / "data.csv"
    df_in = pd.DataFrame({
        "text": ["a"], "label_1": [1]
    })
    df_in.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="at least two"):
        _df_preprocess(csv_path)


def test_df_preprocess_label_not_integer(tmp_path: Path):
    """Test failure with non-integer-like label column."""
    csv_path = tmp_path / "data.csv"
    df_in = pd.DataFrame({
        "text": ["a"],
        "label_1": [1],
        "label_2": ["b"],
    })
    df_in.to_csv(csv_path, index=False)

    with pytest.raises(TypeError):
        _df_preprocess(csv_path)


def test_df_preprocess_label_out_of_bounds(tmp_path: Path):
    """Test failure with non-binary label column."""
    csv_path = tmp_path / "data.csv"
    df_in = pd.DataFrame({
        "text": ["a"],
        "label_1": [1],
        "label_2": [2],
    })
    df_in.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="binary values"):
        _df_preprocess(csv_path)


def test_df_split_basic_shapes():
    """Test output shapes with valid input."""
    df = pd.DataFrame({
        "text": [f"sample {i}" for i in range(10)],
        "label_a": np.random.randint(0,2,10),
        "label_b": np.random.randint(0,2,10),
    })
    X = df["text"].values
    y = df[["label_a", "label_b"]].values

    train, test = _df_split(df=df, X=X, y=y, test_size=0.3, random_seed=42)

    assert len(train) + len(test) == len(df)
    assert len(test) == 3
    assert len(train) == 7
    assert list(train.columns) == list(df.columns)
    assert list(test.columns) == list(df.columns)


def test_df_split_reproducible():
    """Test deterministic behavior."""
    df = pd.DataFrame({
        "text": [f"sample {i}" for i in range(12)],
        "label_a": np.random.randint(0,2,12),
        "label_b": np.random.randint(0,2,12),
    })
    X = df["text"].values
    y = df[["label_a", "label_b"]].values

    train1, test1 = _df_split(df, X, y, test_size=0.25, random_seed=123)
    train2, test2 = _df_split(df, X, y, test_size=0.25, random_seed=123)

    # Identical splits
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_df_split_no_overlap():
    """Test overlap between train and test."""
    df = pd.DataFrame({
        "text": [f"sample {i}" for i in range(20)],
        "label_a": np.random.randint(0,2,20),
        "label_b": np.random.randint(0,2,20),
    })
    X = df["text"].values
    y = df[["label_a", "label_b"]].values

    train, test = _df_split(df, X, y, test_size=0.2, random_seed=99)

    # Unique ids
    train_ids = set(train["text"])
    test_ids = set(test["text"])

    assert train_ids.isdisjoint(test_ids)


def test_df_split_stratifies_labels():
    """Test presence of label categories in splits."""
    df = pd.DataFrame({
        "text": [f"sample {i}" for i in range(30)],
        "label_a": [0]*10 + [1]*20,
        "label_b": [1]*15 + [0]*15,
    })
    X = df["text"].values
    y = df[["label_a", "label_b"]].values

    train, test = _df_split(df, X, y, test_size=0.2, random_seed=7)

    for label in ["label_a", "label_b"]:
        assert train[label].nunique() == 2
        assert test[label].nunique() == 2


def test_df_save_creates_parquet_file(tmp_path: Path):
    """Test presence of parquet files."""
    out_path = tmp_path / "data" / "train.parquet"
    df = pd.DataFrame({
        "text": ["a", "b"], "label_x": [1, 0]
    })

    _df_save(df, out_path)

    assert out_path.exists()


def test_df_save_roundtrip_content(tmp_path: Path):
    """Test preservation of columns and values."""
    out_path = tmp_path / "data" / "train.parquet"
    df = pd.DataFrame({
        "text": ["hello", "world"],
        "label_a": [1, 0],
        "label_b": [0, 1],
    })

    _df_save(df, out_path)
    loaded = pd.read_parquet(out_path)

    # exact equality check
    pd.testing.assert_frame_equal(df, loaded)


def test_df_save_creates_parent_directories(tmp_path: Path):
    """Test automatic creation of parent folders."""
    out_path = tmp_path / "data" / "train.parquet"
    df = pd.DataFrame({"x": [1, 2, 3]})

    assert not out_path.parent.exists()

    _df_save(df, out_path)

    assert out_path.exists()
    assert out_path.parent.exists()