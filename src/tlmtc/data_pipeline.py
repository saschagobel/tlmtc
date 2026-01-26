"""
Transfer Learning for Multi-Label Text Classification.

Dataset preparation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers import AutoTokenizer

from tlmtc.utils import _df_preprocess, _df_save, _df_split


class DataPipeline:
    """
    Load, split, process and tokenize raw multi-label data.

    Attributes
    ----------
    raw_data_path : str or Path
        Path to the raw train CSV data file, with required columns 'text', 'label_*'
    raw_test_data_path : str or Path
        Path to the raw test CSV data file, with required columns 'text', 'label_*'
    train_data_path : str or Path
        Path where the training split will be saved
    val_data_path : str or Path
        Path where the validation split will be saved
    test_data_path : str or Path
        Path where the test split will be saved
    hyperparameter_tuning: bool
        Flag whether hyperparameter tuning should be performed
    validation_size: float
        Proportion of data set to be used for validation
    test_size : float
        Proportion of data set to be used for testing
    random_seed : int
        Random seed
    checkpoint : str
        Name of the pretrained model checkpoint on the Hugging Face Hub, used for tokenization
    sequence_length : int
        Maximum number of tokens per input text
    train_data : pandas.DataFrame
        Data Frame with columns 'text', 'label_*'
    val_data : pandas.DataFrame
        Data Frame with columns 'text', 'label_*'
    test_data : pandas.DataFrame
        Data Frame with columns 'text', 'label_*'
    hf_dataset : DatasetDict
        Hugging Face DatasetDict with 'train', 'validation' and 'test' splits
    tokenized_dataset : DatasetDict
        Tokenized dataset ready for PyTorch

    Methods
    -------
    split_data():
        Split data into train, validation and test partitions, reset indices, and save to disk as parquet files
    get_multi_hot_vectors():
        Combine multiple label columns into a single array per row (multi-hot vector)
    create_hf_dataset():
        Assemble train and test data in a Hugging Face DatasetDict
    tokenize_data():
        Tokenize text and convert multi-hot labels to float tensors
    """

    def __init__(
        self,
        raw_data_path: Union[str, Path],
        raw_test_data_path: Union[str, Path],
        train_data_path: Union[str, Path],
        val_data_path: Union[str, Path],
        test_data_path: Union[str, Path],
        hyperparameter_tuning: bool,
        validation_size: float,
        test_size: float,
        random_seed: int,
        checkpoint: str,
        sequence_length: int,
    ) -> None:
        """
        Initialize configuration.

        Parameters
        ----------
        raw_data_path : str or Path
            Path to the raw train CSV data file, with required columns 'text', 'label_*'
        raw_test_data_path : str or Path
            Path to the raw test CSV data file, with required columns 'text', 'label_*'
        train_data_path : str or Path
            Path where the training split will be saved
        val_data_path : str or Path
            Path where the validation split will be saved
        test_data_path : str or Path
            Path where the test split will be saved
        hyperparameter_tuning: bool
            Flag whether hyperparameter tuning should be performed
        validation_size: float
            Proportion of data set to be used for validation
        test_size : float
            Proportion of data set to be used for testing
        random_seed : int
            Random seed
        checkpoint : str
            Name of the pretrained model checkpoint on the Hugging Face Hub, used for tokenization
        sequence_length : int
            Maximum number of tokens per input text
        """
        self.raw_data_path = raw_data_path
        self.raw_test_data_path = raw_test_data_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.hyperparameter_tuning = hyperparameter_tuning
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.checkpoint = checkpoint
        self.sequence_length = sequence_length
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.hf_dataset: Optional[DatasetDict] = None
        self.tokenized_dataset: Optional[DatasetDict] = None

    def split_data(self) -> DataPipeline:
        """
        Split data into train, validation and test partitions, reset indices, and save to disk.

        Saves
        -----
        train_data_path, val_data_path and test_data_path as parquet files

        Returns
        -------
        DataPipeline
        """
        raw_data_exists = os.path.exists(self.raw_data_path)
        raw_test_data_exists = os.path.exists(self.raw_test_data_path)
        train_data_exists = os.path.exists(self.train_data_path)
        test_data_exists = os.path.exists(self.test_data_path)
        val_data_exists = os.path.exists(self.val_data_path)

        if train_data_exists and val_data_exists and test_data_exists:
            self.train_data = pd.read_parquet(self.train_data_path)
            self.val_data = pd.read_parquet(self.val_data_path)
            self.test_data = pd.read_parquet(self.test_data_path)
            return self

        if not raw_data_exists:
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}.")
        df, label_cols, X, y = _df_preprocess(self.raw_data_path)

        if raw_test_data_exists:
            df_test, label_cols_test, _, _ = _df_preprocess(self.raw_test_data_path)
            if label_cols != label_cols_test:
                raise ValueError("Mismatch between train/test label columns")

        if self.hyperparameter_tuning:
            if raw_test_data_exists:
                self.train_data, self.val_data = _df_split(
                    df=df, X=X, y=y, test_size=self.validation_size, random_seed=self.random_seed
                )
                self.test_data = df_test.reset_index(drop=True)
            else:
                full_train_data, self.test_data = _df_split(
                    df=df, X=X, y=y, test_size=self.test_size, random_seed=self.random_seed
                )
                self.train_data, self.val_data = _df_split(
                    df=full_train_data,
                    X=full_train_data["text"].values,
                    y=full_train_data[label_cols].values,
                    test_size=self.validation_size,
                    random_seed=self.random_seed,
                )
        else:
            if raw_test_data_exists:
                self.train_data = df.reset_index(drop=True)
                self.test_data = df_test.reset_index(drop=True)
            else:
                self.train_data, self.test_data = _df_split(
                    df=df, X=X, y=y, test_size=self.test_size, random_seed=self.random_seed
                )

        _df_save(df=self.train_data, path=self.train_data_path)
        if self.hyperparameter_tuning:
            _df_save(df=self.val_data, path=self.val_data_path)
        _df_save(df=self.test_data, path=self.test_data_path)
        return self

    def get_multi_hot_vectors(self) -> DataPipeline:
        """
        Combine label_* columns into a single 'labels' array per row (multi-hot vector).

        Returns
        -------
        DataPipeline
        """
        if self.hyperparameter_tuning and self.val_data is None:
            raise RuntimeError("Validation data not found. Run split_data() with hyperparameter_tuning=True first.")
        if self.train_data is None or self.test_data is None:
            raise RuntimeError("Train/test data not found. Run split_data() first.")

        label_cols = [col for col in self.train_data.columns if col.startswith("label_")]
        splits = ["train_data", "test_data"]
        if self.hyperparameter_tuning:
            splits.insert(1, "val_data")
        for attr in splits:
            df = getattr(self, attr).copy()
            df["labels"] = df[label_cols].values.tolist()
            df = df[["text", "labels"]]
            setattr(self, attr, df)
        return self

    def create_hf_dataset(self) -> DataPipeline:
        """
        Assemble train and test data in a Hugging Face DatasetDict.

        Returns
        -------
        DataPipeline
        """
        if self.hyperparameter_tuning and self.val_data is None:
            raise RuntimeError("Validation data not found. Run split_data() with hyperparameter_tuning=True first.")
        if self.train_data is None or self.test_data is None:
            raise RuntimeError("Train/test data not found. Run split_data() first.")
        if "labels" not in self.train_data.columns:
            raise RuntimeError("Missing 'labels' column. Run get_multi_hot_vectors() first")
        features = Features(
            {
                "text": Value(dtype="string"),
                "labels": Sequence(Value(dtype="int64")),
            }
        )
        dataset_train = Dataset.from_pandas(self.train_data, features=features)
        dataset_test = Dataset.from_pandas(self.test_data, features=features)
        dataset_dict = DatasetDict({"train": dataset_train, "test": dataset_test})
        if self.hyperparameter_tuning:
            dataset_val = Dataset.from_pandas(self.val_data, features=features)
            dataset_dict["validation"] = dataset_val
        self.hf_dataset = DatasetDict(dataset_dict)
        return self

    def tokenize_data(self) -> DataPipeline:
        """
        Tokenize text and convert multi-hot labels to float tensors.

        Returns
        -------
        DataPipeline
        """
        if self.hf_dataset is None:
            raise RuntimeError("Hugging Face DatasetDict not found. Run create_hf_dataset() first.")
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        td = self.hf_dataset.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, padding="max_length", max_length=self.sequence_length
            ),
            batched=True,
        )
        td.set_format("torch")
        self.tokenized_dataset = td.map(
            lambda batch: {"float_labels": batch["labels"].to(torch.float)}, remove_columns=["labels"]
        ).rename_column("float_labels", "labels")
        return self
