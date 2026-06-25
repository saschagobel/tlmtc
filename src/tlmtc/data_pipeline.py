"""Data pipeline for preparing multi-label text data for Hugging Face training."""

from typing import Self

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers import AutoTokenizer

from tlmtc.data_contracts import (
    LABEL_PREFIX,
    TEXT_COL,
    TEXT_PAIR_COL,
    DataContractError,
    InputMode,
    validate_multilabel_frame,
    validate_split_group_disjointness,
)
from tlmtc.data_preparation import df_preprocess, df_save, df_split, tokenize_batch
from tlmtc.paths import RunPaths
from tlmtc.runtime_output import emit_progress
from tlmtc.settings import ModelSettings, SplitSettings


class DataPipeline:
    """Stateful data preparation pipeline for Hugging Face multi-label text classification.

    Attributes:
        paths: Run-specific filesystem layout for raw inputs and prepared split artifacts.
        split: Split configuration for validation/test partitioning and random seeding.
        model: Tokenization configuration, including checkpoint and sequence length.
        input_mode: Text input mode inferred from raw data or persisted splits.
        train_data: Prepared training split.
        val_data: Prepared validation split.
        test_data: Prepared test split.
        hf_dataset: Hugging Face dataset with train, validation, and test splits.
        tokenized_dataset: Tokenized Hugging Face dataset ready for PyTorch training.
    """

    def __init__(
        self,
        paths: RunPaths,
        split: SplitSettings,
        model: ModelSettings,
        labeled_data: pd.DataFrame | None,
    ) -> None:
        """Initialize the data pipeline.

        Args:
            paths: Run-specific filesystem layout for raw inputs and prepared split artifacts.
            split: Split configuration for validation/test partitioning and random seeding.
            model: Tokenization configuration, including checkpoint and sequence length.
            labeled_data: Optional in-memory labeled data. When omitted, labeled data is loaded
                from `paths.labeled_data_path`.
        """
        self.paths = paths
        self.split = split
        self.model = model
        self.labeled_data = labeled_data
        self.input_mode: InputMode | None = None
        self.train_data: pd.DataFrame | None = None
        self.val_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.hf_dataset: DatasetDict | None = None
        self.tokenized_dataset: DatasetDict | None = None

    def split_data(self) -> Self:
        """Load or create validated train, validation, and test splits.

        Existing split artifacts are reused when all three parquet files are present. Otherwise, raw CSV
        inputs are validated, stratified, and persisted for downstream Hugging Face training.

        Returns:
            Updated pipeline instance.

        Raises:
            FileNotFoundError: If a required raw input file is missing.
            DataContractError: If persisted or raw splits disagree on labels or input mode.
        """
        train_data_exists = self.paths.train_data_path.exists()
        test_data_exists = self.paths.test_data_path.exists()
        val_data_exists = self.paths.val_data_path.exists()

        emit_progress("Preparing multi-label data splits")

        if train_data_exists and val_data_exists and test_data_exists:
            self.train_data, label_cols, self.input_mode = validate_multilabel_frame(
                pd.read_parquet(self.paths.train_data_path)
            )
            self.val_data, val_label_cols, val_input_mode = validate_multilabel_frame(
                pd.read_parquet(self.paths.val_data_path)
            )
            self.test_data, test_label_cols, test_input_mode = validate_multilabel_frame(
                pd.read_parquet(self.paths.test_data_path)
            )

            if label_cols != val_label_cols or label_cols != test_label_cols:
                raise DataContractError(
                    "Label column mismatch between persisted splits: "
                    f"train has {label_cols}, validation has {val_label_cols}, test has {test_label_cols}."
                )

            if self.input_mode is not val_input_mode or self.input_mode is not test_input_mode:
                raise DataContractError(
                    "Input mode mismatch between persisted splits: "
                    f"train is '{self.input_mode.value}', "
                    f"validation is '{val_input_mode.value}', "
                    f"test is '{test_input_mode.value}'."
                )

            validate_split_group_disjointness(
                self.train_data,
                self.val_data,
                self.test_data,
            )

            return self

        if self.labeled_data is None:
            labeled_data = self.paths.labeled_data_path
            assert labeled_data is not None

            if not labeled_data.exists():
                raise FileNotFoundError(f"Labeled data not found at {labeled_data}.")
        else:
            labeled_data = self.labeled_data

        df, label_cols, text_values, label_matrix, input_mode = df_preprocess(labeled_data)
        self.input_mode = input_mode

        if self.paths.raw_test_data_path is not None:
            if not self.paths.raw_test_data_path.exists():
                raise FileNotFoundError(f"Raw test data not found at {self.paths.raw_test_data_path}.")

            df_test, label_cols_test, _, _, test_input_mode = df_preprocess(self.paths.raw_test_data_path)
            if label_cols != label_cols_test:
                raise DataContractError(
                    "Label column mismatch between labeled_data and raw_test_csv: "
                    f"labeled_data has {label_cols}, raw_test_csv has {label_cols_test}."
                )

            if self.input_mode is not test_input_mode:
                raise DataContractError(
                    "Input mode mismatch between labeled_data and raw_test_csv: "
                    f"labeled_data is '{self.input_mode.value}', but raw_test_csv is '{test_input_mode.value}'."
                )

            validate_split_group_disjointness(df, df_test)

            self.train_data, self.val_data = df_split(
                df=df,
                text_values=text_values,
                label_matrix=label_matrix,
                test_size=self.split.validation_size,
                random_seed=self.split.random_seed,
            )
            self.test_data = df_test.reset_index(drop=True)
        else:
            full_train_data, self.test_data = df_split(
                df=df,
                text_values=text_values,
                label_matrix=label_matrix,
                test_size=self.split.test_size,
                random_seed=self.split.random_seed,
            )
            self.train_data, self.val_data = df_split(
                df=full_train_data,
                text_values=full_train_data[TEXT_COL].values,
                label_matrix=full_train_data[label_cols].values,
                test_size=self.split.validation_size,
                random_seed=self.split.random_seed,
            )

        df_save(df=self.train_data, path=self.paths.train_data_path)
        df_save(df=self.val_data, path=self.paths.val_data_path)
        df_save(df=self.test_data, path=self.paths.test_data_path)
        return self

    def get_multi_hot_vectors(self) -> Self:
        """Collapse label columns into a multi-hot `labels` column.

        Returns:
            Updated pipeline instance.

        Raises:
            RuntimeError: If train, validation, or test splits have not been loaded.
        """
        if self.train_data is None or self.test_data is None or self.val_data is None:
            raise RuntimeError("Train/val/test data not found. Run split_data() first.")

        label_cols = [col for col in self.train_data.columns if col.startswith(LABEL_PREFIX)]
        input_cols = [TEXT_COL]
        if self.input_mode is InputMode.PAIRED_TEXT:
            input_cols.append(TEXT_PAIR_COL)

        for attr in ("train_data", "val_data", "test_data"):
            df = getattr(self, attr).copy()
            df["labels"] = df[label_cols].values.tolist()
            df = df[[*input_cols, "labels"]]
            setattr(self, attr, df)
        return self

    def create_hf_dataset(self) -> Self:
        """Create a Hugging Face DatasetDict from prepared splits.

        Returns:
            Updated pipeline instance.

        Raises:
            RuntimeError: If split data or multi-hot labels are missing.
        """
        if self.train_data is None or self.test_data is None or self.val_data is None:
            raise RuntimeError("Train/val/test data not found. Run split_data() first.")

        if "labels" not in self.train_data.columns:
            raise RuntimeError("Missing 'labels' column. Run get_multi_hot_vectors() first")

        feature_spec = {
            TEXT_COL: Value(dtype="string"),
        }
        if self.input_mode is InputMode.PAIRED_TEXT:
            feature_spec[TEXT_PAIR_COL] = Value(dtype="string")
        feature_spec["labels"] = Sequence(Value(dtype="int64"))
        features = Features(feature_spec)

        dataset_train = Dataset.from_pandas(self.train_data, features=features, preserve_index=False)
        dataset_val = Dataset.from_pandas(self.val_data, features=features, preserve_index=False)
        dataset_test = Dataset.from_pandas(self.test_data, features=features, preserve_index=False)

        self.hf_dataset = DatasetDict(
            {
                "train": dataset_train,
                "validation": dataset_val,
                "test": dataset_test,
            }
        )
        return self

    def tokenize_data(self) -> Self:
        """Tokenize text inputs and convert multi-hot labels to float tensors.

        Returns:
            Updated pipeline instance.

        Raises:
            RuntimeError: If the Hugging Face dataset or inferred input mode is missing.
        """
        if self.hf_dataset is None:
            raise RuntimeError("Hugging Face DatasetDict not found. Run create_hf_dataset() first.")

        if self.input_mode is None:
            raise RuntimeError("Input mode not found. Run split_data() first.")

        emit_progress("Tokenizing training inputs")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model.checkpoint,
            trust_remote_code=self.model.trust_remote_code,
        )

        td = self.hf_dataset.map(
            lambda batch: tokenize_batch(
                batch=batch,
                tokenizer=tokenizer,
                input_mode=self.input_mode,
                sequence_length=self.model.sequence_length,
            ),
            batched=True,
        )

        td.set_format("torch")
        self.tokenized_dataset = td.map(
            lambda batch: {"float_labels": batch["labels"].to(torch.float)},
            remove_columns=["labels"],
        ).rename_column("float_labels", "labels")
        return self
