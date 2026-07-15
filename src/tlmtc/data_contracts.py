"""Tabular input contracts for multi-label text classification."""

from enum import StrEnum
from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

TEXT_COL: Final[str] = "text"
TEXT_PAIR_COL: Final[str] = "text_pair"
SPLIT_GROUP_COL: Final[str] = "split_group"
LABEL_PREFIX: Final[str] = "label_"
LABEL_REGEX: Final[str] = r"^label_.+"
MIN_LABEL_COLS: Final[int] = 2


class DataContractError(ValueError):
    """Raised when input data violates the multi-label tabular contract."""


class InputMode(StrEnum):
    """Validated text-input layout inferred from tabular columns."""

    SINGLE_TEXT = "single_text"
    PAIRED_TEXT = "paired_text"


MULTILABEL_SCHEMA = pa.DataFrameSchema(
    {
        TEXT_COL: pa.Column(
            str,
            nullable=False,
            required=True,
            checks=pa.Check(lambda series: series.str.strip().ne(""), error="must not contain blank strings"),
        ),
        TEXT_PAIR_COL: pa.Column(
            str,
            nullable=False,
            required=False,
            checks=pa.Check(lambda series: series.str.strip().ne(""), error="must not contain blank strings"),
        ),
        SPLIT_GROUP_COL: pa.Column(
            nullable=False,
            required=False,
            checks=[
                pa.Check(
                    lambda series: series.map(lambda value: isinstance(value, (str, int, bool))).all(),
                    error="must contain only scalar values (str, int, bool)",
                ),
                pa.Check(
                    lambda series: series.map(lambda value: not isinstance(value, str) or value.strip() != "").all(),
                    error="must not contain blank strings",
                ),
            ],
        ),
        LABEL_REGEX: pa.Column(
            regex=True,
            nullable=False,
            required=True,
            checks=pa.Check.isin([0, 1]),
        ),
    },
    checks=[
        pa.Check(lambda df: len(df) > 0, error="dataframe must contain at least one row"),
        pa.Check(
            lambda df: df.columns.str.startswith(LABEL_PREFIX).sum() >= MIN_LABEL_COLS,
            error=f"expected at least {MIN_LABEL_COLS} '{LABEL_PREFIX}*' columns",
        ),
        pa.Check(
            lambda df: LABEL_PREFIX not in df.columns,
            error=f"label column names must include a non-empty suffix after '{LABEL_PREFIX}'",
        ),
        pa.Check(
            lambda df: (
                df.loc[:, df.columns.str.startswith(LABEL_PREFIX)].eq(0).any(axis=0)
                & df.loc[:, df.columns.str.startswith(LABEL_PREFIX)].eq(1).any(axis=0)
            ).all(),
            error=f"each '{LABEL_PREFIX}*' column must contain positive and negative examples",
        ),
    ],
    strict=False,
    ordered=False,
    coerce=False,
)


PREDICTION_SCHEMA = pa.DataFrameSchema(
    {
        TEXT_COL: pa.Column(
            str,
            nullable=False,
            required=True,
            checks=pa.Check(lambda series: series.str.strip().ne(""), error="must not contain blank strings"),
        ),
        TEXT_PAIR_COL: pa.Column(
            str,
            nullable=False,
            required=False,
            checks=pa.Check(lambda series: series.str.strip().ne(""), error="must not contain blank strings"),
        ),
    },
    checks=[
        pa.Check(lambda df: len(df) > 0, error="dataframe must contain at least one row"),
    ],
    strict=False,
    ordered=False,
    coerce=False,
)


def validate_multilabel_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], InputMode]:
    """Validate and normalize a multi-label input DataFrame.

    Args:
        df: Input DataFrame with a required text column, optional paired-text column,
            optional split-group column, and binary label columns.

    Returns:
        Normalized DataFrame, ordered label column names, and inferred input mode.

    Raises:
        DataContractError: If the input is not a DataFrame or violates the multi-label data contract.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    try:
        validated = MULTILABEL_SCHEMA.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataContractError("Input dataframe violates the multilabel data contract.") from exc

    input_mode = InputMode.PAIRED_TEXT if TEXT_PAIR_COL in validated.columns else InputMode.SINGLE_TEXT
    label_cols = [col for col in validated.columns if col.startswith(LABEL_PREFIX)]
    validated = validated.astype({col: int for col in label_cols})
    text_cols = [TEXT_COL]
    if input_mode is InputMode.PAIRED_TEXT:
        text_cols.append(TEXT_PAIR_COL)

    split_cols = [SPLIT_GROUP_COL] if SPLIT_GROUP_COL in validated.columns else []

    return validated[[*text_cols, *split_cols, *label_cols]], label_cols, input_mode


def validate_split_group_disjointness(
    *dfs: pd.DataFrame,
) -> None:
    """Validate split-group consistency across materialized split dataframes.

    Use this for split dataframes that are loaded or supplied outside the
    internal splitting routine, such as persisted train/validation/test artifacts
    or a user-provided test dataframe.

    Args:
        *dfs: DataFrames representing split partitions.

    Raises:
        DataContractError: If only some dataframes contain the split-group column,
            or if split-group values overlap across dataframes.
    """
    has_split_group = [SPLIT_GROUP_COL in df.columns for df in dfs]

    if not any(has_split_group):
        return

    if not all(has_split_group):
        raise DataContractError(f"Column '{SPLIT_GROUP_COL}' must be present in all split dataframes or none.")

    seen: set[object] = set()
    for df in dfs:
        try:
            groups = set(df[SPLIT_GROUP_COL])
        except TypeError as exc:
            raise DataContractError(f"Column '{SPLIT_GROUP_COL}' must contain hashable scalar values.") from exc

        overlap = seen & groups
        if overlap:
            overlap_sample = sorted(repr(value) for value in overlap)[:10]
            raise DataContractError(
                f"Column '{SPLIT_GROUP_COL}' contains values that cross split boundaries: "
                f"{overlap_sample}. Rows sharing the same split-group value must stay in the same split."
            )

        seen.update(groups)


def validate_prediction_frame(
    df: pd.DataFrame,
    expected_input_mode: InputMode,
) -> pd.DataFrame:
    """Validate an unlabeled prediction input DataFrame.

    Args:
        df: Input DataFrame with a required text column and, for paired-text models,
            a required text_pair column.
        expected_input_mode: Text-input layout persisted by the training run.

    Returns:
        Validated prediction DataFrame with all original columns preserved.

    Raises:
        DataContractError: If the input is not a DataFrame, contains label columns,
            omits required text columns, or violates the prediction data contract.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    try:
        validated = PREDICTION_SCHEMA.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataContractError("Input dataframe violates the prediction data contract.") from exc

    forbidden_label_cols = [col for col in validated.columns if col.startswith(LABEL_PREFIX)]
    if forbidden_label_cols:
        raise DataContractError(f"Prediction input must be unlabeled, but found label columns: {forbidden_label_cols}.")

    if expected_input_mode is InputMode.PAIRED_TEXT and TEXT_PAIR_COL not in validated.columns:
        raise DataContractError(
            f"Prediction input is missing required column '{TEXT_PAIR_COL}' "
            "for a model trained with paired-text inputs."
        )

    return validated
