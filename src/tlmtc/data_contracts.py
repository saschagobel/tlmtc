"""Tabular data contracts."""

from enum import StrEnum
from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

TEXT_COL: Final[str] = "text"
TEXT_PAIR_COL: Final[str] = "text_pair"
LABEL_PREFIX: Final[str] = "label_"
LABEL_REGEX: Final[str] = r"^label_.+"
MIN_LABEL_COLS: Final[int] = 2


class DataContractError(ValueError):
    """Raised when tabular data violates the data contract."""


class InputMode(StrEnum):
    """Input mode inferred from validated input columns."""

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
        LABEL_REGEX: pa.Column(
            int,
            regex=True,
            nullable=False,
            required=True,
            coerce=True,
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
            lambda df: df.loc[:, df.columns.str.startswith(LABEL_PREFIX)].sum(axis=0).gt(0).all(),
            error=f"each '{LABEL_PREFIX}*' column must contain at least one positive example",
        ),
    ],
    strict=False,
    ordered=False,
    coerce=False,
)


def validate_multilabel_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], InputMode]:
    """Validate and normalize a multilabel dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    try:
        validated = MULTILABEL_SCHEMA.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataContractError("Input dataframe violates the multilabel data contract.") from exc

    input_mode = InputMode.PAIRED_TEXT if TEXT_PAIR_COL in validated.columns else InputMode.SINGLE_TEXT
    label_cols = [col for col in validated.columns if col.startswith(LABEL_PREFIX)]
    text_cols = [TEXT_COL]
    if input_mode is InputMode.PAIRED_TEXT:
        text_cols.append(TEXT_PAIR_COL)

    return validated[[*text_cols, *label_cols]], label_cols, input_mode
