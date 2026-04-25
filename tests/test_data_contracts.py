"""Tests for tabular data contracts."""

from collections.abc import Callable

import pandas as pd
import pytest

from tlmtc.data_contracts import (
    TEXT_COL,
    TEXT_PAIR_COL,
    DataContractError,
    validate_multilabel_frame,
)

FrameFactory = Callable[..., pd.DataFrame]


@pytest.fixture
def valid_frame() -> FrameFactory:
    """Return a factory for valid multilabel dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            TEXT_COL: ["first text", "second text"],
            "label_a": [1, 0],
            "label_b": [0, 1],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


class TestValidateMultilabelFrame:
    """Tests for validating raw multilabel dataframe contracts."""

    def test_validates_minimal_multilabel_frame(self, valid_frame: FrameFactory) -> None:
        df = valid_frame()

        validated, label_cols = validate_multilabel_frame(df)

        assert label_cols == ["label_a", "label_b"]
        pd.testing.assert_frame_equal(
            validated,
            pd.DataFrame(
                {
                    TEXT_COL: ["first text", "second text"],
                    "label_a": [1, 0],
                    "label_b": [0, 1],
                }
            ),
        )

    def test_keeps_optional_text_pair_column_when_present(self, valid_frame: FrameFactory) -> None:
        df = valid_frame(**{TEXT_PAIR_COL: ["first pair", "second pair"]})

        validated, label_cols = validate_multilabel_frame(df)

        assert label_cols == ["label_a", "label_b"]
        pd.testing.assert_frame_equal(
            validated,
            pd.DataFrame(
                {
                    TEXT_COL: ["first text", "second text"],
                    TEXT_PAIR_COL: ["first pair", "second pair"],
                    "label_a": [1, 0],
                    "label_b": [0, 1],
                }
            ),
        )

    def test_projects_to_text_and_label_columns(self, valid_frame: FrameFactory) -> None:
        df = valid_frame(unused_metadata=["x", "y"])

        validated, _ = validate_multilabel_frame(df)

        assert list(validated.columns) == [TEXT_COL, "label_a", "label_b"]

    def test_preserves_label_column_order(self) -> None:
        df = pd.DataFrame(
            {
                TEXT_COL: ["first text", "second text"],
                "label_b": [0, 1],
                "label_a": [1, 0],
            }
        )

        validated, label_cols = validate_multilabel_frame(df)

        assert label_cols == ["label_b", "label_a"]
        assert list(validated.columns) == [TEXT_COL, "label_b", "label_a"]

    def test_coerces_numeric_label_values_to_integer(self, valid_frame: FrameFactory) -> None:
        df = valid_frame(label_a=[1.0, 0.0], label_b=[0.0, 1.0])

        validated, _ = validate_multilabel_frame(df)

        pd.testing.assert_frame_equal(
            validated[["label_a", "label_b"]],
            pd.DataFrame({"label_a": [1, 0], "label_b": [0, 1]}, dtype="int64"),
        )

    def test_rejects_non_dataframe_input(self) -> None:
        with pytest.raises(DataContractError, match="Expected a pandas DataFrame"):
            validate_multilabel_frame("not a dataframe")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_df",
        [
            pd.DataFrame({TEXT_COL: ["first text", "second text"], "label_a": [1, 0]}),
            pd.DataFrame({"label_a": [1, 0], "label_b": [0, 1]}),
            pd.DataFrame({TEXT_COL: [], "label_a": [], "label_b": []}),
        ],
    )
    def test_rejects_structurally_invalid_frames(self, bad_df: pd.DataFrame) -> None:
        with pytest.raises(DataContractError, match="multilabel data contract"):
            validate_multilabel_frame(bad_df)

    @pytest.mark.parametrize(
        "overrides",
        [
            {TEXT_COL: ["first text", None]},
            {TEXT_COL: ["first text", "   "]},
            {"label_a": [1, None]},
            {"label_a": [1, 2]},
            {"label_a": [1, "yes"]},
        ],
    )
    def test_rejects_invalid_column_values(self, valid_frame: FrameFactory, overrides: dict[str, object]) -> None:
        with pytest.raises(DataContractError, match="multilabel data contract"):
            validate_multilabel_frame(valid_frame(**overrides))
