"""Tests for tabular data contracts."""

from collections.abc import Callable

import pandas as pd
import pytest

from tlmtc.data_contracts import (
    SPLIT_GROUP_COL,
    TEXT_COL,
    TEXT_PAIR_COL,
    DataContractError,
    InputMode,
    validate_multilabel_frame,
    validate_prediction_frame,
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

        validated, label_cols, input_mode = validate_multilabel_frame(df)

        assert label_cols == ["label_a", "label_b"]
        assert input_mode is InputMode.SINGLE_TEXT
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

        validated, label_cols, input_mode = validate_multilabel_frame(df)

        assert label_cols == ["label_a", "label_b"]
        assert input_mode is InputMode.PAIRED_TEXT
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

    def test_keeps_optional_split_group_column_when_present(self, valid_frame: FrameFactory) -> None:
        df = valid_frame(**{SPLIT_GROUP_COL: ["group_a", "group_b"]})

        validated, label_cols, input_mode = validate_multilabel_frame(df)

        assert label_cols == ["label_a", "label_b"]
        assert input_mode is InputMode.SINGLE_TEXT
        pd.testing.assert_frame_equal(
            validated,
            pd.DataFrame(
                {
                    TEXT_COL: ["first text", "second text"],
                    SPLIT_GROUP_COL: ["group_a", "group_b"],
                    "label_a": [1, 0],
                    "label_b": [0, 1],
                }
            ),
        )

    def test_projects_to_text_and_label_columns(self, valid_frame: FrameFactory) -> None:
        df = valid_frame(unused_metadata=["x", "y"])

        validated, _, input_mode = validate_multilabel_frame(df)

        assert input_mode is InputMode.SINGLE_TEXT
        assert list(validated.columns) == [TEXT_COL, "label_a", "label_b"]

    def test_preserves_label_column_order(self) -> None:
        df = pd.DataFrame(
            {
                TEXT_COL: ["first text", "second text"],
                "label_b": [0, 1],
                "label_a": [1, 0],
            }
        )

        validated, label_cols, input_mode = validate_multilabel_frame(df)

        assert label_cols == ["label_b", "label_a"]
        assert input_mode is InputMode.SINGLE_TEXT
        assert list(validated.columns) == [TEXT_COL, "label_b", "label_a"]

    def test_coerces_numeric_label_values_to_integer(self, valid_frame: FrameFactory) -> None:
        df = valid_frame(label_a=[1.0, 0.0], label_b=[0.0, 1.0])

        validated, _, input_mode = validate_multilabel_frame(df)

        assert input_mode is InputMode.SINGLE_TEXT
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
            {SPLIT_GROUP_COL: ["group_a", None]},
            {SPLIT_GROUP_COL: ["group_a", "   "]},
        ],
    )
    def test_rejects_invalid_column_values(self, valid_frame: FrameFactory, overrides: dict[str, object]) -> None:
        with pytest.raises(DataContractError, match="multilabel data contract"):
            validate_multilabel_frame(valid_frame(**overrides))

    def test_rejects_label_columns_without_positive_examples(self) -> None:
        """Ensure every label column has at least one positive example."""
        df = pd.DataFrame(
            {
                TEXT_COL: ["first text", "second text", "third text"],
                "label_a": [1, 0, 0],
                "label_b": [0, 0, 0],
            }
        )

        with pytest.raises(DataContractError, match="multilabel data contract"):
            validate_multilabel_frame(df)


class TestValidatePredictionFrame:
    """Tests for validating unlabeled prediction dataframe contracts."""

    def test_validates_single_text_prediction_frame_and_preserves_extra_columns(self) -> None:
        df = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                TEXT_COL: ["first text", "second text"],
                "source": ["crm", "email"],
            }
        )

        validated = validate_prediction_frame(
            df,
            expected_input_mode=InputMode.SINGLE_TEXT,
        )

        pd.testing.assert_frame_equal(validated, df)

    def test_validates_paired_text_prediction_frame_and_preserves_extra_columns(self) -> None:
        df = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                TEXT_COL: ["first text", "second text"],
                TEXT_PAIR_COL: ["first pair", "second pair"],
                "source": ["crm", "email"],
            }
        )

        validated = validate_prediction_frame(
            df,
            expected_input_mode=InputMode.PAIRED_TEXT,
        )

        pd.testing.assert_frame_equal(validated, df)

    def test_allows_text_pair_for_single_text_prediction_frame(self) -> None:
        df = pd.DataFrame(
            {
                TEXT_COL: ["first text", "second text"],
                TEXT_PAIR_COL: ["first pair", "second pair"],
            }
        )

        validated = validate_prediction_frame(
            df,
            expected_input_mode=InputMode.SINGLE_TEXT,
        )

        pd.testing.assert_frame_equal(validated, df)

    def test_rejects_non_dataframe_prediction_input(self) -> None:
        with pytest.raises(DataContractError, match="Expected a pandas DataFrame"):
            validate_prediction_frame(
                "not a dataframe",  # type: ignore[arg-type]
                expected_input_mode=InputMode.SINGLE_TEXT,
            )

    @pytest.mark.parametrize(
        "bad_df",
        [
            pd.DataFrame({TEXT_COL: []}),
            pd.DataFrame({"body": ["first text", "second text"]}),
        ],
    )
    def test_rejects_structurally_invalid_prediction_frames(self, bad_df: pd.DataFrame) -> None:
        with pytest.raises(DataContractError, match="prediction data contract"):
            validate_prediction_frame(
                bad_df,
                expected_input_mode=InputMode.SINGLE_TEXT,
            )

    @pytest.mark.parametrize(
        "bad_text",
        ["", "   ", None],
    )
    def test_rejects_missing_or_blank_prediction_text(self, bad_text: str | None) -> None:
        df = pd.DataFrame({TEXT_COL: ["valid text", bad_text]})

        with pytest.raises(DataContractError, match="prediction data contract"):
            validate_prediction_frame(
                df,
                expected_input_mode=InputMode.SINGLE_TEXT,
            )

    def test_rejects_missing_text_pair_when_paired_text_expected(self) -> None:
        df = pd.DataFrame({TEXT_COL: ["first text", "second text"]})

        with pytest.raises(DataContractError, match=f"missing required column '{TEXT_PAIR_COL}'"):
            validate_prediction_frame(
                df,
                expected_input_mode=InputMode.PAIRED_TEXT,
            )

    @pytest.mark.parametrize(
        "bad_text_pair",
        ["", "   ", None],
    )
    def test_rejects_missing_or_blank_prediction_text_pair_when_present(
        self,
        bad_text_pair: str | None,
    ) -> None:
        df = pd.DataFrame(
            {
                TEXT_COL: ["first text", "second text"],
                TEXT_PAIR_COL: ["valid pair", bad_text_pair],
            }
        )

        with pytest.raises(DataContractError, match="prediction data contract"):
            validate_prediction_frame(
                df,
                expected_input_mode=InputMode.PAIRED_TEXT,
            )

    def test_rejects_label_columns_in_prediction_frame(self) -> None:
        df = pd.DataFrame(
            {
                TEXT_COL: ["first text", "second text"],
                "label_a": [1, 0],
            }
        )

        with pytest.raises(DataContractError, match="Prediction input must be unlabeled"):
            validate_prediction_frame(
                df,
                expected_input_mode=InputMode.SINGLE_TEXT,
            )
