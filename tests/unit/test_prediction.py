"""Tests for prediction operations."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset

from tlmtc.data_contracts import TEXT_COL, DataContractError
from tlmtc.prediction import (
    apply_thresholds,
    load_prediction_model,
    make_prediction_frame,
    predict_probabilities,
)


def test_prediction_module_does_not_expose_torch_prediction_imports() -> None:
    """Ensure the Torch prediction stack stays out of prediction module globals."""
    import tlmtc.prediction as prediction_mod

    assert not hasattr(prediction_mod, "Accelerator")
    assert not hasattr(prediction_mod, "AutoModelForSequenceClassification")
    assert not hasattr(prediction_mod, "DataLoader")
    assert not hasattr(prediction_mod, "PeftModel")
    assert not hasattr(prediction_mod, "torch")


class DeterministicPredictionModel(torch.nn.Module):
    """Tiny model that returns deterministic logits from input IDs."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        values = input_ids[:, 0].float()
        return SimpleNamespace(logits=torch.column_stack((values, -values)))


class TestLoadPredictionModel:
    """Test suite for loading prediction models."""

    def test_loads_full_model_from_artifact_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loaded_model = object()
        from_pretrained = Mock(return_value=loaded_model)
        monkeypatch.setattr("transformers.AutoModelForSequenceClassification.from_pretrained", from_pretrained)

        result = load_prediction_model(
            model_dir=tmp_path / "model",
            checkpoint="base-checkpoint",
            num_labels=2,
            wrap_peft=False,
            trust_remote_code=False,
        )

        assert result is loaded_model
        from_pretrained.assert_called_once_with(
            tmp_path / "model",
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=False,
        )

    def test_loads_peft_model_from_base_checkpoint_and_adapter_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base_model = object()
        loaded_model = object()
        auto_from_pretrained = Mock(return_value=base_model)
        peft_from_pretrained = Mock(return_value=loaded_model)

        monkeypatch.setattr(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            auto_from_pretrained,
        )
        monkeypatch.setattr("peft.PeftModel.from_pretrained", peft_from_pretrained)

        result = load_prediction_model(
            model_dir=tmp_path / "adapter",
            checkpoint="base-checkpoint",
            num_labels=3,
            wrap_peft=True,
            trust_remote_code=True,
        )

        assert result is loaded_model
        auto_from_pretrained.assert_called_once_with(
            "base-checkpoint",
            num_labels=3,
            problem_type="multi_label_classification",
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        peft_from_pretrained.assert_called_once_with(
            base_model,
            tmp_path / "adapter",
            low_cpu_mem_usage=True,
        )


class TestPredictProbabilities:
    """Test suite for probability prediction."""

    def test_predicts_sigmoid_probabilities(self) -> None:
        dataset = Dataset.from_dict(
            {
                "input_ids": [[0], [1], [2]],
                "attention_mask": [[1], [1], [1]],
            }
        )
        dataset.set_format("torch")

        model = DeterministicPredictionModel()

        result = predict_probabilities(
            model=model,  # type: ignore[arg-type]
            dataset=dataset,
            batch_size=2,
            use_cpu=True,
        )

        expected_logits = np.array([[0.0, -0.0], [1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
        np.testing.assert_allclose(result, 1.0 / (1.0 + np.exp(-expected_logits)), rtol=1e-6)
        assert model.training is False


class TestApplyThresholds:
    """Test suite for applying prediction thresholds."""

    @pytest.mark.parametrize(
        ("thresholds", "expected"),
        [
            ([0.5], np.array([[0, 1], [1, 1]])),
            ([0.3, 0.6], np.array([[0, 0], [1, 1]])),
        ],
    )
    def test_applies_global_or_label_specific_thresholds(
        self,
        thresholds: list[float],
        expected: np.ndarray,
    ) -> None:
        probabilities = np.array([[0.20, 0.50], [0.80, 0.70]])

        result = apply_thresholds(probabilities=probabilities, thresholds=thresholds)

        np.testing.assert_array_equal(result, expected)


class TestMakePredictionFrame:
    """Test suite for creating prediction output frames."""

    def test_appends_label_values_and_resets_index(self) -> None:
        input_df = pd.DataFrame(
            {
                "external_id": ["doc-b", "doc-a"],
                TEXT_COL: ["first text", "second text"],
            },
            index=[10, 20],
        )
        values = np.array([[0.10, 0.90], [0.80, 0.20]])

        result = make_prediction_frame(
            input_df=input_df,
            values=values,
            label_names=["risk", "routing"],
        )

        expected = pd.DataFrame(
            {
                "external_id": ["doc-b", "doc-a"],
                TEXT_COL: ["first text", "second text"],
                "risk": [0.10, 0.80],
                "routing": [0.90, 0.20],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_raises_error_when_output_row_count_does_not_match_input(self) -> None:
        with pytest.raises(ValueError, match="Output row count does not match prediction input row count"):
            make_prediction_frame(
                input_df=pd.DataFrame({TEXT_COL: ["first text", "second text"]}),
                values=np.array([[0.10, 0.90]]),
                label_names=["risk", "routing"],
            )

    def test_raises_error_when_label_columns_conflict_with_input_columns(self) -> None:
        with pytest.raises(DataContractError, match="Prediction output label columns conflict"):
            make_prediction_frame(
                input_df=pd.DataFrame({TEXT_COL: ["first text"], "risk": ["existing value"]}),
                values=np.array([[0.10, 0.90]]),
                label_names=["risk", "routing"],
            )
