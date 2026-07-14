"""Tests for prediction operations."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

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
            inference_backend="torch",
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
            inference_backend="torch",
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

    def test_loads_onnx_runtime_session_from_single_exported_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        onnx_model = tmp_path / "model" / "onnx" / "model.onnx"
        onnx_model.parent.mkdir(parents=True)
        onnx_model.touch()
        inference_session = Mock(return_value="session")
        monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=inference_session))

        result = load_prediction_model(
            model_dir=tmp_path / "model",
            inference_backend="onnx",
            checkpoint="base-checkpoint",
            num_labels=2,
            wrap_peft=False,
            trust_remote_code=False,
        )

        assert result == "session"
        inference_session.assert_called_once_with(
            str(onnx_model),
            providers=["CPUExecutionProvider"],
        )

    def test_raises_error_when_onnx_export_count_is_not_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Mock()))

        with pytest.raises(RuntimeError, match="Expected exactly one ONNX model"):
            load_prediction_model(
                model_dir=tmp_path / "model",
                inference_backend="onnx",
                checkpoint="base-checkpoint",
                num_labels=2,
                wrap_peft=False,
                trust_remote_code=False,
            )

    @pytest.mark.parametrize(
        ("trust_remote_code", "expected_default_available"),
        [(False, False), (True, True)],
    )
    def test_makes_default_rope_available_only_for_trusted_remote_code(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        trust_remote_code: bool,
        expected_default_available: bool,
    ) -> None:
        """Ensure default RoPE is available at trusted remote-code Torch loading."""
        loaded_model = object()
        original_rope_init_functions = dict(ROPE_INIT_FUNCTIONS)
        ROPE_INIT_FUNCTIONS.pop("default", None)

        def fake_from_pretrained(*_args: object, **_kwargs: object) -> object:
            assert ("default" in ROPE_INIT_FUNCTIONS) is expected_default_available
            return loaded_model

        monkeypatch.setattr("transformers.AutoModelForSequenceClassification.from_pretrained", fake_from_pretrained)

        try:
            result = load_prediction_model(
                model_dir=tmp_path / "model",
                inference_backend="torch",
                checkpoint="base-checkpoint",
                num_labels=2,
                wrap_peft=False,
                trust_remote_code=trust_remote_code,
            )
        finally:
            ROPE_INIT_FUNCTIONS.clear()
            ROPE_INIT_FUNCTIONS.update(original_rope_init_functions)

        assert result is loaded_model


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
            inference_backend="torch",
        )

        expected_logits = np.array([[0.0, -0.0], [1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
        np.testing.assert_allclose(result, 1.0 / (1.0 + np.exp(-expected_logits)), rtol=1e-6)
        assert model.training is False

    def test_gathers_distributed_probabilities_per_batch_in_input_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = Dataset.from_dict(
            {
                "input_ids": [[index] for index in range(7)],
                "attention_mask": [[1] for _ in range(7)],
            }
        )
        dataset.set_format("torch")

        gathered_batches: list[torch.Tensor] = []

        class FakeDistributedAccelerator:
            def __init__(self, *, cpu: bool) -> None:
                assert cpu is True

            def prepare_model(
                self,
                model: DeterministicPredictionModel,
                *,
                evaluation_mode: bool,
            ) -> DeterministicPredictionModel:
                assert evaluation_mode is True
                return model

            def prepare_data_loader(self, _dataloader: object) -> list[dict[str, torch.Tensor]]:
                return [
                    {
                        "input_ids": torch.tensor([[0], [1]]),
                        "attention_mask": torch.ones((2, 1), dtype=torch.int64),
                    },
                    {
                        "input_ids": torch.tensor([[4], [5]]),
                        "attention_mask": torch.ones((2, 1), dtype=torch.int64),
                    },
                ]

            def gather_for_metrics(self, probabilities: torch.Tensor) -> torch.Tensor:
                gathered_batches.append(probabilities)
                assert probabilities.shape == (2, 2)

                other_rank_indices = [2, 3] if len(gathered_batches) == 1 else [6]
                other_rank_logits = torch.tensor([[float(index), -float(index)] for index in other_rank_indices])
                return torch.cat((probabilities, torch.sigmoid(other_rank_logits)))

        monkeypatch.setattr("accelerate.Accelerator", FakeDistributedAccelerator)

        result = predict_probabilities(
            model=DeterministicPredictionModel(),
            dataset=dataset,
            batch_size=2,
            use_cpu=True,
            inference_backend="torch",
        )

        expected_logits = np.array([[index, -index] for index in range(7)], dtype=np.float32)
        np.testing.assert_allclose(result, 1.0 / (1.0 + np.exp(-expected_logits)), rtol=1e-6)
        assert len(gathered_batches) == 2

    def test_predicts_sigmoid_probabilities_with_onnx_runtime_session(self) -> None:
        dataset = Dataset.from_dict(
            {
                "input_ids": [[0], [1], [2]],
                "attention_mask": [[1], [1], [1]],
                "unused": [[10], [20], [30]],
            }
        )
        dataset.set_format("numpy")

        class DeterministicOnnxSession:
            def get_inputs(self) -> list[SimpleNamespace]:
                return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

            def run(self, output_names: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
                assert output_names is None
                assert set(inputs) == {"input_ids", "attention_mask"}
                values = inputs["input_ids"][:, 0].astype(np.float32)
                return [np.column_stack((values, -values))]

        result = predict_probabilities(
            model=DeterministicOnnxSession(),  # type: ignore[arg-type]
            dataset=dataset,
            batch_size=2,
            use_cpu=True,
            inference_backend="onnx",
        )

        expected_logits = np.array([[0.0, -0.0], [1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
        np.testing.assert_allclose(result, 1.0 / (1.0 + np.exp(-expected_logits)), rtol=1e-6)


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
