"""Tests for training helpers."""

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from transformers import BertConfig, BertForSequenceClassification, EvalPrediction, TrainingArguments

from tlmtc.settings import TrainingSettings
from tlmtc.training import (
    TrainingRuntimeState,
    WeightedTrainer,
    compute_metrics,
    get_class_weights,
    get_scaled_lr,
    infer_modules_to_save,
    multi_label_metrics,
    wrap_model_with_peft,
)


@pytest.fixture
def base_test_model():
    """Provide a tiny offline transformer model for testing."""
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=2,
    )
    return BertForSequenceClassification(config)


class DummyModel(torch.nn.Module):
    """Minimal feed-forward classifier used for testing the `WeightedTrainer`."""

    def __init__(self, num_labels=3):
        """Initialize the dummy model."""
        super().__init__()
        self.num_labels = num_labels
        self.linear = torch.nn.Linear(4, num_labels)

    def forward(self, input_ids=None):
        """Compute logits for a batch of inputs."""
        logits = self.linear(input_ids.float())
        return type("Output", (), {"logits": logits})


class DummyNestedHeadNames(torch.nn.Module):
    """Model with nested classifier-like names that should not be inferred as top-level heads."""

    def __init__(self):
        """Initialize the dummy nested model."""
        super().__init__()
        self.encoder = torch.nn.ModuleDict(
            {
                "classifier": torch.nn.Linear(4, 4),
                "head": torch.nn.Linear(4, 4),
            }
        )
        self.output = torch.nn.Linear(4, 2)


def test_get_class_weights_uses_train_split(tmp_path):
    """Ensure `get_class_weights` computes weights from the training split."""
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label_a": [0, 0, 0, 1],
            "label_b": [0, 1, 1, 1],
        }
    )
    train_path = tmp_path / "train.parquet"
    df.to_parquet(train_path, index=False)

    weights = get_class_weights(train_path)

    expected = torch.tensor([2.0, 0.6666667], dtype=torch.float)

    assert torch.allclose(weights, expected, atol=1e-4)


def test_multi_label_metrics_returns_perfect_scores_for_separable_data():
    """Ensure `_multi_label_metrics` returns perfect scores when predictions separate classes perfectly."""
    predictions = np.array(
        [
            [4.0, -2.0],
            [-3.0, 2.5],
            [1.5, 0.1],
        ]
    )

    labels = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )

    metrics = multi_label_metrics(predictions, labels)

    assert metrics["f1_micro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["roc_auc_micro"] == 1.0
    assert metrics["roc_auc_macro"] == 1.0


def test_compute_metrics_forwards_eval_prediction_to_multi_label_metrics():
    """Ensure `_compute_metrics` forwards predictions and labels to `_multi_label_metrics` unchanged."""
    preds = np.array(
        [
            [4.0, -2.0],
            [-3.0, 2.5],
        ]
    )
    labels = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    expected = multi_label_metrics(predictions=preds, labels=labels)

    p = EvalPrediction(predictions=preds, label_ids=labels)
    result = compute_metrics(p)

    assert result == expected


def test_infer_modules_to_save_detects_top_level_classifier_head(base_test_model):
    """Ensure PEFT modules_to_save inference detects top-level classification heads."""
    assert infer_modules_to_save(base_test_model) == ["classifier"]


def test_infer_modules_to_save_ignores_nested_matching_module_names():
    """Ensure PEFT modules_to_save inference does not match nested encoder modules."""
    model = DummyNestedHeadNames()

    assert infer_modules_to_save(model) == []


def test_wrap_peft_attaches_peft_config_to_model(base_test_model):
    """Ensure `_wrap_peft` wraps the base model with LoRA adapters exposing `peft_config`."""
    wrapped = wrap_model_with_peft(
        model=base_test_model,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_bias="none",
    )

    assert hasattr(wrapped, "peft_config")


def test_wrap_peft_includes_inferred_modules_to_save(base_test_model):
    """Ensure PEFT wrapping includes inferred task-specific classification modules."""
    inferred_modules = infer_modules_to_save(base_test_model)

    wrapped = wrap_model_with_peft(
        model=base_test_model,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_bias="none",
    )

    peft_config = wrapped.peft_config["default"]

    assert set(inferred_modules).issubset(set(peft_config.modules_to_save))


def test_get_scaled_lr_conservatively_scales_learning_rate_for_peft_and_non_peft_modes(tmp_path):
    """Ensure get_scaled_lr applies bounded conservative proxy-to-target LR transfer scaling."""
    proxy_dir = tmp_path / "proxy"
    target_dir = tmp_path / "target"

    proxy_config = BertConfig(hidden_size=32)
    target_config = BertConfig(hidden_size=128)

    proxy_config.save_pretrained(proxy_dir)
    target_config.save_pretrained(target_dir)

    lr = 1e-4
    hidden_size_ratio = proxy_config.hidden_size / target_config.hidden_size
    expected_non_peft = lr * hidden_size_ratio**0.5
    expected_peft = lr * hidden_size_ratio**0.25

    scaled_non_peft = get_scaled_lr(
        learning_rate=lr,
        checkpoint=str(target_dir),
        proxy_checkpoint=str(proxy_dir),
        peft=False,
    )

    scaled_peft = get_scaled_lr(
        learning_rate=lr,
        checkpoint=str(target_dir),
        proxy_checkpoint=str(proxy_dir),
        peft=True,
    )

    assert scaled_non_peft == pytest.approx(expected_non_peft)
    assert scaled_peft == pytest.approx(expected_peft)
    assert scaled_peft > scaled_non_peft
    assert scaled_peft <= lr


class TestTrainingRuntimeState:
    """Test suite for the TrainingRuntimeState model."""

    def test_from_settings_copies_runtime_relevant_fields(self):
        """Ensure runtime state is derived from the runtime-relevant training settings fields."""
        training = TrainingSettings(
            batch_size=16,
            train_epochs=10,
            weight_decay=0.01,
            learning_rate=2e-5,
            lr_scheduler="linear",
            best_model_metric="roc_auc_macro",
            early_stopping_patience=5,
        )

        runtime = TrainingRuntimeState.from_settings(training)

        assert runtime.batch_size == 16
        assert runtime.train_epochs == 10
        assert runtime.weight_decay == 0.01
        assert runtime.learning_rate == 2e-5
        assert runtime.lr_scheduler == "linear"
        assert runtime.model_dump() == {
            "batch_size": 16,
            "train_epochs": 10,
            "weight_decay": 0.01,
            "learning_rate": 2e-5,
            "lr_scheduler": "linear",
        }

    def test_from_settings_returns_mutable_copy_independent_of_training_settings(self):
        """Ensure runtime state can change without mutating the resolved training settings."""
        training = TrainingSettings(
            batch_size=16,
            train_epochs=10,
            weight_decay=0.01,
            learning_rate=2e-5,
            lr_scheduler="linear",
        )

        runtime = TrainingRuntimeState.from_settings(training)
        runtime.batch_size = 32
        runtime.learning_rate = 5e-5

        assert training.batch_size == 16
        assert training.learning_rate == 2e-5
        assert runtime.batch_size == 32
        assert runtime.learning_rate == 5e-5

    def test_validate_assignment_rejects_invalid_runtime_updates(self):
        """Ensure runtime state validates invalid assignment updates."""
        training = TrainingSettings(
            batch_size=16,
            train_epochs=10,
            weight_decay=0.01,
            learning_rate=2e-5,
            lr_scheduler="linear",
        )

        runtime = TrainingRuntimeState.from_settings(training)

        with pytest.raises(ValidationError):
            runtime.learning_rate = 0.0

        with pytest.raises(ValidationError):
            runtime.weight_decay = -0.1

        with pytest.raises(ValidationError):
            runtime.batch_size = 0

    def test_direct_init_validates_required_fields(self):
        """Ensure direct initialization enforces required constrained runtime fields."""
        with pytest.raises(ValidationError):
            TrainingRuntimeState(
                batch_size=16,
                train_epochs=10,
                weight_decay=0.01,
                learning_rate=2e-5,
                # missing lr_scheduler
            )


class TestWeightedTrainer:
    """Test suite for the WeightedTrainer class."""

    def test_loss_respects_class_weights(self, tmp_path):
        """Ensure that applying class weights increases the computed BCE loss."""
        model = DummyModel(num_labels=3)

        inputs = {
            "labels": torch.tensor([[1, 0, 1], [0, 1, 0]]).float(),
            "input_ids": torch.zeros((2, 4)),
        }

        args = TrainingArguments(
            output_dir=str(tmp_path / "trainer_output"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            report_to="none",
        )

        unweighted = WeightedTrainer(model=model, args=args, class_weights=None)
        loss_unweighted = unweighted.compute_loss(model, dict(inputs))

        cw = torch.tensor([5.0, 5.0, 5.0])
        weighted = WeightedTrainer(model=model, args=args, class_weights=cw)
        loss_weighted = weighted.compute_loss(model, dict(inputs))

        assert loss_weighted > loss_unweighted

    def test_returns_loss_and_outputs_tuple(self, tmp_path):
        """Test that compute_loss returns (loss, outputs) when return_outputs is True."""
        model = DummyModel(num_labels=3)
        args = TrainingArguments(output_dir=str(tmp_path / "trainer_output"), report_to="none")

        trainer = WeightedTrainer(model=model, args=args)
        inputs = {"labels": torch.zeros((1, 3)), "input_ids": torch.zeros((1, 4))}

        loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

        assert isinstance(loss, torch.Tensor)
        assert hasattr(outputs, "logits")

    def test_reads_num_labels_from_model_module(self, tmp_path):
        """Ensure that compute_loss reads num_labels correctly from model.module when present."""
        model = torch.nn.DataParallel(DummyModel(num_labels=4))
        args = TrainingArguments(output_dir=str(tmp_path / "trainer_output"), report_to="none")

        trainer = WeightedTrainer(model=model, args=args)
        inputs = {"labels": torch.zeros((1, 4)), "input_ids": torch.zeros((1, 4))}

        loss = trainer.compute_loss(model, inputs)
        assert isinstance(loss, torch.Tensor)
