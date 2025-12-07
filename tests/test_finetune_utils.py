"""Tests for fine-tuning utility functions."""

import numpy as np
import pandas as pd
import pytest
import torch
from optuna.trial import FixedTrial
from transformers import BertConfig, BertForSequenceClassification, EvalPrediction

from tlmtc.utils import (
    _compute_metrics,
    _find_optimal_threshold,
    _get_class_weights,
    _get_scaled_lr,
    _make_compute_objective,
    _make_model_init,
    _multi_label_metrics,
    _optuna_hp_space,
    _wrap_peft,
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


@pytest.fixture
def patched_model_loader(base_test_model, monkeypatch):
    """Monkeypatch AutoModelForSequenceClassification.from_pretrained."""

    def fake_from_pretrained(*_args, **_kwargs):
        return base_test_model

    monkeypatch.setattr(
        "src.tlmtc.utils.AutoModelForSequenceClassification.from_pretrained",
        fake_from_pretrained,
    )

    return base_test_model


def test_wrap_peft_returns_peft_model(base_test_model):
    """Ensure that `_wrap_peft` returns a PEFT-wrapped model exposing `peft_config`."""
    wrapped = _wrap_peft(
        model=base_test_model,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_bias="none",
    )

    assert hasattr(wrapped, "peft_config")


@pytest.mark.parametrize("wrap_peft", [False, True])
def test_make_model_init(patched_model_loader, wrap_peft):
    """Test that model_init returns a valid model and wraps it with PEFT when enabled."""
    model_init = _make_model_init(
        checkpoint="dummy",
        num_labels=2,
        wrap_peft=wrap_peft,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_bias="none",
    )

    assert callable(model_init)

    model = model_init(None)

    if wrap_peft:
        assert hasattr(model, "peft_config")
    else:
        assert model is patched_model_loader


def test_make_compute_objective_returns_callable() -> None:
    """Ensure `_make_compute_objective` returns a callable suitable for Optuna."""
    compute_obj = _make_compute_objective("f1_macro")

    assert callable(compute_obj)

    metrics = {"eval_f1_macro": 0.73}
    assert compute_obj(metrics) == 0.73


def test_optuna_hp_space_returns_expected_dict():
    """Ensure `_optuna_hp_space` returns the expected sampled hyperparameters."""
    space = {
        "lr_low": 1e-5,
        "lr_high": 1e-4,
        "batch_sizes": [8, 16],
        "wd_low": 0.0,
        "wd_high": 0.1,
        "schedulers": ["linear", "cosine"],
        "epoch_low": 2,
        "epoch_high": 5,
    }

    trial = FixedTrial(
        {
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 8,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "num_train_epochs": 3,
        }
    )

    result = _optuna_hp_space(trial, space)

    assert result == {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 3,
    }


def test_get_scaled_lr_peft_and_non_peft(tmp_path):
    """Ensure correct scaled LR for PEFT and non-PEFT modes."""
    proxy_dir = tmp_path / "proxy"
    target_dir = tmp_path / "target"

    proxy_config = BertConfig(hidden_size=32)
    target_config = BertConfig(hidden_size=128)

    proxy_config.save_pretrained(proxy_dir)
    target_config.save_pretrained(target_dir)

    lr = 1e-4
    expected_non_peft = lr * (proxy_config.hidden_size / target_config.hidden_size)
    expected_peft = lr * (target_config.hidden_size / proxy_config.hidden_size) ** 0.5

    scaled_non_peft = _get_scaled_lr(
        learning_rate=lr,
        checkpoint=str(target_dir),
        proxy_checkpoint=str(proxy_dir),
        peft=False,
    )

    scaled_peft = _get_scaled_lr(
        learning_rate=lr,
        checkpoint=str(target_dir),
        proxy_checkpoint=str(proxy_dir),
        peft=True,
    )

    assert pytest.approx(scaled_non_peft) == expected_non_peft
    assert pytest.approx(scaled_peft) == expected_peft


def test_get_class_weights_train_only(tmp_path):
    """Ensure class weights are correctly computed from train data alone."""
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label_a": [0, 0, 0, 1],
            "label_b": [0, 1, 1, 1],
        }
    )
    train_path = tmp_path / "train.parquet"
    df.to_parquet(train_path, index=False)

    weights = _get_class_weights(train_path)

    expected = torch.tensor([2.0, 0.6666667], dtype=torch.float)

    assert torch.allclose(weights, expected, atol=1e-4)


def test_get_class_weights_with_validation(tmp_path):
    """Ensure class weights are correctly computed from the concatenated train and validation data."""
    train_df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "label_x": [0, 1],
        }
    )
    val_df = pd.DataFrame(
        {
            "text": ["c", "d", "e"],
            "label_x": [1, 1, 0],
        }
    )

    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    weights = _get_class_weights(train_path, val_path)

    expected = torch.tensor([0.8333333], dtype=torch.float)
    assert torch.allclose(weights, expected, atol=1e-4)


@pytest.mark.parametrize("metric", ["f1_micro", "f1_macro"])
def test_find_optimal_threshold_global(metric):
    """Ensure global threshold is correctly selected using f1_micro and f1_macro."""
    y_true = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y_prob = np.array(
        [
            [0.2, 0.8],
            [0.7, 0.3],
            [0.9, 0.9],
        ]
    )

    threshold = _find_optimal_threshold(
        y_true=y_true,
        y_prob=y_prob,
        best_threshold_metric=metric,
        threshold_type="global",
    )

    assert isinstance(threshold, np.ndarray)
    assert threshold.shape == (1,)
    assert 0.3 <= threshold[0] <= 0.32


def test_find_optimal_threshold_label_specific():
    """Ensure label-specific thresholds are computed independently."""
    y_true = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
        ]
    )
    y_prob = np.array(
        [
            [0.4, 0.2],
            [0.8, 0.9],
            [0.1, 0.8],
        ]
    )

    thresholds = _find_optimal_threshold(
        y_true=y_true,
        y_prob=y_prob,
        best_threshold_metric="f1_macro",
        threshold_type="label",
    )

    assert thresholds.shape == (2,)
    assert 0.10 <= thresholds[0] <= 0.12
    assert 0.19 <= thresholds[1] <= 0.21


def test_find_optimal_threshold_invalid_metric():
    """Ensure invalid best_threshold_metric raises ValueError."""
    y_true = np.array([[1], [0]])
    y_prob = np.array([[0.8], [0.2]])

    with pytest.raises(ValueError):
        _find_optimal_threshold(
            y_true=y_true,
            y_prob=y_prob,
            best_threshold_metric="not_a_metric",
            threshold_type="global",
        )


def test_find_optimal_threshold_invalid_threshold_type():
    """Ensure unsupported threshold_type raises ValueError."""
    y_true = np.array([[1], [0]])
    y_prob = np.array([[0.8], [0.2]])

    with pytest.raises(ValueError):
        _find_optimal_threshold(
            y_true=y_true,
            y_prob=y_prob,
            best_threshold_metric="f1_micro",
            threshold_type="wrong",
        )


def test_multi_label_metrics_correctness():
    """Ensure metrics are computed correctly."""
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

    metrics = _multi_label_metrics(predictions, labels)

    assert metrics["f1_micro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["roc_auc_micro"] == 1.0
    assert metrics["roc_auc_macro"] == 1.0


def test_compute_metrics_numeric_forwarding():
    """Ensure _compute_metrics forwards predictions/labels correctly to _multi_label_metrics."""
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

    expected = _multi_label_metrics(predictions=preds, labels=labels)

    p = EvalPrediction(predictions=preds, label_ids=labels)
    result = _compute_metrics(p)

    assert result == expected
