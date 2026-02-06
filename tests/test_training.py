"""Tests for training helpers."""

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, EvalPrediction

from tlmtc.training import (
    _compute_metrics,
    _get_class_weights,
    _get_scaled_lr,
    _multi_label_metrics,
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


def test_get_class_weights_uses_train_split_only_when_validation_missing(tmp_path):
    """Ensure `_get_class_weights` computes weights from train data when no validation split is provided."""
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


def test_get_class_weights_merges_train_and_validation_splits(tmp_path):
    """Ensure `_get_class_weights` concatenates train and validation data before computing weights."""
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

    metrics = _multi_label_metrics(predictions, labels)

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

    expected = _multi_label_metrics(predictions=preds, labels=labels)

    p = EvalPrediction(predictions=preds, label_ids=labels)
    result = _compute_metrics(p)

    assert result == expected


def test_wrap_peft_attaches_peft_config_to_model(base_test_model):
    """Ensure `_wrap_peft` wraps the base model with LoRA adapters exposing `peft_config`."""
    wrapped = _wrap_peft(
        model=base_test_model,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_bias="none",
    )

    assert hasattr(wrapped, "peft_config")


def test_get_scaled_lr_scales_learning_rate_for_peft_and_non_peft_modes(tmp_path):
    """Ensure _get_scaled_lr scales the base learning rate correctly for PEFT and non-PEFT configurations."""
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
