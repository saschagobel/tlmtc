"""Tests for hpo helpers."""

import math
from types import SimpleNamespace

import pytest
from optuna.trial import FixedTrial
from transformers import BertConfig, BertForSequenceClassification

from tlmtc.hpo import get_existing_trial_count, make_compute_objective, make_model_init, optuna_hp_space
from tlmtc.settings import OptunaSpaceSettings


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
        "tlmtc.hpo.AutoModelForSequenceClassification.from_pretrained",
        fake_from_pretrained,
    )

    return base_test_model


@pytest.mark.parametrize("wrap_peft", [False, True])
def test_make_model_init_creates_base_or_peft_wrapped_models(patched_model_loader, wrap_peft):
    """Ensure _make_model_init returns a factory that loads the base model and conditionally applies PEFT."""
    model_init = make_model_init(
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
    compute_obj = make_compute_objective("f1_macro")

    assert callable(compute_obj)

    metrics = {"eval_f1_macro": 0.73}
    assert compute_obj(metrics) == 0.73


def test_optuna_hp_space_returns_expected_dict() -> None:
    """Ensure `optuna_hp_space` returns sampled hyperparameters with batch-size-scaled LR."""
    space = OptunaSpaceSettings(
        lr_low=1e-5,
        lr_high=1e-4,
        batch_sizes=[8, 16],
        wd_low=0.0,
        wd_high=0.1,
        schedulers=["linear", "cosine"],
        epoch_low=2,
        epoch_high=5,
        lr_reference_batch_size=16,
    )

    scaled_learning_rate = 5e-5 * math.sqrt(8 / 16)
    trial = FixedTrial(
        {
            "learning_rate": scaled_learning_rate,
            "per_device_train_batch_size": 8,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "num_train_epochs": 3,
        }
    )

    result = optuna_hp_space(trial, space)

    assert result == {
        "learning_rate": scaled_learning_rate,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 3,
    }


def test_optuna_hp_space_keeps_learning_rate_unchanged_at_reference_batch_size() -> None:
    """At the reference batch size, the sampled effective LR should be unchanged."""
    space = OptunaSpaceSettings(
        lr_low=1e-5,
        lr_high=1e-4,
        batch_sizes=[8, 16],
        wd_low=0.0,
        wd_high=0.1,
        schedulers=["linear", "cosine"],
        epoch_low=2,
        epoch_high=5,
        lr_reference_batch_size=16,
    )

    trial = FixedTrial(
        {
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 16,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "num_train_epochs": 3,
        }
    )

    result = optuna_hp_space(trial, space)

    assert result["learning_rate"] == 5e-5


def test_get_existing_trial_count_returns_zero_when_study_does_not_exist(monkeypatch) -> None:
    """Return zero when the requested Optuna study is not yet persisted."""

    def fake_load_study(*, study_name: str, storage: str):
        raise KeyError("Record does not exist.")

    monkeypatch.setattr(
        "tlmtc.hpo.optuna.load_study",
        fake_load_study,
    )

    assert (
        get_existing_trial_count(
            study_name="missing_study",
            storage="sqlite:///dummy.db",
        )
        == 0
    )


def test_get_existing_trial_count_returns_number_of_persisted_trials(monkeypatch) -> None:
    """Return the number of trials already stored for an existing Optuna study."""
    study = SimpleNamespace(
        trials=[
            SimpleNamespace(number=0),
            SimpleNamespace(number=1),
            SimpleNamespace(number=2),
        ]
    )

    def fake_load_study(*, study_name: str, storage: str):
        return study

    monkeypatch.setattr(
        "tlmtc.hpo.optuna.load_study",
        fake_load_study,
    )

    assert (
        get_existing_trial_count(
            study_name="existing_study",
            storage="sqlite:///dummy.db",
        )
        == 3
    )
