"""Tests for hpo helpers."""

import math
from types import SimpleNamespace

from optuna.trial import FixedTrial

from tlmtc.hpo import ensure_study_and_get_existing_trial_count, make_compute_objective, optuna_hp_space
from tlmtc.settings import OptunaSpaceSettings


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


def test_ensure_study_and_get_existing_trial_count_creates_or_loads_study(monkeypatch) -> None:
    """Create or load the Optuna study and return the trial count."""
    study = SimpleNamespace(
        trials=[
            SimpleNamespace(number=0),
            SimpleNamespace(number=1),
            SimpleNamespace(number=2),
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_create_study(*, study_name: str, storage: str, direction: str, load_if_exists: bool):
        calls.append(
            {
                "study_name": study_name,
                "storage": storage,
                "direction": direction,
                "load_if_exists": load_if_exists,
            }
        )
        return study

    monkeypatch.setattr(
        "tlmtc.hpo.optuna.create_study",
        fake_create_study,
    )

    assert (
        ensure_study_and_get_existing_trial_count(
            study_name="existing_study",
            storage="sqlite:///dummy.db",
            direction="maximize",
        )
        == 3
    )

    assert calls == [
        {
            "study_name": "existing_study",
            "storage": "sqlite:///dummy.db",
            "direction": "maximize",
            "load_if_exists": True,
        }
    ]


def test_ensure_study_and_get_existing_trial_count_defaults_to_maximize(monkeypatch) -> None:
    """Use maximize as the default Optuna study direction."""
    directions: list[str] = []

    def fake_create_study(*, study_name: str, storage: str, direction: str, load_if_exists: bool):
        directions.append(direction)
        return SimpleNamespace(trials=[])

    monkeypatch.setattr(
        "tlmtc.hpo.optuna.create_study",
        fake_create_study,
    )

    assert (
        ensure_study_and_get_existing_trial_count(
            study_name="new_study",
            storage="sqlite:///dummy.db",
        )
        == 0
    )

    assert directions == ["maximize"]
