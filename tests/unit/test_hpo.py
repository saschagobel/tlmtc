"""Tests for hpo helpers."""

import math
from pathlib import Path
from types import SimpleNamespace

from optuna.trial import FixedTrial

from tlmtc.hpo import (
    BestHyperparameters,
    ensure_study_and_get_existing_trial_count,
    make_best_hyperparameters,
    make_compute_objective,
    optuna_hp_space,
    read_best_hyperparameters,
    write_best_hyperparameters,
)
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


def test_make_best_hyperparameters_converts_trainer_hpo_params() -> None:
    """Convert raw Trainer HPO params into effective tlmtc hyperparameters."""
    result = make_best_hyperparameters(
        hpo_params={
            "learning_rate": 3e-5,
            "lr_scheduler_type": "cosine",
            "per_device_train_batch_size": 8,
            "weight_decay": 0.01,
            "num_train_epochs": 4,
        },
        scale_learning_rate=False,
        checkpoint="target-checkpoint",
        proxy_checkpoint="proxy-checkpoint",
        wrap_peft=True,
        trust_remote_code=False,
    )

    assert result == BestHyperparameters(
        learning_rate=3e-5,
        lr_scheduler="cosine",
        batch_size=8,
        weight_decay=0.01,
        train_epochs=4,
    )


def test_make_best_hyperparameters_scales_learning_rate_when_requested(monkeypatch) -> None:
    """Scale the selected proxy learning rate when final fine-tuning requires it."""
    calls: list[dict[str, object]] = []

    def fake_get_scaled_lr(
        *,
        learning_rate: float,
        checkpoint: str,
        proxy_checkpoint: str,
        peft: bool,
        trust_remote_code: bool,
    ) -> float:
        calls.append(
            {
                "learning_rate": learning_rate,
                "checkpoint": checkpoint,
                "proxy_checkpoint": proxy_checkpoint,
                "peft": peft,
                "trust_remote_code": trust_remote_code,
            }
        )
        return 2e-5

    monkeypatch.setattr("tlmtc.hpo.get_scaled_lr", fake_get_scaled_lr)

    result = make_best_hyperparameters(
        hpo_params={
            "learning_rate": 4e-5,
            "lr_scheduler_type": "linear",
            "per_device_train_batch_size": 16,
            "weight_decay": 0.0,
            "num_train_epochs": 3,
        },
        scale_learning_rate=True,
        checkpoint="target-checkpoint",
        proxy_checkpoint="proxy-checkpoint",
        wrap_peft=False,
        trust_remote_code=True,
    )

    assert result.learning_rate == 2e-5
    assert result.lr_scheduler == "linear"
    assert result.batch_size == 16
    assert result.weight_decay == 0.0
    assert result.train_epochs == 3

    assert calls == [
        {
            "learning_rate": 4e-5,
            "checkpoint": "target-checkpoint",
            "proxy_checkpoint": "proxy-checkpoint",
            "peft": False,
            "trust_remote_code": True,
        }
    ]


def test_best_hyperparameters_roundtrip_json_artifact(tmp_path: Path) -> None:
    """Write and read selected HPO hyperparameters as a validated JSON artifact."""
    path = tmp_path / "best_hyperparameters.json"
    params = BestHyperparameters(
        learning_rate=2e-5,
        lr_scheduler="cosine",
        batch_size=8,
        weight_decay=0.01,
        train_epochs=5,
    )

    write_best_hyperparameters(params=params, path=path)

    assert path.is_file()
    assert read_best_hyperparameters(path) == params


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
