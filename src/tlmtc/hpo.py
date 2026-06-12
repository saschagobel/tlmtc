"""Optuna integration for Hugging Face hyperparameter tuning."""

import math
from typing import Any, Callable, Literal

import optuna

from tlmtc.settings import OptunaSpaceSettings


def optuna_hp_space(
    trial: optuna.trial.Trial,
    space: OptunaSpaceSettings,
) -> dict[str, Any]:
    """Sample Trainer hyperparameters from a resolved Optuna search space.

    Args:
        trial: Optuna trial used to sample candidate values.
        space: Resolved hyperparameter search-space settings.

    Returns:
        Sampled hyperparameters for Hugging Face Trainer.
    """
    batch_size = trial.suggest_categorical(
        "per_device_train_batch_size",
        space.batch_sizes,
    )
    lr_scale = math.sqrt(
        batch_size / space.lr_reference_batch_size,
    )
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            space.lr_low * lr_scale,
            space.lr_high * lr_scale,
            log=True,
        ),
        "per_device_train_batch_size": batch_size,
        "weight_decay": trial.suggest_float(
            "weight_decay",
            space.wd_low,
            space.wd_high,
        ),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type",
            space.schedulers,
        ),
        "num_train_epochs": trial.suggest_int(
            "num_train_epochs",
            space.epoch_low,
            space.epoch_high,
        ),
    }


def make_compute_objective(
    best_model_metric: str,
) -> Callable[[dict[str, Any]], float]:
    """Create an objective extractor for Trainer hyperparameter search.

    Args:
        best_model_metric: Model-selection metric name as configured in training settings.

    Returns:
        Callable that extracts the objective value from Trainer evaluation metrics.
    """

    def compute_objective(metrics: dict[str, Any]) -> float:
        """Extract the configured objective value from evaluation metrics.

        Args:
            metrics: Evaluation metrics returned by Hugging Face Trainer.

        Returns:
            Objective value used by Optuna.
        """
        return metrics["eval_" + best_model_metric]

    return compute_objective


def ensure_study_and_get_existing_trial_count(
    study_name: str,
    storage: str,
    direction: Literal["maximize", "minimize"] = "maximize",
) -> int:
    """Create or load an Optuna study and return its existing trial count.

    Args:
        study_name: Optuna study name.
        storage: Optuna storage URL.
        direction: Optimization direction.

    Returns:
        Number of existing trials.
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
    )
    return len(study.trials)


def get_pruner_for_world_size(
    world_size: int,
) -> optuna.pruners.BasePruner | None:
    """Return a conservative Optuna pruner for the current training topology.

    Args:
        world_size: Number of processes participating in the training run.

    Returns:
        A no-op pruner for distributed training, otherwise `None` to keep
        Optuna's default pruning behavior.
    """
    if world_size > 1:
        return optuna.pruners.NopPruner()

    return None
