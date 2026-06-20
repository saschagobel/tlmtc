"""Optuna integration for Hugging Face hyperparameter tuning."""

import math
from pathlib import Path
from typing import Any, Callable, Literal

import optuna
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from tlmtc.settings import OptunaSpaceSettings
from tlmtc.training import get_scaled_lr


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


class BestHyperparameters(BaseModel):
    """Effective hyperparameters selected by HPO for final fine-tuning.

    Attributes:
        learning_rate: Effective learning rate used for final fine-tuning. If
            learning-rate scaling is enabled, this stores the scaled target-checkpoint
            learning rate rather than the raw proxy-HPO learning rate.
        lr_scheduler: Learning-rate scheduler name used for final fine-tuning.
        batch_size: Per-device train and evaluation batch size used for final fine-tuning.
        weight_decay: Weight decay used for final fine-tuning.
        train_epochs: Number of training epochs used for final fine-tuning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_rate: float = Field(..., gt=0.0)
    lr_scheduler: str
    batch_size: PositiveInt
    weight_decay: float = Field(..., ge=0.0)
    train_epochs: PositiveInt


def make_best_hyperparameters(
    hpo_params: dict[str, Any],
    *,
    scale_learning_rate: bool,
    checkpoint: str,
    proxy_checkpoint: str,
    wrap_peft: bool,
    trust_remote_code: bool,
) -> BestHyperparameters:
    """Convert raw Trainer/Optuna hyperparameters into effective training parameters.

    Args:
        hpo_params: Best hyperparameters returned by Hugging Face Trainer HPO.
            Expected keys are `learning_rate`, `lr_scheduler_type`,
            `per_device_train_batch_size`, `weight_decay`, and `num_train_epochs`.
        scale_learning_rate: Whether to scale the proxy-selected learning rate for
            final fine-tuning on the target checkpoint.
        checkpoint: Target checkpoint used for final fine-tuning.
        proxy_checkpoint: Proxy checkpoint used during hyperparameter optimization.
        wrap_peft: Whether final fine-tuning uses PEFT/LoRA wrapping.
        trust_remote_code: Whether Hugging Face config loading may execute custom remote code.

    Returns:
        Validated effective hyperparameters for final fine-tuning.
    """
    learning_rate = hpo_params["learning_rate"]

    if scale_learning_rate:
        learning_rate = get_scaled_lr(
            learning_rate=learning_rate,
            checkpoint=checkpoint,
            proxy_checkpoint=proxy_checkpoint,
            peft=wrap_peft,
            trust_remote_code=trust_remote_code,
        )

    return BestHyperparameters(
        learning_rate=learning_rate,
        lr_scheduler=hpo_params["lr_scheduler_type"],
        batch_size=hpo_params["per_device_train_batch_size"],
        weight_decay=hpo_params["weight_decay"],
        train_epochs=hpo_params["num_train_epochs"],
    )


def write_best_hyperparameters(
    params: BestHyperparameters,
    path: Path,
) -> None:
    """Write selected HPO hyperparameters as a JSON artifact.

    Args:
        params: Effective selected hyperparameters to persist.
        path: Destination JSON path.
    """
    path.write_text(
        params.model_dump_json(indent=4),
        encoding="utf-8",
    )


def read_best_hyperparameters(
    path: Path,
) -> BestHyperparameters:
    """Read selected HPO hyperparameters from a JSON artifact.

    Args:
        path: Source JSON path.

    Returns:
        Validated effective hyperparameters selected by HPO.
    """
    return BestHyperparameters.model_validate_json(path.read_text(encoding="utf-8"))
