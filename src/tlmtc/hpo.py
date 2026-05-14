"""Optuna integration for Hugging Face hyperparameter tuning."""

import math
from typing import Any, Callable, Literal

import optuna
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from tlmtc.settings import OptunaSpaceSettings
from tlmtc.training import wrap_model_with_peft


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


def make_model_init(
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: Literal["none", "all", "lora_only"],
) -> Callable[[optuna.trial.Trial | None], PreTrainedModel]:
    """Create a Trainer-compatible model factory for hyperparameter search.

    Args:
        checkpoint: Pretrained model checkpoint identifier
        num_labels: Number of labels in the multi-label classification task.
        wrap_peft: Whether to wrap the model with PEFT/LoRA adapters.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        lora_bias: LoRA bias handling mode.

    Returns:
        model_init: A function that initializes and returns a model instance during each Optuna trial.
    """

    def model_init(trial: optuna.trial.Trial | None = None) -> PreTrainedModel:
        """Initialize a fresh sequence-classification model.

        Args:
            trial: Optional Optuna trial accepted for Trainer compatibility.

        Returns:
            Pretrained model configured for multi-label classification.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels, problem_type="multi_label_classification"
        )
        if wrap_peft:
            model = wrap_model_with_peft(
                model=model,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
            )
        return model

    return model_init


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


def get_existing_trial_count(
    study_name: str,
    storage: str,
) -> int:
    """Count trials already persisted for an Optuna study.

    Args:
        study_name: Optuna study name.
        storage: Optuna storage URL.

    Returns:
        Number of existing trials, or zero if the study does not exist.
    """
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
    except KeyError:
        return 0

    return len(study.trials)
