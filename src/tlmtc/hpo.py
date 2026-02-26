"""Internal helpers for hyperparameter tuning.

Defines helpers for building Optuna search spaces and trial-scoped model setup.
"""

from copy import deepcopy
from typing import Any, Callable

import optuna
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from tlmtc.training import wrap_model_with_peft
from tlmtc.types import LoraBias, OptunaSpace, OptunaSpaceOverride


def optuna_hp_space(
    trial: optuna.trial.Trial,
    space: OptunaSpace,
) -> dict[str, Any]:
    """Define the hyperparameter search space for Optuna tuning.

    Args:
        trial: Current Optuna trial object.
        space: A fully resolved hyperparameter search space dictionary.

    Returns:
        Dictionary specifying the sampled hyperparameters and their values for the current trial.
    """
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            space["lr_low"],
            space["lr_high"],
            log=True,
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size",
            space["batch_sizes"],
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            space["wd_low"],
            space["wd_high"],
        ),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type",
            space["schedulers"],
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", space["epoch_low"], space["epoch_high"]),
    }


def make_model_init(
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: LoraBias,
) -> Callable[[optuna.trial.Trial | None], PreTrainedModel]:
    """Create a model initialization function for hyperparameter search.

    Args:
        checkpoint: Name of the pretrained model checkpoint on the Hugging Face Hub.
        num_labels: Number of labels in the multi-label classification task.
        wrap_peft: Flag whether to wrap model in parameter-efficient fine-tuning.
        lora_r: Rank of the LoRA matrices. Controls adapter capacity.
        lora_alpha: Scaling factor for the LoRA updates.
        lora_dropout: Dropout probability for LoRA layers.
        lora_bias: Whether to train bias terms, 'none', 'all', or 'lora_only'.

    Returns:
        model_init: A function that initializes and returns a model instance during each Optuna trial.
    """

    def model_init(trial: optuna.trial.Trial | None = None) -> PreTrainedModel:
        """Initialize a new model instance for the current trial.

        Args:
            trial: Current Optuna trial object.

        Returns:
            model: Pretrained model ready for fine-tuning.
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
    """Create an objective function for Optuna hyperparameter search.

    Args:
        best_model_metric: Metric to monitor for selecting the best-performing model checkpoint.

    Returns:
        compute_objective: A function that extracts and returns the target metric from the Trainer evaluation output.
    """

    def compute_objective(metrics: dict[str, Any]) -> float:
        """Extract the objective value for the current Optuna trial.

        Args:
            metrics: Dictionary of evaluation results returned by the Trainer.

        Returns:
            Value of the target metric to be optimized.
        """
        return metrics["eval_" + best_model_metric]

    return compute_objective


def resolve_optuna_space(
    wrap_peft: bool,
    space_base: OptunaSpace,
    space_peft: OptunaSpace,
    override: OptunaSpaceOverride | None,
) -> OptunaSpace:
    """...

    Args:
        wrap_peft: Flag whether to wrap model in parameter-efficient fine-tuning.
        space_base: ...
        space_peft: ...
        override: ...

    Returns:
        resolved: ...
    """
    default = space_peft if wrap_peft else space_base
    resolved = deepcopy(default)

    if override:
        resolved.update(override)
    return resolved
