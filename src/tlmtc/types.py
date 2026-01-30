"""Type definitions.

Defines shared type aliases and structured typing helpers.
"""

from __future__ import annotations

from typing import Literal, TypedDict

Threshold = Literal["global", "label"]
BestModelMetric = Literal["f1_micro", "f1_macro", "roc_auc_micro", "roc_auc_macro"]
BestThresholdMetric = Literal["f1_micro", "f1_macro", "roc_auc_micro", "roc_auc_macro"]
LoraBias = Literal["none", "all", "lora_only"]

class OptunaSpace(TypedDict):
    """Optuna hyperparameter search space specification.

    Attributes:
        lr_low: Lower bound for the learning rate.
        lr_high: Upper bound for the learning rate.
        batch_sizes: Candidate batch sizes to consider.
        wd_low: Lower bound for weight decay.
        wd_high: Upper bound for weight decay.
        schedulers: Candidate learning rate schedulers.
        epoch_low: Lower bound for the number of training epochs.
        epoch_high: Upper bound for the number of training epochs.
    """
    lr_low: float
    lr_high: float
    batch_sizes: list[int]
    wd_low: float
    wd_high: float
    schedulers: list[str]
    epoch_low: int
    epoch_high: int

class OptunaSpaceOverride(OptunaSpace, total=False):
    """Partial override for the Optuna hyperparameter search space.

    See `OptunaSpace` for the full list of supported keys.
    Any keys not provided fall back to the package defaults.
    """
