"""Settings bundles.

Groups runner/pipeline configuration into coherent units.
"""

from dataclasses import dataclass

from tlmtc.types import (
    BestModelMetric,
    BestThresholdMetric,
    LoraBias,
    OptunaSpace,
    Threshold,
)


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """..."""

    target_name: str
    proxy_checkpoint: str
    checkpoint: str
    sequence_length: int


@dataclass(frozen=True, slots=True)
class SplitSettings:
    """..."""

    validation_size: float
    test_size: float
    random_seed: int


@dataclass(frozen=True, slots=True)
class WorkflowSettings:
    """..."""
    hyperparameter_tuning: bool
    threshold_optimization: bool
    transfer_learning: bool
    scale_learning_rate: bool
    wrap_peft: bool


@dataclass(slots=True)
class TrainingSettings:
    """..."""
    batch_size: int
    train_epochs: int
    weight_decay: float
    learning_rate: float
    lr_scheduler: str
    best_model_metric: BestModelMetric
    early_stopping_patience: int


@dataclass(frozen=True, slots=True)
class ThresholdSettings:
    """..."""
    threshold_type: Threshold
    best_threshold_metric: BestThresholdMetric


@dataclass(frozen=True, slots=True)
class HpoSettings:
    """..."""
    tuning_trials: int
    optuna_space: OptunaSpace


@dataclass(frozen=True, slots=True)
class PeftSettings:
    """..."""
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: LoraBias

@dataclass(frozen=True, slots=True)
class HardwareSettings:
    """..."""
    use_cpu: bool
