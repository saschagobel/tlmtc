"""Settings bundles and layered resolution.

Shared infrastructure for resolving run settings from layered inputs + settings bundles
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import yaml
from pydantic import BaseModel, ConfigDict

from tlmtc.types import (
    BestModelMetric,
    BestThresholdMetric,
    LoraBias,
    OptunaSpace,
    Threshold,
)


class _UnsetType:
    """Sentinel representing an omitted override value."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        raise TypeError("UNSET has no truth value. Use explicit checks: `x is UNSET` (omitted) or `x is not UNSET`.")


UNSET: Final[_UnsetType] = _UnsetType()
type Unset = _UnsetType


def deep_merge(
    base: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge a higher-precedence settings layer into a base dictionary.

    Args:
        base: Lower-precedence settings layer.
        incoming: Higher-precedence settings layer.

    Returns:
        A new dictionary containing the merged settings.
    """
    merged: dict[str, Any] = dict(base)

    for key, value in incoming.items():
        current = merged.get(key)

        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value

    return merged


def prune_unset(
    value: Any,
) -> Any:
    """Recursively remove values marked as UNSET from override data.

    Args:
        value: Arbitrary nested structure.

    Returns:
        The same structure with all `UNSET` values removed.
    """
    if isinstance(value, dict):
        return {key: prune_unset(item) for key, item in value.items() if not isinstance(item, _UnsetType)}

    if isinstance(value, list):
        return [prune_unset(item) for item in value if not isinstance(item, _UnsetType)]

    return value


def load_config_file(
    path: str | Path,
) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Args:
        path: Path to the config file.

    Returns:
        Parsed config data.

    Raises:
        FileNotFoundError: If the config file does not exist.
        TypeError: If the YAML root is not a mapping.
    """
    config_path = Path(path).expanduser()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    content = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"Config file root must be a mapping, got {type(data).__name__}: {config_path}")

    return data


class ResolvableSettings(BaseModel):
    """Shared base model for resolving layered run settings."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def resolve(
        cls,
        *,
        config: dict[str, Any] | None = None,
        env: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Self:
        """Resolve settings from config, environment, and explicit overrides.

        Precedence is low to high:
            config < env < overrides

        Override values marked as `UNSET` are removed before validation.

        Args:
            config: Config-file settings layer.
            env: Environment-derived settings layer.
            overrides: Explicit call-site overrides.

        Returns:
            A validated settings instance.
        """
        resolved: dict[str, Any] = {}
        resolved = deep_merge(resolved, config or {})
        resolved = deep_merge(resolved, env or {})
        resolved = deep_merge(resolved, prune_unset(overrides or {}))
        return cls.model_validate(resolved)


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
