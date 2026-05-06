"""Settings models and layered configuration resolution for tlmtc runs."""

from pathlib import Path
from typing import Any, Final, Literal, Self
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class _UnsetType:
    """Sentinel type for omitted layered override values."""

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
    """Merge a higher-precedence settings layer into a base dictionary.

    Args:
        base: Lower-precedence settings layer.
        incoming: Higher-precedence settings layer.

    Returns:
        New dictionary containing the merged settings.
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
    """Remove UNSET sentinel values from nested override data.

    Args:
        value: Arbitrary nested override structure.

    Returns:
        Equivalent structure with UNSET values removed.
    """
    if isinstance(value, dict):
        return {key: prune_unset(item) for key, item in value.items() if not isinstance(item, _UnsetType)}

    if isinstance(value, list):
        return [prune_unset(item) for item in value if not isinstance(item, _UnsetType)]

    return value


def load_config_file(
    path: str | Path,
) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration mapping.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
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
    """Base model for resolving settings from layered configuration sources."""

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

        Later layers take precedence over earlier layers in the order `config < env < overrides`.
        Override values marked as UNSET are removed before validation.

        Args:
            config: Configuration-file settings layer.
            env: Environment-derived settings layer.
            overrides: Explicit call-site override layer.

        Returns:
            Validated settings instance.
        """
        resolved: dict[str, Any] = {}
        resolved = deep_merge(resolved, config or {})
        resolved = deep_merge(resolved, env or {})
        resolved = deep_merge(resolved, prune_unset(overrides or {}))
        return cls.model_validate(resolved)


class ModelSettings(BaseModel):
    """Model and tokenizer settings.

    Attributes:
        target_name: Display name for the classification target.
        proxy_checkpoint: Proxy checkpoint used during hyperparameter optimization.
        checkpoint: Target checkpoint used for final fine-tuning.
        sequence_length: Maximum tokenized sequence length.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_name: str = "Target"
    proxy_checkpoint: str = "microsoft/deberta-v3-xsmall"
    checkpoint: str = "microsoft/deberta-v3-base"
    sequence_length: PositiveInt = 128


class SplitSettings(BaseModel):
    """Data splitting settings.

    Attributes:
        validation_size: Fraction of training data reserved for validation.
        test_size: Fraction of raw data reserved for testing when no raw test CSV is provided.
        random_seed: Random seed used for reproducible splitting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    random_seed: int = 2469


class WorkflowSettings(BaseModel):
    """Workflow stage toggles.

    Attributes:
        hyperparameter_tuning: Whether to run Optuna hyperparameter tuning.
        threshold_optimization: Whether to tune post-training decision thresholds.
        transfer_learning: Whether to fine-tune a pretrained checkpoint.
        scale_learning_rate: Whether to scale proxy-tuned learning rates for the target checkpoint.
        wrap_peft: Whether to apply PEFT/LoRA wrapping.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hyperparameter_tuning: bool = True
    threshold_optimization: bool = True
    transfer_learning: bool = True
    scale_learning_rate: bool = False
    wrap_peft: bool = True


class TrainingSettings(BaseModel):
    """Training hyperparameter settings.

    Attributes:
        batch_size: Training and evaluation batch size.
        train_epochs: Number of training epochs.
        weight_decay: Weight decay applied during optimization.
        learning_rate: Initial optimizer learning rate.
        lr_scheduler: Learning-rate scheduler name.
        best_model_metric: Model-selection metric.
        early_stopping_patience: Early stopping patience in epochs without improvement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: PositiveInt = 16
    train_epochs: PositiveInt = 20
    weight_decay: float = Field(default=0.01, ge=0.0)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    lr_scheduler: str = "linear"
    best_model_metric: Literal["f1_micro", "f1_macro", "roc_auc_micro", "roc_auc_macro"] = "roc_auc_macro"
    early_stopping_patience: PositiveInt = 10


class ThresholdSettings(BaseModel):
    """Decision-threshold optimization settings.

    Attributes:
        threshold_type: Thresholding mode, either global or per-label.
        best_threshold_metric: Metric used to select decision thresholds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold_type: Literal["global", "label"] = "label"
    best_threshold_metric: Literal["f1_micro", "f1_macro"] = "f1_macro"


class OptunaSpaceSettings(BaseModel):
    """Validated Optuna hyperparameter search space.

    Attributes:
        lr_low: Lower learning-rate bound.
        lr_high: Upper learning-rate bound.
        batch_sizes: Candidate batch sizes.
        wd_low: Lower weight-decay bound.
        wd_high: Upper weight-decay bound.
        schedulers: Candidate learning-rate schedulers.
        epoch_low: Lower epoch-count bound.
        epoch_high: Upper epoch-count bound.
        lr_reference_batch_size: Reference batch size used for learning-rate scaling.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lr_low: float = Field(..., gt=0.0)
    lr_high: float = Field(..., gt=0.0)
    batch_sizes: list[PositiveInt] = Field(..., min_length=1)
    wd_low: float = Field(..., ge=0.0)
    wd_high: float = Field(..., ge=0.0)
    schedulers: list[str] = Field(..., min_length=1)
    epoch_low: PositiveInt
    epoch_high: PositiveInt
    lr_reference_batch_size: PositiveInt

    @model_validator(mode="after")
    def validate_space(self) -> Self:
        """Validate Optuna search-space bounds and categorical choices.

        Returns:
            Validated Optuna search-space settings.

        Raises:
            ValueError: If numeric bounds are inconsistent or schedulers contain empty values.
        """
        if self.lr_low >= self.lr_high:
            raise ValueError("optuna_space.lr_low must be strictly smaller than optuna_space.lr_high.")

        if self.wd_low > self.wd_high:
            raise ValueError("optuna_space.wd_low must be less than or equal to optuna_space.wd_high.")

        if self.epoch_low > self.epoch_high:
            raise ValueError("optuna_space.epoch_low must be less than or equal to optuna_space.epoch_high.")

        if any(not scheduler.strip() for scheduler in self.schedulers):
            raise ValueError("optuna_space.schedulers must not contain empty strings.")

        return self


_DEFAULT_OPTUNA_SPACE_BASE: Final[OptunaSpaceSettings] = OptunaSpaceSettings(
    lr_low=1e-5,
    lr_high=8e-5,
    batch_sizes=[8, 16, 32],
    wd_low=0.0,
    wd_high=0.1,
    schedulers=["linear", "cosine", "polynomial"],
    epoch_low=5,
    epoch_high=30,
    lr_reference_batch_size=32,
)

_DEFAULT_OPTUNA_SPACE_PEFT: Final[OptunaSpaceSettings] = OptunaSpaceSettings(
    lr_low=5e-5,
    lr_high=4e-4,
    batch_sizes=[8, 16, 32],
    wd_low=0.0,
    wd_high=0.01,
    schedulers=["linear", "cosine"],
    epoch_low=5,
    epoch_high=20,
    lr_reference_batch_size=32,
)


class HpoSettings(BaseModel):
    """Hyperparameter optimization settings.

    Attributes:
        tuning_trials: Number of Optuna trials.
        optuna_space: Resolved Optuna hyperparameter search space.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tuning_trials: PositiveInt = 10
    optuna_space: OptunaSpaceSettings


class PeftSettings(BaseModel):
    """PEFT/LoRA settings.

    Attributes:
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        lora_bias: LoRA bias handling mode.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lora_r: PositiveInt = 8
    lora_alpha: PositiveInt = 32
    lora_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    lora_bias: Literal["none", "all", "lora_only"] = "none"


class HardwareSettings(BaseModel):
    """Hardware execution settings.

    Attributes:
        use_cpu: Whether to force CPU execution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    use_cpu: bool = False


class RunSettings(ResolvableSettings):
    """Resolved top-level settings for a tlmtc training run.

    Attributes:
        raw_csv: Path to the raw training CSV.
        raw_test_csv: Optional path to a raw test CSV.
        work_dir: Base directory for resolving inputs and writing run artifacts.
        run_id: Run identifier used to name the output directory.
        model: Model and tokenizer settings.
        split: Data splitting settings.
        workflow: Workflow stage toggles.
        training: Training hyperparameter settings.
        threshold: Decision-threshold optimization settings.
        hpo: Hyperparameter optimization settings.
        peft: PEFT/LoRA settings.
        hardware: Hardware execution settings.
    """

    model_config = ConfigDict(extra="forbid")

    raw_csv: Path
    raw_test_csv: Path | None = None
    work_dir: Path = Field(default_factory=Path.cwd)
    run_id: str = Field(default_factory=lambda: uuid4().hex)

    model: ModelSettings = Field(default_factory=ModelSettings)
    split: SplitSettings = Field(default_factory=SplitSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    threshold: ThresholdSettings = Field(default_factory=ThresholdSettings)
    hpo: HpoSettings
    peft: PeftSettings = Field(default_factory=PeftSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)

    @model_validator(mode="before")
    @classmethod
    def resolve_optuna_space(cls, value: Any) -> Any:
        """Resolve the effective Optuna search space before settings validation.

        Args:
            value: Raw settings mapping passed to Pydantic validation.

        Returns:
            Settings mapping with defaults and user-provided Optuna search-space values merged.

        Raises:
            TypeError: If `hpo.optuna_space` is not a mapping.
        """
        workflow = value.get("workflow") or {}
        hpo = value.get("hpo") or {}

        wrap_peft = workflow.get("wrap_peft", WorkflowSettings.model_fields["wrap_peft"].default)
        default_space = _DEFAULT_OPTUNA_SPACE_PEFT if wrap_peft else _DEFAULT_OPTUNA_SPACE_BASE

        user_space = hpo.get("optuna_space") or {}
        if not isinstance(user_space, dict):
            raise TypeError("hpo.optuna_space must be a mapping of Optuna-space fields to override values.")
        hpo["optuna_space"] = deep_merge(default_space.model_dump(mode="python"), user_space)

        value["hpo"] = hpo
        return value
