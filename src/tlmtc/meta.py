"""Metadata models and IO helpers for tlmtc run artifacts."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from tlmtc.data_contracts import InputMode


class TrainRunMeta(BaseModel):
    """Persisted metadata for a completed tlmtc training run.

    This metadata file captures the stable contract needed to inspect and reuse
    a training run, especially from downstream workflows such as prediction.

    Attributes:
        run_id: Run identifier used to name the training-run directory.
        created_at: UTC timestamp at which the metadata file was created.
        tlmtc_version: Installed tlmtc package version used for the training run.
        target_name: Display name for the classification target.
        checkpoint: Target checkpoint used for final fine-tuning.
        proxy_checkpoint: Proxy checkpoint used during hyperparameter tuning.
        sequence_length: Maximum tokenized sequence length used during training.
        trust_remote_code: Whether training loaded Hugging Face artifacts with custom remote code enabled.
        input_mode: Text-input layout inferred from the training data.
        label_names: Ordered human-readable label names without the `label_` prefix.
        threshold_type: Thresholding mode used for the persisted decision thresholds.
        thresholds: Global or label-specific decision thresholds.
        transfer_learning: Whether target-checkpoint fine-tuning was enabled.
        hyperparameter_tuning: Whether Optuna hyperparameter tuning was enabled.
        hpo_hyperparameters_applied: Whether HPO-produced hyperparameters were applied to final fine-tuning.
        threshold_optimization: Whether validation-set threshold optimization was enabled.
        scale_learning_rate: Whether proxy-tuned learning rates were scaled for the target checkpoint.
        wrap_peft: Whether PEFT/LoRA wrapping was enabled.
        model_backends: Model artifact backends available for downstream inference.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tlmtc_version: str

    target_name: str
    checkpoint: str
    proxy_checkpoint: str
    sequence_length: PositiveInt
    trust_remote_code: bool

    input_mode: InputMode | None
    label_names: list[str] | None

    threshold_type: Literal["global", "label"]
    thresholds: list[float]

    transfer_learning: bool
    hyperparameter_tuning: bool
    hpo_hyperparameters_applied: bool
    threshold_optimization: bool
    scale_learning_rate: bool
    wrap_peft: bool
    model_backends: list[Literal["torch", "onnx"]]


def write_run_meta(
    meta: TrainRunMeta,
    path: Path,
) -> None:
    """Write training-run metadata as a JSON artifact.

    Args:
        meta: Training-run metadata to persist.
        path: Destination JSON path.
    """
    path.write_text(
        meta.model_dump_json(indent=4),
        encoding="utf-8",
    )


def read_run_meta(
    path: Path,
) -> TrainRunMeta:
    """Read training-run metadata from a JSON artifact.

    Args:
        path: Source JSON path.

    Returns:
        Validated training-run metadata.
    """
    return TrainRunMeta.model_validate_json(path.read_text(encoding="utf-8"))
