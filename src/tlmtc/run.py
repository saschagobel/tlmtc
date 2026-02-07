"""Run the tlmtc pipeline end-to-end.

Defines the user-facing library entrypoint for executing the tlmtc pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tlmtc import config
from tlmtc.data_pipeline import DataPipeline
from tlmtc.finetune_pipeline import FinetunePipeline
from tlmtc.paths import RunPaths, resolve_paths
from tlmtc.settings import (
    HardwareSettings,
    HpoSettings,
    ModelSettings,
    PeftSettings,
    SplitSettings,
    ThresholdSettings,
    TrainingSettings,
    WorkflowSettings,
)
from tlmtc.types import BestModelMetric, BestThresholdMetric, LoraBias, OptunaSpaceOverride, Threshold
from tlmtc.hpo import resolve_optuna_space

@dataclass(frozen=True, slots=True)
class RunResult:
    """Run metadata returned by `run_tlmtc`.

    Attributes:
        paths: Resolved filesystem layout for this run.
    """

    paths: RunPaths


def run_tlmtc(
    raw_csv: str | Path,
    *,
    raw_test_csv: str | Path | None = None,
    work_dir: str | Path | None = None,
    run_id: str | None = None,
    target_name: str = config.TARGET_NAME,
    validation_size: float = config.VALIDATION_SIZE,
    test_size: float = config.TEST_SIZE,
    random_seed: int = config.RANDOM_SEED,
    transfer_learning: bool = config.TRANSFER_LEARNING,
    hyperparameter_tuning: bool = config.HYPERPARAMETER_TUNING,
    threshold_optimization: bool = config.THRESHOLD_OPTIMIZATION,
    threshold_type: Threshold = config.THRESHOLD_TYPE,
    scale_learning_rate: bool = config.SCALE_LEARNING_RATE,
    wrap_peft: bool = config.WRAP_PEFT,
    proxy_checkpoint: str = config.PROXY_CHECKPOINT,
    checkpoint: str = config.CHECKPOINT,
    sequence_length: int = config.SEQUENCE_LENGTH,
    best_model_metric: BestModelMetric = config.BEST_MODEL_METRIC,
    batch_size: int = config.BATCH_SIZE,
    train_epochs: int = config.TRAIN_EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    lr_scheduler: str = config.LR_SCHEDULER,
    best_threshold_metric: BestThresholdMetric = config.BEST_THRESHOLD_METRIC,
    tuning_trials: int = config.TUNING_TRIALS,
    optuna_space_user: OptunaSpaceOverride | None = None,
    lora_r: int = config.LORA_R,
    lora_alpha: int = config.LORA_ALPHA,
    lora_dropout: float = config.LORA_DROPOUT,
    lora_bias: LoraBias = config.LORA_BIAS,
    early_stopping_patience: int = 1,
    use_cpu: bool = config.USE_CPU,
) -> RunResult:
    """Run the full tlmtc pipeline end-to-end.

    Args:
        raw_csv: Path to the multilabel CSV.
        raw_test_csv: Optional path to a test CSV. If omitted, a test split is created from
            `raw_csv` according to `test_size`.
        work_dir: Base directory for resolving relative inputs and creating the run directory.
        run_id: Optional run identifier used to name the run directory. If exists will resume.
        target_name: Display name for the classification target/task (used in logs/outputs).
        validation_size: Fraction of data used for validation split.
        test_size: Fraction of data used for test split (only used when `raw_test_csv` is None).
        random_seed: Random seed used for splitting/shuffling.
        transfer_learning: Whether to fine-tune a pretrained checkpoint.
        hyperparameter_tuning: Whether to run Optuna hyperparameter tuning.
        threshold_optimization: Whether to tune decision threshold(s) post-training.
        threshold_type: Threshold mode (e.g., global vs per-label).
        scale_learning_rate: Whether to scale learning rate based on batch size / device.
        wrap_peft: Whether to apply PEFT (LoRA) wrapping.
        proxy_checkpoint: Optional proxy checkpoint. If unset assumes checkpoint
        checkpoint: Base pretrained model checkpoint identifier.
        sequence_length: Max sequence length for tokenization.
        best_model_metric: Metric name used to select the best model checkpoint.
        batch_size: Training batch size.
        train_epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay.
        lr_scheduler: Scheduler identifier/name.
        best_threshold_metric: Metric name used to select optimal threshold(s).
        tuning_trials: Number of Optuna trials.
        optuna_space_user: Optional partial override for the Optuna search space. Values are merged into the
            default space (base or PEFT, depending on `wrap_peft`). See `OptunaSpaceOverride`
            for supported keys.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        lora_bias: LoRA bias handling (validated downstream).
        early_stopping_patience: Early stopping patience (epochs without improvement).
        use_cpu: Force CPU execution.

    Returns:
        RunResult: Metadata for this run.
    """
    paths = resolve_paths(
        raw_csv=raw_csv,
        raw_test_csv=raw_test_csv,
        work_dir=work_dir,
        run_id=run_id,
    ).ensure_dirs()

    optuna_space = resolve_optuna_space(
        wrap_peft=wrap_peft,
        space_base=config.OPTUNA_SPACE_BASE,
        space_peft=config.OPTUNA_SPACE_PEFT,
        override=optuna_space_user,
    )

    model_settings = ModelSettings(
        target_name=target_name,
        proxy_checkpoint=proxy_checkpoint,
        checkpoint=checkpoint,
        sequence_length=sequence_length,
        best_model_metric=best_model_metric,
    )
    split_settings = SplitSettings(
        validation_size=validation_size,
        test_size=test_size,
        random_seed=random_seed,
    )
    workflow_settings = WorkflowSettings(
        hyperparameter_tuning=hyperparameter_tuning,
        threshold_optimization=threshold_optimization,
        transfer_learning=transfer_learning,
        scale_learning_rate=scale_learning_rate,
        wrap_peft=wrap_peft,
    )
    training_settings = TrainingSettings(
        batch_size=batch_size,
        train_epochs=train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
    )
    threshold_settings = ThresholdSettings(
        threshold_type=threshold_type,
        best_threshold_metric=best_threshold_metric,
    )
    hpo_settings = HpoSettings(
        tuning_trials=tuning_trials,
        optuna_space=optuna_space,
    )
    peft_settings = PeftSettings(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
    )
    hardware_settings = HardwareSettings(use_cpu=use_cpu)

    processed = (
        DataPipeline(
            raw_data_path=paths.raw_data_path,
            raw_test_data_path=paths.raw_test_data_path,
            train_data_path=paths.train_data_path,
            val_data_path=paths.val_data_path,
            test_data_path=paths.test_data_path,
            validation_size=validation_size,
            test_size=test_size,
            random_seed=random_seed,
            checkpoint=checkpoint,
            sequence_length=sequence_length,
        )
        .split_data()
        .get_multi_hot_vectors()
        .create_hf_dataset()
        .tokenize_data()
    )

    (
        FinetunePipeline(
            tokenized_dataset=processed.tokenized_dataset,
            train_data_path=paths.train_data_path,
            val_data_path=paths.val_data_path,
            output_logging_path=paths.logs_dir,
            output_model_path=paths.model_dir,
            target_name=target_name,
            proxy_checkpoint=proxy_checkpoint,
            checkpoint=checkpoint,
            transfer_learning=transfer_learning,
            hyperparameter_tuning=hyperparameter_tuning,
            threshold_optimization=threshold_optimization,
            threshold_type=threshold_type,
            scale_learning_rate=scale_learning_rate,
            wrap_peft=wrap_peft,
            optuna_space_default_base=config.OPTUNA_SPACE_BASE,
            optuna_space_default_peft=config.OPTUNA_SPACE_PEFT,
            tuning_trials=tuning_trials,
            batch_size=batch_size,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            epochs=train_epochs,
            best_model_metric=best_model_metric,
            best_threshold_metric=best_threshold_metric,
            early_stopping_patience=early_stopping_patience,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
            use_cpu=use_cpu,
            optuna_space_user=optuna_space_user,
        )
        .load_pretrained()
        .tune_hyperparameters()
        .fine_tune_pretrained()
        .tune_thresholds()
        .save_pretrained()
    )
    return RunResult(paths=paths)
