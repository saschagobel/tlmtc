"""Run the tlmtc pipeline end-to-end.

Defines the user-facing library entrypoint for executing the tlmtc pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tlmtc.data_pipeline import DataPipeline
from tlmtc.evaluation_pipeline import EvaluationPipeline
from tlmtc.finetune_pipeline import FinetunePipeline
from tlmtc.paths import RunPaths, resolve_paths
from tlmtc.settings import UNSET, RunSettings, Unset, load_config_file


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Train metadata returned by `train_tlmtc`.

    Attributes:
        paths: Resolved filesystem layout for this run.
    """

    paths: RunPaths


def train_tlmtc(
    raw_csv: str | Path,
    *,
    raw_test_csv: str | Path | Unset = UNSET,
    work_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
    run_id: str | None | Unset = UNSET,
    target_name: str | Unset = UNSET,
    validation_size: float | Unset = UNSET,
    test_size: float | Unset = UNSET,
    random_seed: int | Unset = UNSET,
    transfer_learning: bool | Unset = UNSET,
    hyperparameter_tuning: bool | Unset = UNSET,
    threshold_optimization: bool | Unset = UNSET,
    threshold_type: str | Unset = UNSET,
    scale_learning_rate: bool | Unset = UNSET,
    wrap_peft: bool | Unset = UNSET,
    proxy_checkpoint: str | Unset = UNSET,
    checkpoint: str | Unset = UNSET,
    sequence_length: int | Unset = UNSET,
    best_model_metric: str | Unset = UNSET,
    batch_size: int | Unset = UNSET,
    train_epochs: int | Unset = UNSET,
    learning_rate: float | Unset = UNSET,
    weight_decay: float | Unset = UNSET,
    lr_scheduler: str | Unset = UNSET,
    best_threshold_metric: str | Unset = UNSET,
    tuning_trials: int | Unset = UNSET,
    optuna_space: dict[str, Any] | Unset = UNSET,
    lora_r: int | Unset = UNSET,
    lora_alpha: int | Unset = UNSET,
    lora_dropout: float | Unset = UNSET,
    lora_bias: str | Unset = UNSET,
    early_stopping_patience: int | Unset = UNSET,
    use_cpu: bool | Unset = UNSET,
) -> TrainResult:
    """Run the full tlmtc training pipeline end-to-end.

    Args:
        raw_csv: Path to the multilabel CSV.
        raw_test_csv: Optional path to a test CSV. If omitted, a test split is created from
            `raw_csv` according to `test_size`.
        work_dir: Base directory for resolving relative inputs and creating the run directory.
        config_path: Path to a YAML configuration file.
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
        optuna_space: Optional partial override for the Optuna search space. Values are merged into the
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
    settings = RunSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env=None,
        overrides={
            "raw_csv": raw_csv,
            "raw_test_csv": raw_test_csv,
            "work_dir": work_dir,
            "run_id": run_id,
            "model": {
                "target_name": target_name,
                "proxy_checkpoint": proxy_checkpoint,
                "checkpoint": checkpoint,
                "sequence_length": sequence_length,
            },
            "split": {
                "validation_size": validation_size,
                "test_size": test_size,
                "random_seed": random_seed,
            },
            "workflow": {
                "hyperparameter_tuning": hyperparameter_tuning,
                "threshold_optimization": threshold_optimization,
                "transfer_learning": transfer_learning,
                "scale_learning_rate": scale_learning_rate,
                "wrap_peft": wrap_peft,
            },
            "training": {
                "batch_size": batch_size,
                "train_epochs": train_epochs,
                "weight_decay": weight_decay,
                "learning_rate": learning_rate,
                "lr_scheduler": lr_scheduler,
                "best_model_metric": best_model_metric,
                "early_stopping_patience": early_stopping_patience,
            },
            "threshold": {
                "threshold_type": threshold_type,
                "best_threshold_metric": best_threshold_metric,
            },
            "hpo": {
                "tuning_trials": tuning_trials,
                "optuna_space": optuna_space,
            },
            "peft": {
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_bias": lora_bias,
            },
            "hardware": {
                "use_cpu": use_cpu,
            },
        },
    )

    paths = resolve_paths(
        raw_csv=settings.raw_csv,
        raw_test_csv=settings.raw_test_csv,
        work_dir=settings.work_dir,
        run_id=settings.run_id,
    ).ensure_dirs()

    data_pipeline = DataPipeline(
        paths=paths,
        split=settings.split,
        model=settings.model,
    )
    data_pipeline.split_data()
    data_pipeline.get_multi_hot_vectors()
    data_pipeline.create_hf_dataset()
    data_pipeline.tokenize_data()

    finetune_pipeline = FinetunePipeline(
        tokenized_dataset=data_pipeline.tokenized_dataset,
        paths=paths,
        model=settings.model,
        workflow=settings.workflow,
        peft=settings.peft,
        training=settings.training,
        hpo=settings.hpo,
        threshold=settings.threshold,
        hardware=settings.hardware,
    )
    finetune_pipeline.load_pretrained()
    finetune_pipeline.tune_hyperparameters()
    finetune_pipeline.fine_tune_pretrained()
    finetune_pipeline.tune_thresholds()
    finetune_pipeline.save_pretrained()

    evaluation_pipeline = EvaluationPipeline(
        tokenized_dataset=data_pipeline.tokenized_dataset,
        updated_trainer=finetune_pipeline.updated_trainer,
        paths=paths,
        model=settings.model,
        workflow=settings.workflow,
        training=settings.training,
        tuned_threshold=finetune_pipeline.tuned_threshold,
    )
    evaluation_pipeline.run_evaluation()
    evaluation_pipeline.save_metrics()
    evaluation_pipeline.render_tables()
    evaluation_pipeline.render_figures()

    return TrainResult(paths=paths)
