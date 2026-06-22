"""Public Python API for running tlmtc training and prediction workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tlmtc.data_pipeline import DataPipeline
from tlmtc.data_preparation import create_prediction_dataset, read_prediction_csv, tokenize_prediction_dataset
from tlmtc.distributed import DistributedContext
from tlmtc.evaluation_pipeline import EvaluationPipeline
from tlmtc.finetune_pipeline import FinetunePipeline
from tlmtc.meta import TrainRunMeta, read_run_meta, write_run_meta
from tlmtc.paths import PredictionPaths, RunPaths, resolve_paths, resolve_prediction_paths
from tlmtc.prediction import (
    apply_thresholds,
    load_prediction_model,
    make_prediction_frame,
    predict_probabilities,
)
from tlmtc.runtime_output import configure_runtime_output, emit_progress
from tlmtc.settings import UNSET, PredictionSettings, RunSettings, Unset, load_config_file


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Result metadata for a completed tlmtc training run.

    Attributes:
        paths: Resolved filesystem layout containing input paths and generated run artifacts.
    """

    paths: RunPaths


@dataclass(frozen=True, slots=True)
class PredictResult:
    """Result metadata for a completed tlmtc prediction run.

    Attributes:
        paths: Resolved filesystem layout containing prediction inputs and generated artifacts.
    """

    paths: PredictionPaths


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
    trust_remote_code: bool | Unset = UNSET,
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
    trainer_args: dict[str, Any] | Unset = UNSET,
    verbosity: str | Unset = UNSET,
) -> TrainResult:
    """Run the full multi-label text classification training workflow.

    The workflow can perform data preparation, hyperparameter tuning, model fine-tuning,
    threshold optimization, evaluation, and reporting end-to-end according to the selected
    workflow flags.

    Args:
        raw_csv: Path to the raw multi-label training CSV. The file must contain a `text` column,
            at least two binary `label_*` columns, and optionally a `text_pair` column.
        raw_test_csv: Path to a separate raw test CSV. If omitted, a test split is created
            from `raw_csv` using `test_size`. Defaults to no separate test CSV.
        work_dir: Base directory for resolving inputs and writing run artifacts. Defaults to the
            current working directory.
        config_path: Path to a YAML configuration file. Defaults to no configuration file.
        run_id: Run identifier used to name the run directory. If omitted, a random
            identifier is generated.
        target_name: Display name for the classification target in logs and reports. Defaults to
            `"Target"`.
        validation_size: Fraction reserved for validation splitting. Defaults to `0.15`.
        test_size: Fraction reserved for test splitting when `raw_test_csv` is omitted. Defaults to
            `0.15`.
        random_seed: Random seed used for reproducible splitting and shuffling. Defaults to `2469`.
        transfer_learning: Whether to fine-tune the target checkpoint and produce model/evaluation
            artifacts. If `False`, data preparation still runs; with `hyperparameter_tuning=True`,
            tlmtc runs proxy-checkpoint hyperparameter tuning only. Defaults to `True`.
        hyperparameter_tuning: Whether to evaluate candidate hyperparameter configurations with
            Optuna before final fine-tuning. If `True` and `transfer_learning=False`, only the
            proxy-checkpoint tuning stage is run after data preparation. If both are `False`,
            the workflow stops after data preparation. Defaults to `True`.
        threshold_optimization: Whether to tune decision thresholds on validation-set predictions
            after fine-tuning. If `False`, evaluation uses the default threshold `0.5`. Ignored
            when `transfer_learning=False`. Defaults to `True`.
        threshold_type: Thresholding mode. Supported values are `"global"` and `"label"`. Defaults to
            `"label"`.
        scale_learning_rate: Whether to scale a proxy-tuned learning rate for the target checkpoint.
            Defaults to `False`.
        wrap_peft: Whether to use parameter-efficient fine-tuning with LoRA adapters. Defaults to `True`.
        proxy_checkpoint: Compatible encoder-only Hugging Face checkpoint identifier used during
            hyperparameter tuning. Defaults to `"microsoft/deberta-v3-small"`. If `checkpoint`
            is supplied and `proxy_checkpoint` is omitted, the proxy checkpoint defaults to the
            selected `checkpoint`. Loaded with the resolved `trust_remote_code` setting. Keep `trust_remote_code=False`
            unless you trust the checkpoint repository.
        checkpoint: Compatible encoder-only Hugging Face checkpoint identifier or local path used for
            final fine-tuning. Defaults to `"microsoft/deberta-v3-base"`. Prediction reloads the trained model or
            adapter artifacts using the resolved prediction `trust_remote_code` setting. Keep it disabled unless you
            trust the saved artifacts and checkpoint repository.
        sequence_length: Maximum tokenized sequence length. Defaults to `128`.
        trust_remote_code: Whether Hugging Face tokenizer, config, and model loading may execute
            custom remote code. Defaults to `False`. Only enable this for model repositories or
            local model artifacts you trust.
        best_model_metric: Metric used to select the best model checkpoint. Supported values are
            `"f1_micro"`, `"f1_macro"`, `"roc_auc_micro"`, and `"roc_auc_macro"`. Defaults to
            `"roc_auc_macro"`.
        batch_size: Initial training and evaluation batch size. Used directly when hyperparameter tuning is
            disabled, otherwise replaced by the tuned value. Defaults to `16`.
        train_epochs: Initial number of training epochs. Used directly when hyperparameter tuning is
            disabled, otherwise replaced by the tuned value. Defaults to `20`.
        learning_rate: Initial optimizer learning rate. Used directly when hyperparameter tuning is
            disabled, otherwise replaced by the tuned value. Defaults to `2e-5`.
        weight_decay: Initial weight decay for training. Used directly when hyperparameter tuning is
            disabled, otherwise replaced by the tuned value. Defaults to `0.01`.
        lr_scheduler: Initial learning-rate scheduler name. Used directly when hyperparameter tuning is
            disabled, otherwise replaced by the tuned value. Defaults to `"linear"`.
        best_threshold_metric: Metric used to select decision thresholds. Supported values are
            `"f1_micro"` and `"f1_macro"`. Defaults to `"f1_macro"`.
        tuning_trials: Number of hyperparameter configurations to evaluate during Optuna tuning. Higher
            values may improve the selected configuration but increase runtime. Defaults to `10`.
        optuna_space: Optional partial override for the hyperparameter tuning ranges and candidate
            values. Supported keys are `lr_low`, `lr_high`, `batch_sizes`, `wd_low`, `wd_high`,
            `schedulers`, `epoch_low`, `epoch_high`. Missing keys are filled from the default tuning space
            selected by `wrap_peft`.

            Defaults to the PEFT search space when `wrap_peft=True`:

            {
                "lr_low": 5e-5,
                "lr_high": 4e-4,
                "batch_sizes": [8, 16, 32],
                "wd_low": 0.0,
                "wd_high": 0.01,
                "schedulers": ["linear", "cosine"],
                "epoch_low": 5,
                "epoch_high": 20,
                "lr_reference_batch_size": 32,
            }

            Defaults to the full fine-tuning search space when `wrap_peft=False`:

            {
                "lr_low": 1e-5,
                "lr_high": 1e-4,
                "batch_sizes": [8, 16, 32],
                "wd_low": 0.0,
                "wd_high": 0.1,
                "schedulers": ["linear", "cosine", "polynomial"],
                "epoch_low": 5,
                "epoch_high": 30,
                "lr_reference_batch_size": 32,
            }
        lora_r: LoRA rank. Defaults to `8`.
        lora_alpha: LoRA scaling factor. Defaults to `32`.
        lora_dropout: LoRA dropout probability. Defaults to `0.1`.
        lora_bias: LoRA bias handling mode. Supported values are `"none"`, `"all"`, and `"lora_only"`.
            Defaults to `"none"`.
        early_stopping_patience: Early stopping patience in epochs without improvement. Defaults to
            `10`.
        use_cpu: Whether to force CPU execution. Defaults to `False`.
        trainer_args: Additional Hugging Face TrainingArguments keyword arguments.
            Keys already managed by tlmtc, such as batch size, epochs, learning rate,
            output directory, model-selection settings, logging/reporting behavior,
            and CPU selection, are rejected. Defaults to no additional arguments.
        verbosity: Runtime output mode. Supported values are `"progress"` and `"quiet"`. Defaults to
            `"progress"`.

    Returns:
        Result metadata containing the resolved input and artifact paths.
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
                "trust_remote_code": trust_remote_code,
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
                "trainer_args": trainer_args,
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
            "runtime": {
                "verbosity": verbosity,
            },
        },
    )

    distributed = DistributedContext.create(use_cpu=settings.hardware.use_cpu)
    configure_runtime_output(
        settings.runtime.verbosity,
        is_main_process=distributed.is_main_process,
    )
    distributed.warn_if_multi_gpu_without_launcher(use_cpu=settings.hardware.use_cpu)

    if distributed.is_distributed and settings.workflow.hyperparameter_tuning:
        raise RuntimeError(
            "Hyperparameter tuning is not supported under distributed launch. "
            "Run HPO in a single-process run, then rerun distributed final training "
            "with hyperparameter_tuning=False and the selected training settings."
        )

    resolved_run_id = distributed.resolve_run_id(settings.run_id)

    paths = resolve_paths(
        raw_csv=settings.raw_csv,
        raw_test_csv=settings.raw_test_csv,
        work_dir=settings.work_dir,
        run_id=resolved_run_id,
    )

    distributed.run_on_main(paths.ensure_dirs, sync=True)

    emit_progress("Starting training run")

    data_pipeline = DataPipeline(
        paths=paths,
        split=settings.split,
        model=settings.model,
    )
    with distributed.main_process_first():
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
        input_mode=data_pipeline.input_mode,
    )
    evaluation_pipeline.run_evaluation()
    distributed.run_on_main(evaluation_pipeline.save_metrics)
    distributed.run_on_main(evaluation_pipeline.render_tables)
    distributed.run_on_main(evaluation_pipeline.render_figures)

    distributed.run_on_main(
        write_run_meta,
        meta=TrainRunMeta(
            run_id=resolved_run_id,
            target_name=settings.model.target_name,
            checkpoint=settings.model.checkpoint,
            proxy_checkpoint=settings.model.proxy_checkpoint,
            sequence_length=settings.model.sequence_length,
            trust_remote_code=settings.model.trust_remote_code,
            input_mode=data_pipeline.input_mode,
            label_names=evaluation_pipeline.label_names,
            threshold_type=settings.threshold.threshold_type,
            thresholds=finetune_pipeline.tuned_threshold.tolist(),
            transfer_learning=settings.workflow.transfer_learning,
            hyperparameter_tuning=settings.workflow.hyperparameter_tuning,
            threshold_optimization=settings.workflow.threshold_optimization,
            scale_learning_rate=settings.workflow.scale_learning_rate,
            wrap_peft=settings.workflow.wrap_peft,
        ),
        path=paths.train_run_meta_path,
    )

    emit_progress("Training run complete")
    return TrainResult(paths=paths)


def predict_tlmtc(
    prediction_csv: str | Path,
    *,
    work_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
    run_id: str | None | Unset = UNSET,
    batch_size: int | Unset = UNSET,
    trust_remote_code: bool | Unset = UNSET,
    use_cpu: bool | Unset = UNSET,
    verbosity: str | Unset = UNSET,
) -> PredictResult:
    """Run the multi-label text classification prediction workflow.

    Prediction consumes persisted metadata and model artifacts from a completed
    training run, applies the persisted decision thresholds, and writes probability
    and binary prediction artifacts.

    Args:
        prediction_csv: Path to the unlabeled prediction CSV. The file must contain a `text`
            column and, for models trained with paired-text inputs, a `text_pair` column.
            Prediction artifacts preserve input text columns unchanged.
        work_dir: Base directory for resolving inputs, reading training artifacts, and writing
            prediction artifacts. Defaults to the current working directory.
        config_path: Path to a YAML configuration file. Defaults to no configuration file.
        run_id: Run identifier used to select the completed training run. If omitted, the latest
            completed training run is selected from persisted training metadata. Prediction reloads
            the trained model or adapter artifacts for this run with `trust_remote_code=False`;
            artifacts that require custom remote code are not supported. Only use saved model
            artifacts and adapters you trust.
        batch_size: Prediction batch size used for batched inference. Defaults to `32`.
        trust_remote_code: Whether Hugging Face tokenizer and model loading may execute custom
            remote code during prediction. Defaults to `False`. Required for runs trained with
            `trust_remote_code=True`; only enable it for artifacts and checkpoints you trust.
        use_cpu: Whether to force CPU execution. Defaults to `False`.
        verbosity: Runtime output mode. Supported values are `"progress"` and `"quiet"`. Defaults to
            `"progress"`.

    Returns:
        Result metadata containing the resolved input and artifact paths.
    """
    settings = PredictionSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env=None,
        overrides={
            "prediction_csv": prediction_csv,
            "work_dir": work_dir,
            "run_id": run_id,
            "batch_size": batch_size,
            "trust_remote_code": trust_remote_code,
            "hardware": {
                "use_cpu": use_cpu,
            },
            "runtime": {
                "verbosity": verbosity,
            },
        },
    )

    distributed = DistributedContext.create(use_cpu=settings.hardware.use_cpu)
    configure_runtime_output(
        settings.runtime.verbosity,
        is_main_process=distributed.is_main_process,
    )

    emit_progress("Starting prediction run")

    paths = resolve_prediction_paths(
        input_csv=settings.prediction_csv,
        work_dir=settings.work_dir,
        run_id=settings.run_id,
    )

    distributed.run_on_main(paths.ensure_dirs, sync=True)

    emit_progress("Reading training metadata")

    meta = read_run_meta(paths.train_run_meta_path)

    if not meta.transfer_learning:
        raise RuntimeError(
            "Prediction requires a training run with transfer_learning=True. "
            f"Run '{meta.run_id}' did not persist a fine-tuned prediction model."
        )

    if meta.trust_remote_code and not settings.trust_remote_code:
        raise RuntimeError(
            f"Prediction run '{meta.run_id}' was trained with trust_remote_code=True, "
            "but prediction was started with trust_remote_code=False. "
            "Re-run prediction with trust_remote_code=True or --trust-remote-code only if you trust "
            f"the Hugging Face model repository or local model artifacts for '{meta.checkpoint}'."
        )

    input_mode = meta.input_mode
    label_names = meta.label_names

    assert input_mode is not None
    assert label_names is not None

    emit_progress("Reading prediction inputs")

    input_df = read_prediction_csv(
        df_path=paths.input_data_path,
        expected_input_mode=input_mode,
    )
    prediction_dataset = create_prediction_dataset(
        df=input_df,
        input_mode=input_mode,
    )
    emit_progress("Tokenizing prediction inputs")
    tokenized_dataset = tokenize_prediction_dataset(
        dataset=prediction_dataset,
        checkpoint=meta.checkpoint,
        input_mode=input_mode,
        sequence_length=meta.sequence_length,
        trust_remote_code=settings.trust_remote_code,
    )
    emit_progress("Loading fine-tuned prediction model")
    model = load_prediction_model(
        model_dir=paths.train_run_model_dir,
        checkpoint=meta.checkpoint,
        num_labels=len(label_names),
        wrap_peft=meta.wrap_peft,
        trust_remote_code=settings.trust_remote_code,
    )
    emit_progress("Running prediction")
    probabilities = predict_probabilities(
        model=model,
        dataset=tokenized_dataset,
        batch_size=settings.batch_size,
        use_cpu=settings.hardware.use_cpu,
    )
    probability_df = make_prediction_frame(
        input_df=input_df,
        values=probabilities,
        label_names=label_names,
    )
    predictions = apply_thresholds(
        probabilities=probabilities,
        thresholds=meta.thresholds,
    )
    prediction_df = make_prediction_frame(
        input_df=input_df,
        values=predictions,
        label_names=label_names,
    )

    emit_progress("Writing prediction artifacts")
    distributed.run_on_main(probability_df.to_csv, paths.probabilities_path, index=False)
    distributed.run_on_main(prediction_df.to_csv, paths.predictions_path, index=False)

    emit_progress("Prediction run complete")
    return PredictResult(paths=paths)
