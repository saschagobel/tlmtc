"""Filesystem path layout for tlmtc training runs."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Self

from tlmtc.meta import read_run_meta

DEFAULT_TRAIN_OUTPUTS_DIRNAME: Final[str] = "train_outputs"
DEFAULT_PREDICTION_OUTPUTS_DIRNAME: Final[str] = "prediction_outputs"
DEFAULT_DATA_DIRNAME: Final[str] = "data"
DEFAULT_LOGS_DIRNAME: Final[str] = "logs"
DEFAULT_HPO_CHECKPOINTS_DIRNAME: Final[str] = "hpo_checkpoints"
DEFAULT_MODEL_DIRNAME: Final[str] = "model"
DEFAULT_EVAL_DIRNAME: Final[str] = "evaluation"
TRAIN_RUN_META_FILENAME: Final[str] = "train_run_meta.json"
BEST_HYPERPARAMETERS_FILENAME: Final[str] = "best_hyperparameters.json"

RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(
    run_id: str,
) -> str:
    """Validate a run identifier as a safe single path segment.

    Args:
        run_id: Run identifier to validate.

    Returns:
        The validated run identifier.

    Raises:
        ValueError: If `run_id` is not a safe path segment.
    """
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be a safe path segment matching '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'.")

    return run_id


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Resolved filesystem paths for a single tlmtc training run.

    Directory structure:
        <work_dir>/
            train_outputs/
                <run_id>/
                    train_run_meta.json
                    data/
                    logs/
                        hpo_checkpoints/
                    model/
                    evaluation/

    Attributes:
        work_dir: Base directory used to resolve inputs and contain training outputs.
        run_dir: Root output directory for the training run.
        run_id: Run identifier used to name the output directory.
        train_run_meta_path: Path to the persisted training-run metadata sidecar.
        raw_data_path: Resolved path to the raw training CSV.
        raw_test_data_path: Resolved path to the optional raw test CSV.
        data_dir: Directory for prepared dataset split artifacts.
        logs_dir: Directory for training logs and HPO artifacts.
        hpo_checkpoints_dir: Directory for Trainer checkpoints created during HPO.
        model_dir: Directory for saved model artifacts.
        eval_dir: Directory for evaluation metrics, tables, and figures.
        train_data_path: Path to the prepared training split.
        val_data_path: Path to the prepared validation split.
        test_data_path: Path to the prepared test split.
        optuna_trials_path: Path to the persisted Optuna study database.
        best_hyperparameters_path: Path to the persisted effective hyperparameters selected by HPO.
        global_metrics_path: Path to the aggregate metrics JSON artifact.
        label_metrics_path: Path to the per-label metrics JSON artifact.
        global_metrics_table_path: Path to the aggregate metrics HTML table.
        label_metrics_table_path: Path to the per-label metrics HTML table.
        hyperparameters_table_path: Path to the hyperparameters HTML table.
        roc_plot_path: Path to the ROC curve PDF.
        co_occurrence_plot_path: Path to the label co-occurrence PDF.
        loss_plot_path: Path to the loss curve PDF.
        objective_values_plot_path: Path to the Optuna objective-values PDF.
    """

    work_dir: Path
    run_dir: Path
    run_id: str

    train_run_meta_path: Path

    raw_data_path: Path
    raw_test_data_path: Path | None

    data_dir: Path
    logs_dir: Path
    hpo_checkpoints_dir: Path
    model_dir: Path
    eval_dir: Path

    train_data_path: Path
    val_data_path: Path
    test_data_path: Path

    optuna_trials_path: Path
    best_hyperparameters_path: Path

    global_metrics_path: Path
    label_metrics_path: Path
    global_metrics_table_path: Path
    label_metrics_table_path: Path
    hyperparameters_table_path: Path
    roc_plot_path: Path
    co_occurrence_plot_path: Path
    loss_plot_path: Path
    objective_values_plot_path: Path

    def ensure_dirs(
        self,
    ) -> Self:
        """Create run artifact directories.

        Returns:
            Updated path bundle.
        """
        for directory in (
            self.data_dir,
            self.hpo_checkpoints_dir,
            self.model_dir,
            self.eval_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def resolve_paths(
    *,
    raw_csv: Path,
    raw_test_csv: Path | None,
    work_dir: Path,
    run_id: str,
    outputs_dirname: str = DEFAULT_TRAIN_OUTPUTS_DIRNAME,
    data_dirname: str = DEFAULT_DATA_DIRNAME,
    logs_dirname: str = DEFAULT_LOGS_DIRNAME,
    hpo_checkpoints_dirname: str = DEFAULT_HPO_CHECKPOINTS_DIRNAME,
    model_dirname: str = DEFAULT_MODEL_DIRNAME,
    eval_dirname: str = DEFAULT_EVAL_DIRNAME,
) -> RunPaths:
    """Resolve input and artifact paths for a tlmtc training run.

    Args:
        raw_csv: Path to the raw training CSV.
        raw_test_csv: Optional path to a raw test CSV.
        work_dir: Base directory used to resolve inputs and contain run outputs.
        run_id: Run identifier used to name the output directory.
        outputs_dirname: Name of the outputs root directory under `work_dir`.
        data_dirname: Name of the prepared-data directory under `run_dir`.
        logs_dirname: Name of the logs directory under `run_dir`.
        hpo_checkpoints_dirname: Name of the HPO checkpoint directory under `logs_dir`.
        model_dirname: Name of the model directory under `run_dir`.
        eval_dirname: Name of the evaluation directory under `run_dir`.

    Returns:
        Resolved path bundle for inputs and run artifacts.
    """
    resolved_work_dir = work_dir.expanduser().resolve()

    raw_data_path = raw_csv.expanduser().resolve()
    raw_test_data_path = None if raw_test_csv is None else raw_test_csv.expanduser().resolve()

    resolved_run_id = validate_run_id(run_id)

    run_dir = resolved_work_dir / outputs_dirname / resolved_run_id
    data_dir = run_dir / data_dirname
    logs_dir = run_dir / logs_dirname
    hpo_checkpoints_dir = logs_dir / hpo_checkpoints_dirname
    model_dir = run_dir / model_dirname
    eval_dir = run_dir / eval_dirname

    return RunPaths(
        work_dir=resolved_work_dir,
        run_dir=run_dir,
        run_id=run_id,
        train_run_meta_path=run_dir / TRAIN_RUN_META_FILENAME,
        raw_data_path=raw_data_path,
        raw_test_data_path=raw_test_data_path,
        data_dir=data_dir,
        logs_dir=logs_dir,
        hpo_checkpoints_dir=hpo_checkpoints_dir,
        model_dir=model_dir,
        eval_dir=eval_dir,
        train_data_path=data_dir / "train.parquet",
        val_data_path=data_dir / "val.parquet",
        test_data_path=data_dir / "test.parquet",
        global_metrics_path=eval_dir / "global_metrics.json",
        label_metrics_path=eval_dir / "label_metrics.json",
        global_metrics_table_path=eval_dir / "global_metrics_table.html",
        label_metrics_table_path=eval_dir / "label_metrics_table.html",
        hyperparameters_table_path=eval_dir / "hyperparameters_table.html",
        roc_plot_path=eval_dir / "roc_plot.pdf",
        co_occurrence_plot_path=eval_dir / "co_occurrence.pdf",
        loss_plot_path=eval_dir / "loss_plot.pdf",
        objective_values_plot_path=eval_dir / "objective_values_plot.pdf",
        optuna_trials_path=logs_dir / "optuna_trials.db",
        best_hyperparameters_path=logs_dir / BEST_HYPERPARAMETERS_FILENAME,
    )


@dataclass(frozen=True, slots=True)
class PredictionPaths:
    """Resolved filesystem paths for a tlmtc prediction run.

    Prediction consumes artifacts from an existing training run and writes
    prediction artifacts under a separate prediction-output root.

    Directory structure:
        <work_dir>/
            train_outputs/
                <run_id>/
                    train_run_meta.json
                    model/
            prediction_outputs/
                <run_id>/
                    probabilities.csv
                    predictions.csv

    Attributes:
        work_dir: Existing base directory containing tlmtc training outputs.
        run_id: Training-run identifier used as the prediction source.
        input_data_path: Resolved path to the unlabeled prediction input CSV.
        train_outputs_dir: Directory containing all training runs.
        train_run_dir: Existing training-run directory consumed for prediction.
        train_run_meta_path: Path to the required training-run metadata.
        train_run_model_dir: Path to the saved model or adapter artifacts.
        prediction_outputs_dir: Root directory for all prediction outputs.
        prediction_run_dir: Output directory for predictions from this training run.
        predictions_path: Destination CSV path for prediction results.
    """

    work_dir: Path
    run_id: str

    input_data_path: Path

    train_outputs_dir: Path
    train_run_dir: Path
    train_run_meta_path: Path
    train_run_model_dir: Path

    prediction_outputs_dir: Path
    prediction_run_dir: Path
    probabilities_path: Path
    predictions_path: Path

    def ensure_dirs(
        self,
    ) -> Self:
        """Create prediction artifact directories.

        Returns:
            Updated path bundle.
        """
        self.prediction_run_dir.mkdir(parents=True, exist_ok=True)
        return self


def find_latest_train_run_id(
    train_outputs_dir: Path,
) -> str:
    """Find the most recent completed training run from persisted metadata.

    Args:
        train_outputs_dir: Directory containing training-run subdirectories.

    Returns:
        Run identifier of the latest training run with target-checkpoint fine-tuning enabled.

    Raises:
        FileNotFoundError: If no training-run metadata files exist.
    """
    completed_runs: list[tuple[datetime, str]] = []

    for meta_path in train_outputs_dir.glob(f"*/{TRAIN_RUN_META_FILENAME}"):
        meta = read_run_meta(meta_path)
        if meta.transfer_learning:
            completed_runs.append((meta.created_at, meta.run_id))

    if not completed_runs:
        raise FileNotFoundError(
            "No completed tlmtc training runs found. "
            f"Expected at least one {TRAIN_RUN_META_FILENAME} with transfer_learning=True "
            f"under {train_outputs_dir}."
        )

    return max(completed_runs, key=lambda item: item[0])[1]


def resolve_prediction_paths(
    *,
    input_csv: Path,
    work_dir: Path,
    run_id: str | None,
    train_outputs_dirname: str = DEFAULT_TRAIN_OUTPUTS_DIRNAME,
    prediction_outputs_dirname: str = DEFAULT_PREDICTION_OUTPUTS_DIRNAME,
    model_dirname: str = DEFAULT_MODEL_DIRNAME,
) -> PredictionPaths:
    """Resolve paths for prediction from an existing tlmtc training run.

    Prediction consumes a completed training run under `train_outputs/<run_id>/`
    and writes prediction artifacts under `prediction_outputs/<run_id>/`.

    Args:
        input_csv: Path to the unlabeled prediction input CSV.
        work_dir: Existing base directory containing tlmtc training outputs.
        run_id: Optional training-run identifier. If omitted, the latest completed
            training run is selected from persisted training metadata.
        train_outputs_dirname: Name of the training outputs root directory.
        prediction_outputs_dirname: Name of the prediction outputs root directory.
        model_dirname: Name of the model artifact directory inside a training run.

    Returns:
        Resolved path bundle for prediction.

    Raises:
        FileNotFoundError: If the work directory, prediction input CSV, training
            outputs directory, selected training run, training metadata, or model
            artifact directory is missing.
    """
    resolved_work_dir = work_dir.expanduser().resolve()
    if not resolved_work_dir.is_dir():
        raise FileNotFoundError(f"`work_dir` does not exist: {resolved_work_dir}")

    input_data_path = input_csv.expanduser().resolve()
    if not input_data_path.is_file():
        raise FileNotFoundError(f"Unlabeled prediction input CSV does not exist: {input_data_path}")

    train_outputs_dir = resolved_work_dir / train_outputs_dirname
    if not train_outputs_dir.is_dir():
        raise FileNotFoundError(
            f"No tlmtc training outputs found. Expected an existing training outputs directory at {train_outputs_dir}."
        )

    resolved_run_id = run_id if run_id is not None else find_latest_train_run_id(train_outputs_dir)
    resolved_run_id = validate_run_id(resolved_run_id)

    train_run_dir = train_outputs_dir / resolved_run_id
    if not train_run_dir.is_dir():
        raise FileNotFoundError(
            f"Requested tlmtc training run not found. Expected training run directory at {train_run_dir}."
        )

    train_run_meta_path = train_run_dir / TRAIN_RUN_META_FILENAME
    if not train_run_meta_path.is_file():
        raise FileNotFoundError(
            f"Training run metadata not found. Expected {TRAIN_RUN_META_FILENAME} at {train_run_meta_path}. "
        )

    train_run_model_dir = train_run_dir / model_dirname
    if not train_run_model_dir.is_dir():
        raise FileNotFoundError(
            f"Training model directory not found. Expected model artifacts under {train_run_model_dir}."
        )

    if not any(train_run_model_dir.iterdir()):
        raise FileNotFoundError(
            f"Training model directory is empty. Expected saved model or adapter artifacts under {train_run_model_dir}."
        )

    prediction_outputs_dir = resolved_work_dir / prediction_outputs_dirname
    prediction_run_dir = prediction_outputs_dir / resolved_run_id

    return PredictionPaths(
        work_dir=resolved_work_dir,
        run_id=resolved_run_id,
        input_data_path=input_data_path,
        train_outputs_dir=train_outputs_dir,
        train_run_dir=train_run_dir,
        train_run_meta_path=train_run_meta_path,
        train_run_model_dir=train_run_model_dir,
        prediction_outputs_dir=prediction_outputs_dir,
        prediction_run_dir=prediction_run_dir,
        probabilities_path=prediction_run_dir / "probabilities.csv",
        predictions_path=prediction_run_dir / "predictions.csv",
    )
