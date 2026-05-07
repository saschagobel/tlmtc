"""Filesystem path layout for tlmtc training runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

DEFAULT_OUTPUTS_DIRNAME: Final[str] = "train_outputs"
DEFAULT_DATA_DIRNAME: Final[str] = "data"
DEFAULT_LOGS_DIRNAME: Final[str] = "logs"
DEFAULT_MODEL_DIRNAME: Final[str] = "model"
DEFAULT_EVAL_DIRNAME: Final[str] = "evaluation"


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
        model_dir: Directory for saved model artifacts.
        eval_dir: Directory for evaluation metrics, tables, and figures.
        train_data_path: Path to the prepared training split.
        val_data_path: Path to the prepared validation split.
        test_data_path: Path to the prepared test split.
        optuna_trials_path: Path to the persisted Optuna study database.
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
    model_dir: Path
    eval_dir: Path

    train_data_path: Path
    val_data_path: Path
    test_data_path: Path

    optuna_trials_path: Path

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
            self.logs_dir,
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
    outputs_dirname: str = DEFAULT_OUTPUTS_DIRNAME,
    data_dirname: str = DEFAULT_DATA_DIRNAME,
    logs_dirname: str = DEFAULT_LOGS_DIRNAME,
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
        model_dirname: Name of the model directory under `run_dir`.
        eval_dirname: Name of the evaluation directory under `run_dir`.

    Returns:
        Resolved path bundle for inputs and run artifacts.
    """
    resolved_work_dir = work_dir.expanduser().resolve()

    raw_data_path = raw_csv.expanduser().resolve()
    raw_test_data_path = None if raw_test_csv is None else raw_test_csv.expanduser().resolve()

    run_dir = resolved_work_dir / outputs_dirname / run_id
    data_dir = run_dir / data_dirname
    logs_dir = run_dir / logs_dirname
    model_dir = run_dir / model_dirname
    eval_dir = run_dir / eval_dirname

    return RunPaths(
        work_dir=resolved_work_dir,
        run_dir=run_dir,
        run_id=run_id,
        train_run_meta_path=run_dir / "train_run_meta.json",
        raw_data_path=raw_data_path,
        raw_test_data_path=raw_test_data_path,
        data_dir=data_dir,
        logs_dir=logs_dir,
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
    )
