"""Resolve and manage run directories.

Builds a `RunPaths` bundle that defines the standard on-disk layout for a single run.
"""

from __future__ import annotations

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
    """Resolved filesystem paths for a single tlmtc run.

    Directory structure:
      <work_dir>/
        train_outputs/
          <run_id>/
            data/     (train/val/test splits)
            logs/  (training/HPO logs)
            model/    (exported model artifacts)
            evaluation/  (metrics, tables, and plots)

    Attributes:
        work_dir: Base directory used to resolve relative inputs and contain tlmtc outputs.
        run_dir: Root directory for this run.
        run_id: Identifier for the run.
        raw_data_path: Resolved path to the raw training CSV.
        raw_test_data_path: Resolved path to the raw test CSV.
        data_dir: Directory containing prepared dataset splits for this run.
        logs_dir: Directory containing logs and HPO artifacts for this run.
        model_dir: Directory containing exported model artifacts for this run.
        eval_dir: Directory containing evaluation reporting artifacts for this run.
        train_data_path: Path to the prepared training split artifact (`train.parquet`).
        val_data_path: Path to the prepared validation split artifact (`val.parquet`).
        test_data_path: Path to the prepared test split artifact (`test.parquet`).
    """

    work_dir: Path
    run_dir: Path
    run_id: str

    raw_data_path: Path
    raw_test_data_path: Path | None

    data_dir: Path
    logs_dir: Path
    model_dir: Path
    eval_dir: Path

    train_data_path: Path
    val_data_path: Path
    test_data_path: Path

    def ensure_dirs(
        self,
    ) -> Self:
        """Create the run directory structure.

        Returns:
            RunPaths
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
    """Resolve all filesystem paths for a single tlmtc run.

    Args:
        raw_csv: Path to the raw training CSV.
        raw_test_csv: Optional path to a raw test CSV.
        work_dir: Optional base directory used to resolve relative inputs and contain outputs.
            Defaults to the current working directory.
        run_id: Identifier for the run.
        outputs_dirname: Name of the outputs root directory under `work_dir`.
        data_dirname: Name of the data subdirectory under `run_dir`.
        logs_dirname: Name of the logs subdirectory under `run_dir`.
        model_dirname: Name of the model subdirectory under `run_dir`.
        eval_dirname: Name of the evaluation subdirectory under `run_dir`.

    Returns:
        RunPaths: Bundle containing absolute paths for inputs and run artifacts.
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
        raw_data_path=raw_data_path,
        raw_test_data_path=raw_test_data_path,
        data_dir=data_dir,
        logs_dir=logs_dir,
        model_dir=model_dir,
        eval_dir=eval_dir,
        train_data_path=data_dir / "train.parquet",
        val_data_path=data_dir / "val.parquet",
        test_data_path=data_dir / "test.parquet",
    )
