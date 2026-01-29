"""Resolve and manage run directories.

Builds a `RunPaths` bundle that defines the standard on-disk layout for a single run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_OUTPUTS_DIRNAME: Final[str] = "tlmtc_outputs"
DEFAULT_DATA_DIRNAME: Final[str] = "data"
DEFAULT_LOGS_DIRNAME: Final[str] = "logs"
DEFAULT_MODEL_DIRNAME: Final[str] = "model"


def _coerce_path(
    input_path: str | Path,
) -> Path:
    """Coerce a path-like input into a `Path`.

    Args:
        input_path: A filesystem path given as `str` or `Path`.

    Returns:
        Path: The corresponding `Path` object.
    """
    return input_path if isinstance(input_path, Path) else Path(input_path)


def _resolve_under(
    base_dir: Path,
    input_path: Path,
) -> Path:
    """Resolve an input path with `base_dir` as the anchor for relative paths.

    Args:
        base_dir: Base directory used to anchor relative paths.
        input_path: Path to resolve. If relative, it is joined with `base_dir`.
            If absolute, it is resolved as-is.

    Returns:
        Path: An absolute, resolved path.
    """
    input_path = input_path.expanduser()
    return (base_dir / input_path).resolve() if not input_path.is_absolute() else input_path.resolve()


def _default_run_id() -> str:
    """Generate a default run identifier.

    Returns:
        str: A UUID4 hex string suitable for use as a directory name.
    """
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Resolved filesystem paths for a single tlmtc run.

    Directory structure:
      <work_dir>/
        tlmtc_outputs/
          <run_id>/
            data/     (train/val/test splits)
            logs/  (training/HPO logs)
            model/    (exported model artifacts)

    Attributes:
        work_dir: Base directory used to resolve relative inputs and contain tlmtc outputs.
        run_dir: Root directory for this run.
        run_id: Identifier for the run.
        raw_data_path: Resolved path to the raw training CSV.
        raw_test_data_path: Resolved path to the raw test CSV. If no `raw_test_csv` is provided,
            this defaults to `raw_data_path` with filename replaced by `raw_test.csv`.
        data_dir: Directory containing prepared dataset splits for this run.
        logs_dir: Directory containing logs and HPO artifacts for this run.
        model_dir: Directory containing exported model artifacts for this run.
        train_data_path: Path to the prepared training split artifact (`train.parquet`).
        val_data_path: Path to the prepared validation split artifact (`val.parquet`).
        test_data_path: Path to the prepared test split artifact (`test.parquet`).
    """

    work_dir: Path
    run_dir: Path
    run_id: str

    raw_data_path: Path
    raw_test_data_path: Path

    data_dir: Path
    logs_dir: Path
    model_dir: Path

    train_data_path: Path
    val_data_path: Path
    test_data_path: Path

    def ensure_dirs(
        self,
    ) -> RunPaths:
        """Create the run directory structure.

        Returns:
            RunPaths
        """
        for directory in (
            self.run_dir,
            self.data_dir,
            self.logs_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def resolve_paths(
    *,
    raw_csv: str | Path,
    raw_test_csv: str | Path | None = None,
    work_dir: str | Path | None = None,
    run_id: str | None = None,
    outputs_dirname: str = DEFAULT_OUTPUTS_DIRNAME,
    data_dirname: str = DEFAULT_DATA_DIRNAME,
    logs_dirname: str = DEFAULT_LOGS_DIRNAME,
    model_dirname: str = DEFAULT_MODEL_DIRNAME,
) -> RunPaths:
    """Resolve all filesystem paths for a single tlmtc run.

    Intended use:
        paths = resolve_paths(...).ensure_dirs()
        # pass `paths` into pipeline stages

    Args:
        raw_csv: Path to the raw training CSV.
        raw_test_csv: Optional path to a raw test CSV.
        work_dir: Optional base directory used to resolve relative inputs and contain outputs.
            Defaults to the current working directory.
        run_id: Optional identifier for the run.
        outputs_dirname: Name of the outputs root directory under `work_dir`.
        data_dirname: Name of the data subdirectory under `run_dir`.
        logs_dirname: Name of the logs subdirectory under `run_dir`.
        model_dirname: Name of the model subdirectory under `run_dir`.

    Returns:
        RunPaths: Bundle containing absolute paths for inputs and run artifacts.
    """
    base_dir = _coerce_path(work_dir).expanduser().resolve() if work_dir is not None else Path.cwd().resolve()

    raw_data_path = _resolve_under(base_dir, _coerce_path(raw_csv))
    raw_test_data_path = (
        raw_data_path.with_name("raw_test.csv")
        if raw_test_csv is None
        else _resolve_under(base_dir, _coerce_path(raw_test_csv))
    )

    resolved_run_id = run_id or _default_run_id()
    resolved_run_dir = (base_dir / outputs_dirname / resolved_run_id).resolve()

    data_dir = resolved_run_dir / data_dirname
    logs_dir = resolved_run_dir / logs_dirname
    model_dir = resolved_run_dir / model_dirname

    train_data_path = data_dir / "train.parquet"
    val_data_path = data_dir / "val.parquet"
    test_data_path = data_dir / "test.parquet"

    return RunPaths(
        work_dir=base_dir,
        run_dir=resolved_run_dir,
        run_id=resolved_run_id,
        raw_data_path=raw_data_path,
        raw_test_data_path=raw_test_data_path,
        data_dir=data_dir,
        logs_dir=logs_dir,
        model_dir=model_dir,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
    )
