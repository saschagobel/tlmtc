"""Resolve and manage run directories.

Builds a `RunPaths` bundle that defines the standard on-disk layout for a single run.

Default layout:
  <work_dir>/
    tlmtc_outputs/
      <run_id>/
        data/     (train/val/test splits)
        logging/  (training/HPO logs)
        model/    (exported model artifacts)

Intended use: call `resolve_paths(...)` once in the runner, then `paths.ensure_dirs()`,
and pass the resulting `RunPaths` to pipeline stages.
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
    """... .

    Args:
        input_path: ...

    Returns
    -------
        ...
    """
    return input_path if isinstance(input_path, Path) else Path(input_path)


def _resolve_under(
    base_dir: Path,
    input_path: Path,
) -> Path:
    """... .

    Args:
        base_dir: ...
        input_path: ...

    Returns
    -------
        ...
    """
    input_path = input_path.expanduser()
    return (base_dir / input_path).resolve() if not input_path.is_absolute() else input_path.resolve()


def _default_run_id() -> str:
    """... .

    Returns
    -------
        ...
    """
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class RunPaths:
    """... .

    ... .

    Attributes
    ----------
        ...
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
        """... ."""
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
    """... .

    Args:
        : ...
        : ...

    Returns
    -------
        ...
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
