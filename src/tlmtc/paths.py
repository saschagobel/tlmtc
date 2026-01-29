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
DEFAULT_LOGGING_DIRNAME: Final[str] = "logs"
DEFAULT_MODEL_DIRNAME: Final[str] = "model"


def _coerce_path(
    input_path: str | Path,
) -> Path:
    """... .

    Args:
        input_path: ...

    Returns:
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

    Returns:
        ...
    """
    input_path = input_path.expanduser()
    return (base_dir / input_path).resolve() if not input_path.is_absolute() else input_path.resolve()


def _default_run_id() -> str:
    """... .

    Returns:
        ...
    """
    return uuid.uuid4().hex
