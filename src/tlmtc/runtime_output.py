"""Runtime console output policy for tlmtc workflows."""

import logging
import sys
from typing import Final, Literal

import datasets
import huggingface_hub
import optuna
from transformers import Trainer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.utils import logging as transformers_logging

PROGRESS_LOGGER_NAME: Final[str] = "tlmtc.progress"

_PROGRESS_LOGGER = logging.getLogger(PROGRESS_LOGGER_NAME)


def configure_runtime_output(
    verbosity: Literal["progress", "quiet"],
    *,
    is_main_process: bool = True,
) -> None:
    """Configure runtime console behavior for a tlmtc workflow.

    Args:
        verbosity: Runtime output mode.
        is_main_process: Whether package-owned progress output should be emitted.
    """
    _apply_third_party_suppression()
    _configure_progress_logger(
        verbosity=verbosity,
        is_main_process=is_main_process,
    )


def emit_progress(
    message: str,
) -> None:
    """Emit a package-owned progress message if progress output is enabled.

    Args:
        message: User-facing workflow progress message.
    """
    _PROGRESS_LOGGER.info(message)


def suppress_trainer_console_callbacks(
    trainer: Trainer,
) -> Trainer:
    """Remove Trainer callbacks responsible for routine console output.

    Args:
        trainer: Hugging Face Trainer instance to update.

    Returns:
        The same Trainer instance with console callbacks removed.
    """
    for callback_type in (PrinterCallback, ProgressCallback):
        trainer.remove_callback(callback_type)

    return trainer


def _apply_third_party_suppression() -> None:
    """Suppress routine console output from third-party ML libraries."""
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    # Suppress benign Transformers checkpoint-ordering fallback noise on filesystems
    # where checkpoint mtimes are not reliable
    transformers_logger = logging.getLogger("transformers.trainer_utils")
    transformers_logger.addFilter(
        lambda record: "mtime may not be reliable on this filesystem" not in record.getMessage()
    )

    datasets.logging.set_verbosity_error()
    datasets.disable_progress_bars()

    huggingface_hub.logging.set_verbosity_error()
    huggingface_hub.utils.disable_progress_bars()  # type: ignore[attr-defined]

    optuna.logging.set_verbosity(optuna.logging.WARNING)


def _configure_progress_logger(
    verbosity: Literal["progress", "quiet"],
    *,
    is_main_process: bool = True,
) -> None:
    """Configure the package-owned progress logger idempotently.

    Args:
        verbosity: Runtime output mode.
        is_main_process: Whether progress output should be enabled for this process.

    Raises:
        ValueError: If an unsupported verbosity value is passed.
    """
    if verbosity not in {"progress", "quiet"}:
        raise ValueError(f"Unsupported runtime verbosity: {verbosity!r}. Use 'progress' or 'quiet'.")

    handler_marker = "_tlmtc_progress_handler"

    _PROGRESS_LOGGER.setLevel(logging.INFO)
    _PROGRESS_LOGGER.propagate = False
    _PROGRESS_LOGGER.disabled = verbosity == "quiet" or not is_main_process

    if verbosity == "progress" and is_main_process and not any(
        getattr(handler, handler_marker, False) for handler in _PROGRESS_LOGGER.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("tlmtc: %(message)s"))
        setattr(handler, handler_marker, True)
        _PROGRESS_LOGGER.addHandler(handler)
