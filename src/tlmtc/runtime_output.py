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
) -> None:
    """Configure runtime console behavior for a tlmtc workflow.

    Args:
        verbosity: Runtime output mode.
    """
    _apply_third_party_suppression()
    _configure_progress_logger(verbosity=verbosity)


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

    datasets.logging.set_verbosity_error()
    datasets.disable_progress_bars()

    huggingface_hub.logging.set_verbosity_error()
    huggingface_hub.utils.disable_progress_bars()  # type: ignore[attr-defined]

    optuna.logging.set_verbosity(optuna.logging.WARNING)


def _configure_progress_logger(
    verbosity: Literal["progress", "quiet"],
) -> None:
    """Configure the package-owned progress logger idempotently.

    Args:
        verbosity: Runtime output mode.

    Raises:
        ValueError: If an unsupported verbosity value is passed.
    """
    if verbosity not in {"progress", "quiet"}:
        raise ValueError(f"Unsupported runtime verbosity: {verbosity!r}. Use 'progress' or 'quiet'.")

    handler_marker = "_tlmtc_progress_handler"

    _PROGRESS_LOGGER.setLevel(logging.INFO)
    _PROGRESS_LOGGER.propagate = False
    _PROGRESS_LOGGER.disabled = verbosity == "quiet"

    if verbosity == "progress" and not any(
        getattr(handler, handler_marker, False) for handler in _PROGRESS_LOGGER.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("tlmtc: %(message)s"))
        setattr(handler, handler_marker, True)
        _PROGRESS_LOGGER.addHandler(handler)
