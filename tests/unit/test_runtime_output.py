"""Tests for runtime console output policy."""

import logging
from collections.abc import Iterator
from unittest.mock import MagicMock

import optuna
import pytest
from transformers import Trainer
from transformers.trainer_callback import PrinterCallback, ProgressCallback

import tlmtc.runtime_output as runtime_output


@pytest.fixture(autouse=True)
def restore_progress_logger() -> Iterator[None]:
    """Restore the package progress logger after each test."""
    logger = logging.getLogger(runtime_output.PROGRESS_LOGGER_NAME)

    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    old_disabled = logger.disabled

    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    logger.disabled = False

    yield

    logger.handlers.clear()
    logger.handlers.extend(old_handlers)
    logger.setLevel(old_level)
    logger.propagate = old_propagate
    logger.disabled = old_disabled


def test_configure_runtime_output_applies_suppression_and_configures_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure runtime output configuration delegates to suppression and progress setup."""
    calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        runtime_output,
        "_apply_third_party_suppression",
        lambda: calls.append(("suppression", None)),
    )
    monkeypatch.setattr(
        runtime_output,
        "_configure_progress_logger",
        lambda verbosity: calls.append(("logger", verbosity)),
    )

    runtime_output.configure_runtime_output("quiet")

    assert calls == [
        ("suppression", None),
        ("logger", "quiet"),
    ]


def test_apply_third_party_suppression_calls_official_suppression_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure third-party routine output is suppressed through official library controls."""
    calls: list[object] = []

    monkeypatch.setattr(
        runtime_output.transformers_logging,
        "set_verbosity_error",
        lambda: calls.append("transformers_logging"),
    )
    monkeypatch.setattr(
        runtime_output.transformers_logging,
        "disable_progress_bar",
        lambda: calls.append("transformers_progress"),
    )
    monkeypatch.setattr(
        runtime_output.datasets.logging,
        "set_verbosity_error",
        lambda: calls.append("datasets_logging"),
    )
    monkeypatch.setattr(
        runtime_output.datasets,
        "disable_progress_bars",
        lambda: calls.append("datasets_progress"),
    )
    monkeypatch.setattr(
        runtime_output.huggingface_hub.logging,
        "set_verbosity_error",
        lambda: calls.append("hub_logging"),
    )
    monkeypatch.setattr(
        runtime_output.huggingface_hub.utils,
        "disable_progress_bars",
        lambda: calls.append("hub_progress"),
    )
    monkeypatch.setattr(
        runtime_output.optuna.logging,
        "set_verbosity",
        lambda level: calls.append(("optuna_logging", level)),
    )

    runtime_output._apply_third_party_suppression()

    assert calls == [
        "transformers_logging",
        "transformers_progress",
        "datasets_logging",
        "datasets_progress",
        "hub_logging",
        "hub_progress",
        ("optuna_logging", optuna.logging.WARNING),
    ]


@pytest.mark.parametrize(
    ("verbosity", "expected_stderr"),
    [
        ("progress", "tlmtc: Preparing data\n"),
        ("quiet", ""),
    ],
)
def test_configure_progress_logger_controls_progress_emission(
    verbosity: str,
    expected_stderr: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure runtime verbosity controls package-owned progress messages."""
    runtime_output._configure_progress_logger(verbosity)  # type: ignore[arg-type]

    runtime_output.emit_progress("Preparing data")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == expected_stderr


def test_configure_progress_logger_is_idempotent() -> None:
    """Ensure repeated progress configuration does not add duplicate handlers."""
    runtime_output._configure_progress_logger("progress")
    runtime_output._configure_progress_logger("progress")

    logger = logging.getLogger(runtime_output.PROGRESS_LOGGER_NAME)

    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert logger.disabled is False


def test_configure_progress_logger_reenables_after_quiet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure progress mode re-enables the progress logger after quiet mode."""
    runtime_output._configure_progress_logger("quiet")
    runtime_output._configure_progress_logger("progress")

    runtime_output.emit_progress("Tokenizing inputs")

    captured = capsys.readouterr()
    assert captured.err == "tlmtc: Tokenizing inputs\n"


def test_configure_progress_logger_rejects_invalid_verbosity() -> None:
    """Ensure unsupported runtime verbosity values are rejected."""
    with pytest.raises(ValueError, match=r"Unsupported runtime verbosity"):
        runtime_output._configure_progress_logger("verbose")  # type: ignore[arg-type]


def test_suppress_trainer_console_callbacks_removes_console_callbacks() -> None:
    """Ensure Trainer console callbacks are removed via the Trainer callback API."""
    trainer = MagicMock(spec=Trainer)

    result = runtime_output.suppress_trainer_console_callbacks(trainer)

    assert result is trainer
    trainer.remove_callback.assert_any_call(PrinterCallback)
    trainer.remove_callback.assert_any_call(ProgressCallback)
    assert trainer.remove_callback.call_count == 2
