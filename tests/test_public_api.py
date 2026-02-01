"""Tests for the tlmtc public API surface."""

import builtins
import re
import sys
from types import ModuleType

import pytest


def _install_dummy_run_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight `tlmtc.run` module so API tests don't import heavy ML deps."""
    mod = ModuleType("tlmtc.run")

    def run_tlmtc(*_args: object, **_kwargs: object) -> str:
        return "ok"

    setattr(mod, "run_tlmtc", run_tlmtc)
    monkeypatch.setitem(sys.modules, "tlmtc.run", mod)


def test_public_api_exports_version() -> None:
    """Tests that the package exposes a non-empty version string."""
    import tlmtc

    assert isinstance(tlmtc.__version__, str)
    assert tlmtc.__version__


def test_public_api_lazy_exports_run_tlmtc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that run_tlmtc is resolved lazily via the public API."""
    _install_dummy_run_module(monkeypatch)

    import tlmtc

    assert tlmtc.run_tlmtc() == "ok"


def test_public_api_rejects_unknown_attribute() -> None:
    """Ensures that unknown public API attributes raise AttributeError."""
    import tlmtc

    with pytest.raises(AttributeError):
        _ = tlmtc.not_a_real_symbol  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("missing", "expected_msg"),
    [
        (
            "torch",
            "Using tlmtc.run_tlmtc requires PyTorch, but it was not found in your environment.",
        ),
        (
            "peft",
            "PEFT support was requested, but `peft` is not installed.",
        ),
    ],
)
def test_public_api_surfaces_helpful_error_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    expected_msg: str,
) -> None:
    """Ensures that a helpful ImportError is raised when an optional dependency is missing."""
    import tlmtc

    sys.modules.pop("tlmtc.run", None)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        # The public API imports *tlmtc.run*, not torch/peft directly.
        if name == "tlmtc.run":
            raise ModuleNotFoundError(missing, name=missing)
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match=re.escape(expected_msg)):
            _ = tlmtc.run_tlmtc
