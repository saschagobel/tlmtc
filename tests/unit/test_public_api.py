"""Tests for the tlmtc public API surface."""

import builtins
import re
import sys
from types import ModuleType

import pytest


def _install_dummy_api_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight `tlmtc.api` module so API tests don't import heavy ML deps."""
    mod = ModuleType("tlmtc.api")

    def train_tlmtc(*_args: object, **_kwargs: object) -> str:
        return "ok"

    setattr(mod, "train_tlmtc", train_tlmtc)
    monkeypatch.setitem(sys.modules, "tlmtc.api", mod)


def test_public_api_exports_version() -> None:
    """Tests that the package exposes a non-empty version string."""
    import tlmtc

    assert isinstance(tlmtc.__version__, str)
    assert tlmtc.__version__


def test_public_api_lazy_exports_train_tlmtc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that train_tlmtc is resolved lazily via the public API."""
    _install_dummy_api_module(monkeypatch)

    import tlmtc

    assert tlmtc.train_tlmtc() == "ok"


def test_public_api_rejects_unknown_attribute() -> None:
    """Ensures that unknown public API attributes raise AttributeError."""
    import tlmtc

    with pytest.raises(AttributeError):
        _ = tlmtc.not_a_real_symbol  # type: ignore[attr-defined]


@pytest.mark.parametrize("missing", ["torch", "peft"])
def test_public_api_surfaces_helpful_error_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    """Ensures that a helpful ImportError is raised when training dependencies are missing."""
    import tlmtc

    sys.modules.pop("tlmtc.api", None)

    expected_msg = (
        "`torch` and `peft` are required for `tlmtc.train_tlmtc`. Install them with: `pip install 'tlmtc[training]'`."
    )

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        # The public API imports *tlmtc.api*, not torch/peft directly.
        if name == "tlmtc.api":
            raise ModuleNotFoundError(missing, name=missing)
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match=re.escape(expected_msg)):
            _ = tlmtc.train_tlmtc
