"""Tests for the tlmtc public API surface."""

import importlib
import logging
import re
import sys
from types import ModuleType

import pytest

PUBLIC_ENTRYPOINTS: tuple[str, ...] = (
    "predict_tlmtc",
    "train_tlmtc",
)

OPTIONAL_DEPENDENCIES: tuple[str, ...] = (
    "torch",
    "peft",
    "accelerate",
)


def _reset_public_api_import_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear cached lazy public API imports."""
    import tlmtc

    monkeypatch.delitem(sys.modules, "tlmtc.api", raising=False)
    monkeypatch.delitem(sys.modules, "tlmtc.api.predict", raising=False)
    monkeypatch.delitem(sys.modules, "tlmtc.api.train", raising=False)
    monkeypatch.delitem(tlmtc.__dict__, "api", raising=False)

    for name in PUBLIC_ENTRYPOINTS:
        monkeypatch.delitem(tlmtc.__dict__, name, raising=False)


def _install_dummy_api_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight API modules so API tests do not import heavy ML deps."""
    predict_mod = ModuleType("tlmtc.api.predict")
    train_mod = ModuleType("tlmtc.api.train")

    def predict_tlmtc(*_args: object, **_kwargs: object) -> str:
        return "predict-ok"

    def train_tlmtc(*_args: object, **_kwargs: object) -> str:
        return "train-ok"

    predict_mod.predict_tlmtc = predict_tlmtc  # type: ignore[attr-defined]
    train_mod.train_tlmtc = train_tlmtc  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tlmtc.api.predict", predict_mod)
    monkeypatch.setitem(sys.modules, "tlmtc.api.train", train_mod)


def test_public_api_exports_version() -> None:
    """Test that the package exposes a non-empty version string."""
    import tlmtc

    assert isinstance(tlmtc.__version__, str)
    assert tlmtc.__version__


def test_public_api_declares_expected_exports() -> None:
    """Test that the package declares the expected public API symbols."""
    import tlmtc

    assert set(tlmtc.__all__) == {
        "__version__",
        "predict_tlmtc",
        "train_tlmtc",
    }


def test_public_api_attaches_package_null_handler() -> None:
    """Test that importing tlmtc installs an inert package logger handler."""
    import tlmtc

    package_logger = logging.getLogger(tlmtc.__name__)

    assert any(isinstance(handler, logging.NullHandler) for handler in package_logger.handlers)


def test_public_api_dir_lists_declared_exports() -> None:
    """Test that dir(tlmtc) exposes the declared public API symbols."""
    import tlmtc

    assert dir(tlmtc) == sorted(tlmtc.__all__)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("predict_tlmtc", "predict-ok"),
        ("train_tlmtc", "train-ok"),
    ],
)
def test_public_api_lazy_exports_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    expected: str,
) -> None:
    """Test that public entrypoints are resolved lazily via the package API."""
    _reset_public_api_import_state(monkeypatch)
    _install_dummy_api_module(monkeypatch)

    import tlmtc

    entrypoint = getattr(tlmtc, name)

    assert entrypoint() == expected
    assert tlmtc.__dict__[name] is entrypoint


def test_public_api_rejects_unknown_attribute() -> None:
    """Ensure that unknown public API attributes raise AttributeError."""
    import tlmtc

    with pytest.raises(AttributeError):
        _ = tlmtc.not_a_real_symbol  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", PUBLIC_ENTRYPOINTS)
@pytest.mark.parametrize("missing", OPTIONAL_DEPENDENCIES)
def test_public_api_surfaces_helpful_error_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    missing: str,
) -> None:
    """Ensure lazy entrypoints raise a helpful ImportError when optional deps are missing."""
    _reset_public_api_import_state(monkeypatch)

    import tlmtc

    extra = name.removesuffix("_tlmtc")
    expected_msg = (
        f"Optional dependencies are required for `{name}`. Install them with: `pip install 'tlmtc[{extra}]'`."
    )

    real_import_module = importlib.import_module
    target_module_path = f"tlmtc.api.{name.removesuffix('_tlmtc')}"

    def fake_import_module(module_path: str, package: str | None = None) -> ModuleType:
        if module_path == target_module_path:
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        return real_import_module(module_path, package)

    monkeypatch.setattr(tlmtc.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=re.escape(expected_msg)):
        _ = getattr(tlmtc, name)


def test_public_api_reraises_unexpected_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure unexpected missing modules are not rewritten as optional-dependency errors."""
    _reset_public_api_import_state(monkeypatch)

    import tlmtc

    real_import_module = importlib.import_module

    def fake_import_module(module_path: str, package: str | None = None) -> ModuleType:
        if module_path == "tlmtc.api.train":
            raise ModuleNotFoundError("No module named 'some_other_dependency'", name="some_other_dependency")
        return real_import_module(module_path, package)

    monkeypatch.setattr(tlmtc.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="some_other_dependency"):
        _ = tlmtc.train_tlmtc
