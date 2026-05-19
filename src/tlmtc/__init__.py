"""Public package interface and lazy-loaded API exports."""

import importlib
import logging
from importlib import metadata
from typing import TYPE_CHECKING, Any

try:
    __version__ = metadata.version("tlmtc")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "predict_tlmtc",
    "train_tlmtc",
    "__version__",
]

logging.getLogger("tlmtc").addHandler(logging.NullHandler())

_LAZY: dict[str, tuple[str, str]] = {
    "predict_tlmtc": ("tlmtc.api", "predict_tlmtc"),
    "train_tlmtc": ("tlmtc.api", "train_tlmtc"),
}


def __getattr__(
    name: str,
) -> Any:
    try:
        module_path, attr = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    try:
        value = getattr(importlib.import_module(module_path), attr)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)

        if missing in {"torch", "peft", "accelerate"}:
            raise ImportError(
                f"`torch`, `peft`, and `accelerate` are required for `tlmtc.{name}`. "
                "Install them with: `pip install 'tlmtc[full]'`."
            ) from exc

        raise

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from tlmtc.api import predict_tlmtc as predict_tlmtc
    from tlmtc.api import train_tlmtc as train_tlmtc
