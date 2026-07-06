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

_API_EXPORTS = ("predict_tlmtc", "train_tlmtc")


def __getattr__(
    name: str,
) -> Any:
    if name not in _API_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(importlib.import_module("tlmtc.api"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from tlmtc.api import predict_tlmtc as predict_tlmtc
    from tlmtc.api import train_tlmtc as train_tlmtc
