"""Public Python API for running tlmtc workflows."""

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "PredictResult",
    "TrainResult",
    "predict_tlmtc",
    "train_tlmtc",
]

_LAZY: dict[str, tuple[str, str]] = {
    "PredictResult": ("tlmtc.api.predict", "PredictResult"),
    "TrainResult": ("tlmtc.api.train", "TrainResult"),
    "predict_tlmtc": ("tlmtc.api.predict", "predict_tlmtc"),
    "train_tlmtc": ("tlmtc.api.train", "train_tlmtc"),
}

_OPTIONAL_DEPENDENCIES = (
    "accelerate",
    "datasets",
    "great_tables",
    "huggingface_hub",
    "iterstrat",
    "matplotlib",
    "numpy",
    "optuna",
    "pandera",
    "peft",
    "pyarrow",
    "seaborn",
    "sklearn",
    "torch",
    "transformers",
)
_INSTALL_EXTRAS = {
    "PredictResult": "predict",
    "TrainResult": "train",
    "predict_tlmtc": "predict",
    "train_tlmtc": "train",
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

        if missing in _OPTIONAL_DEPENDENCIES:
            extra = _INSTALL_EXTRAS[name]
            raise ImportError(
                f"Optional dependencies are required for `{name}`. Install them with: `pip install 'tlmtc[{extra}]'`."
            ) from exc

        raise

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from tlmtc.api.predict import PredictResult as PredictResult
    from tlmtc.api.predict import predict_tlmtc as predict_tlmtc
    from tlmtc.api.train import TrainResult as TrainResult
    from tlmtc.api.train import train_tlmtc as train_tlmtc
