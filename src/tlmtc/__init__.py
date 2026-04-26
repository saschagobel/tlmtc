"""tlmtc package."""

from typing import TYPE_CHECKING, Any

__version__ = "0.0.1"

__all__ = ["train_tlmtc", "__version__"]

if TYPE_CHECKING:
    from tlmtc.api import train_tlmtc


def __getattr__(
    name: str,
) -> Any:
    if name != "train_tlmtc":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from tlmtc.api import train_tlmtc
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)

        if missing == "torch":
            raise ImportError(
                "Using tlmtc.train_tlmtc requires PyTorch, but it was not found in your environment."
            ) from exc

        if missing == "peft":
            raise ImportError("PEFT support was requested, but `peft` is not installed.") from exc

        raise

    return train_tlmtc
