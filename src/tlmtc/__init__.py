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

        if missing in {"torch", "peft"}:
            raise ImportError(
                "`torch` and `peft` are required for `tlmtc.train_tlmtc`. "
                "Install them with: `pip install 'tlmtc[training]'`."
            ) from exc

        raise

    return train_tlmtc
