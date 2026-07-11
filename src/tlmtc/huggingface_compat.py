"""Compatibility helpers for narrow Hugging Face runtime gaps."""

from typing import Any


def ensure_transformers_default_rope_init_available() -> bool:
    """Install a missing Transformers default RoPE initializer.

    Some Transformers 5 releases miss ``ROPE_INIT_FUNCTIONS["default"]`` while
    some remote model code still indexes that registry key directly.

    Returns:
        Whether the compatibility initializer was installed.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return False

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    return True


def _compute_default_rope_parameters(
    config: Any,
    device: Any = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[Any, float]:
    """Compute default RoPE parameters in the format expected by Transformers."""
    import torch

    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if layer_type is not None and layer_type in rope_parameters:
        rope_parameters = rope_parameters[layer_type]

    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    base = _get_rope_theta(
        config=config,
        config_dict=config_dict,
        rope_parameters=rope_parameters,
    )
    partial_rotary_factor = rope_parameters.get(
        "partial_rotary_factor",
        getattr(config, "partial_rotary_factor", config_dict.get("partial_rotary_factor", 1.0)),
    )
    if partial_rotary_factor is None:
        partial_rotary_factor = 1.0

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _get_rope_theta(
    *,
    config: Any,
    config_dict: dict[str, Any],
    rope_parameters: dict[str, Any],
) -> Any:
    """Resolve RoPE theta without silently inventing a fallback."""
    if "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    if "rope_theta" in config_dict:
        return config_dict["rope_theta"]
    if hasattr(config, "rope_theta"):
        return config.rope_theta
    if hasattr(config, "default_theta"):
        return config.default_theta

    raise RuntimeError("Cannot compute default RoPE parameters: no rope_theta/default_theta found on config.")
