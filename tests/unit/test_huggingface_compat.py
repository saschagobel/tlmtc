"""Tests for Hugging Face compatibility helpers."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tlmtc.huggingface_compat import ensure_transformers_default_rope_init_available


@pytest.fixture
def rope_init_functions() -> Iterator[dict[str, Any]]:
    """Provide an isolated view of the Transformers RoPE initializer registry."""
    original = dict(ROPE_INIT_FUNCTIONS)
    try:
        yield ROPE_INIT_FUNCTIONS
    finally:
        ROPE_INIT_FUNCTIONS.clear()
        ROPE_INIT_FUNCTIONS.update(original)


def test_default_rope_compat_noops_when_default_initializer_exists(
    rope_init_functions: dict[str, Any],
) -> None:
    """Ensure official or preexisting default initializers are not replaced."""
    default_initializer = object()
    rope_init_functions["default"] = default_initializer

    installed = ensure_transformers_default_rope_init_available()

    assert installed is False
    assert rope_init_functions["default"] is default_initializer


def test_default_rope_compat_installs_only_missing_default_initializer(
    rope_init_functions: dict[str, Any],
) -> None:
    """Ensure the shim only adds the missing default registry key."""
    linear_initializer = object()
    rope_init_functions.clear()
    rope_init_functions["linear"] = linear_initializer

    installed = ensure_transformers_default_rope_init_available()

    assert installed is True
    assert "default" in rope_init_functions
    assert rope_init_functions["linear"] is linear_initializer


def test_default_rope_compat_computes_inverse_frequencies_from_rope_parameters(
    rope_init_functions: dict[str, Any],
) -> None:
    """Ensure the shim reads theta from config rope parameters."""
    rope_init_functions.pop("default", None)
    ensure_transformers_default_rope_init_available()
    config = SimpleNamespace(
        rope_parameters={"rope_theta": 10000.0},
        hidden_size=8,
        num_attention_heads=1,
    )

    inv_freq, attention_factor = rope_init_functions["default"](config)

    expected = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.int64).float() / 8))
    assert attention_factor == 1.0
    torch.testing.assert_close(inv_freq, expected)


def test_default_rope_compat_computes_inverse_frequencies_from_config_dict(
    rope_init_functions: dict[str, Any],
) -> None:
    """Ensure the shim reads theta from serialized config data."""

    class Config:
        hidden_size = 8
        num_attention_heads = 1

        def to_dict(self) -> dict[str, float]:
            return {
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
            }

    rope_init_functions.pop("default", None)
    ensure_transformers_default_rope_init_available()

    inv_freq, attention_factor = rope_init_functions["default"](Config())

    expected = 1.0 / (1000.0 ** (torch.arange(0, 4, 2, dtype=torch.int64).float() / 4))
    assert attention_factor == 1.0
    torch.testing.assert_close(inv_freq, expected)


def test_default_rope_compat_fails_when_theta_is_missing(
    rope_init_functions: dict[str, Any],
) -> None:
    """Ensure the shim does not silently invent a theta fallback."""
    rope_init_functions.pop("default", None)
    ensure_transformers_default_rope_init_available()
    config = SimpleNamespace(hidden_size=8, num_attention_heads=1)

    with pytest.raises(RuntimeError, match="no rope_theta/default_theta found"):
        rope_init_functions["default"](config)
