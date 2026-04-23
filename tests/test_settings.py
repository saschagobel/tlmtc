"""Tests for layered settings resolution infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from tlmtc.settings import (
    UNSET,
    HardwareSettings,
    ModelSettings,
    PeftSettings,
    ResolvableSettings,
    SplitSettings,
    ThresholdSettings,
    TrainingSettings,
    WorkflowSettings,
    deep_merge,
    load_config_file,
    prune_unset,
)


class NestedSettings(BaseModel):
    """Nested test settings model."""

    model_config = ConfigDict(extra="forbid")

    alpha: int
    beta: int = 0


class ExampleSettings(ResolvableSettings):
    """Example settings model used to test layered resolution."""

    foo: int
    bar: str = "default"
    nested: NestedSettings


def test_unset_has_stable_repr() -> None:
    """UNSET should render predictably for debugging."""
    assert repr(UNSET) == "UNSET"


def test_unset_has_no_truth_value() -> None:
    """UNSET should reject implicit truthiness checks."""
    with pytest.raises(TypeError, match="UNSET has no truth value"):
        bool(UNSET)


@pytest.mark.parametrize(
    ("base", "incoming", "expected"),
    [
        (
            {"a": 1, "nested": {"x": 1, "y": 2}},
            {"b": 2, "nested": {"y": 99, "z": 3}},
            {"a": 1, "b": 2, "nested": {"x": 1, "y": 99, "z": 3}},
        ),
        (
            {"nested": {"x": 1}, "scalar": 1},
            {"nested": 5, "scalar": {"y": 2}},
            {"nested": 5, "scalar": {"y": 2}},
        ),
        (
            {},
            {"foo": 1},
            {"foo": 1},
        ),
    ],
)
def test_deep_merge(base: dict[str, object], incoming: dict[str, object], expected: dict[str, object]) -> None:
    """deep_merge should recursively merge higher-precedence settings."""
    merged = deep_merge(base, incoming)
    assert merged == expected


def test_deep_merge_does_not_mutate_inputs() -> None:
    """deep_merge should not mutate either input mapping."""
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    incoming = {"b": 2, "nested": {"y": 99, "z": 3}}

    _ = deep_merge(base, incoming)

    assert base == {"a": 1, "nested": {"x": 1, "y": 2}}
    assert incoming == {"b": 2, "nested": {"y": 99, "z": 3}}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {
                "a": 1,
                "b": UNSET,
                "nested": {
                    "c": 2,
                    "d": UNSET,
                    "items": [1, UNSET, {"x": 3, "y": UNSET}],
                },
                "items": [UNSET, 4, {"z": UNSET, "w": 5}],
            },
            {
                "a": 1,
                "nested": {
                    "c": 2,
                    "items": [1, {"x": 3}],
                },
                "items": [4, {"w": 5}],
            },
        ),
        (3, 3),
        ("hello", "hello"),
        (None, None),
        ([UNSET, 1, {"a": UNSET, "b": 2}], [1, {"b": 2}]),
    ],
)
def test_prune_unset(value: object, expected: object) -> None:
    """prune_unset should recursively remove UNSET values."""
    assert prune_unset(value) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            """
foo: 1
bar: hello
nested:
  alpha: 10
  beta: 20
""".strip(),
            {
                "foo": 1,
                "bar": "hello",
                "nested": {
                    "alpha": 10,
                    "beta": 20,
                },
            },
        ),
        ("", {}),
        ("   \n\n   ", {}),
        ("# comment only\n# another comment\n", {}),
    ],
)
def test_load_config_file_reads_yaml_mapping(
    tmp_path: Path,
    content: str,
    expected: dict[str, object],
) -> None:
    """load_config_file should return parsed YAML mapping data."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")

    assert load_config_file(config_path) == expected


def test_load_config_file_raises_for_missing_file(tmp_path: Path) -> None:
    """load_config_file should fail clearly for missing config files."""
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        load_config_file(missing)


def test_load_config_file_raises_for_non_mapping_yaml_root(tmp_path: Path) -> None:
    """load_config_file should reject YAML files whose root is not a mapping."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Config file root must be a mapping"):
        load_config_file(config_path)


def test_resolve_uses_config_only() -> None:
    """Resolve should validate config-only input."""
    settings = ExampleSettings.resolve(
        config={
            "foo": 1,
            "nested": {"alpha": 10},
        }
    )

    assert settings.foo == 1
    assert settings.bar == "default"
    assert settings.nested.alpha == 10
    assert settings.nested.beta == 0


def test_resolve_applies_precedence_config_then_env_then_overrides() -> None:
    """Resolve should apply layers in increasing precedence order."""
    settings = ExampleSettings.resolve(
        config={
            "foo": 1,
            "bar": "from-config",
            "nested": {"alpha": 10, "beta": 20},
        },
        env={
            "bar": "from-env",
            "nested": {"beta": 30},
        },
        overrides={
            "foo": 2,
            "nested": {"alpha": 99},
        },
    )

    assert settings.foo == 2
    assert settings.bar == "from-env"
    assert settings.nested.alpha == 99
    assert settings.nested.beta == 30


def test_resolve_prunes_unset_override_values() -> None:
    """Resolve should ignore override values explicitly marked as UNSET."""
    settings = ExampleSettings.resolve(
        config={
            "foo": 1,
            "bar": "from-config",
            "nested": {"alpha": 10, "beta": 20},
        },
        overrides={
            "foo": UNSET,
            "bar": UNSET,
            "nested": {"alpha": UNSET, "beta": 99},
        },
    )

    assert settings.foo == 1
    assert settings.bar == "from-config"
    assert settings.nested.alpha == 10
    assert settings.nested.beta == 99


def test_resolve_rejects_extra_keys() -> None:
    """Resolve should respect extra='forbid' on the resolved model."""
    with pytest.raises(ValidationError):
        ExampleSettings.resolve(
            config={
                "foo": 1,
                "nested": {"alpha": 10},
                "unknown": "boom",
            }
        )


def test_resolve_requires_required_fields() -> None:
    """Resolve should fail validation when required fields are missing."""
    with pytest.raises(ValidationError):
        ExampleSettings.resolve(
            config={
                "bar": "hello",
                "nested": {"alpha": 10},
            }
        )


def test_resolve_rejects_extra_keys_in_nested_models() -> None:
    """Resolve should also enforce extra='forbid' for nested models."""
    with pytest.raises(ValidationError):
        ExampleSettings.resolve(
            config={
                "foo": 1,
                "nested": {
                    "alpha": 10,
                    "gamma": 99,
                },
            }
        )


def test_model_settings_defaults() -> None:
    """ModelSettings should expose the package defaults."""
    settings = ModelSettings()

    assert settings.target_name == "Target"
    assert settings.proxy_checkpoint == "microsoft/deberta-v3-xsmall"
    assert settings.checkpoint == "microsoft/deberta-v3-base"
    assert settings.sequence_length == 128


def test_split_settings_defaults() -> None:
    """SplitSettings should expose the package defaults."""
    settings = SplitSettings()

    assert settings.validation_size == 0.15
    assert settings.test_size == 0.15
    assert settings.random_seed == 2469


def test_workflow_settings_defaults() -> None:
    """WorkflowSettings should expose the package defaults."""
    settings = WorkflowSettings()

    assert settings.hyperparameter_tuning is True
    assert settings.threshold_optimization is True
    assert settings.transfer_learning is True
    assert settings.scale_learning_rate is False
    assert settings.wrap_peft is True


def test_training_settings_defaults() -> None:
    """TrainingSettings should expose the package defaults."""
    settings = TrainingSettings()

    assert settings.batch_size == 16
    assert settings.train_epochs == 20
    assert settings.weight_decay == 0.01
    assert settings.learning_rate == 2e-5
    assert settings.lr_scheduler == "linear"
    assert settings.best_model_metric == "roc_auc_macro"
    assert settings.early_stopping_patience == 10


def test_threshold_settings_defaults() -> None:
    """ThresholdSettings should expose the package defaults."""
    settings = ThresholdSettings()

    assert settings.threshold_type == "label"
    assert settings.best_threshold_metric == "f1_macro"


def test_peft_settings_defaults() -> None:
    """PeftSettings should expose the package defaults."""
    settings = PeftSettings()

    assert settings.lora_r == 8
    assert settings.lora_alpha == 32
    assert settings.lora_dropout == 0.1
    assert settings.lora_bias == "none"


def test_hardware_settings_defaults() -> None:
    """HardwareSettings should expose the package defaults."""
    settings = HardwareSettings()

    assert settings.use_cpu is False


@pytest.mark.parametrize(
    "factory",
    [
        ModelSettings,
        SplitSettings,
        WorkflowSettings,
        ThresholdSettings,
        PeftSettings,
        HardwareSettings,
    ],
)
def test_frozen_settings_reject_assignment(factory: type[BaseModel]) -> None:
    """Frozen settings bundles should reject mutation."""
    settings = factory()

    field_name = next(iter(settings.model_fields))
    current_value = getattr(settings, field_name)

    with pytest.raises(ValidationError):
        setattr(settings, field_name, current_value)


def test_training_settings_allows_valid_assignment() -> None:
    """TrainingSettings should validate successful assignment updates."""
    settings = TrainingSettings()

    settings.learning_rate = 1e-4
    settings.batch_size = 32
    settings.train_epochs = 5

    assert settings.learning_rate == 1e-4
    assert settings.batch_size == 32
    assert settings.train_epochs == 5


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("batch_size", 0),
        ("train_epochs", 0),
        ("early_stopping_patience", 0),
    ],
)
def test_training_settings_rejects_invalid_assignment(field_name: str, value: object) -> None:
    """TrainingSettings should validate assignment-time updates."""
    settings = TrainingSettings()

    with pytest.raises(ValidationError):
        setattr(settings, field_name, value)


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"sequence_length": 0}, "sequence_length"),
        ({"validation_size": 1.0}, "validation_size"),
        ({"test_size": 0.0}, "test_size"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"lora_dropout": 1.0}, "lora_dropout"),
        ({"lora_r": 0}, "lora_r"),
    ],
)
def test_settings_bundles_reject_invalid_values(kwargs: dict[str, object], pattern: str) -> None:
    """Representative bundle validations should fail clearly for invalid values."""
    model_map: dict[str, type[BaseModel]] = {
        "sequence_length": ModelSettings,
        "validation_size": SplitSettings,
        "test_size": SplitSettings,
        "learning_rate": TrainingSettings,
        "weight_decay": TrainingSettings,
        "lora_dropout": PeftSettings,
        "lora_r": PeftSettings,
    }

    field_name = next(iter(kwargs))
    model_cls = model_map[field_name]

    with pytest.raises(ValidationError, match=pattern):
        model_cls(**kwargs)


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (ModelSettings, {"unknown": 1}),
        (TrainingSettings, {"unknown": "boom"}),
        (PeftSettings, {"unknown": True}),
    ],
)
def test_settings_bundles_reject_extra_keys(
    model_cls: type[BaseModel],
    kwargs: dict[str, object],
) -> None:
    """Settings bundles should enforce extra='forbid'."""
    with pytest.raises(ValidationError):
        model_cls(**kwargs)
