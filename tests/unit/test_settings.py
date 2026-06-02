"""Tests for layered settings resolution infrastructure."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from tlmtc.settings import (
    UNSET,
    HardwareSettings,
    HpoSettings,
    ModelSettings,
    OptunaSpaceSettings,
    PeftSettings,
    PredictionSettings,
    ResolvableSettings,
    RunSettings,
    RuntimeSettings,
    SplitSettings,
    ThresholdSettings,
    TrainingSettings,
    WorkflowSettings,
    deep_merge,
    load_config_file,
    prune_unset,
)

CUSTOM_OPTUNA_SPACE = {
    "lr_low": 1e-5,
    "lr_high": 3e-4,
    "batch_sizes": [8, 16, 32],
    "wd_low": 0.0,
    "wd_high": 0.3,
    "schedulers": ["linear", "cosine"],
    "epoch_low": 5,
    "epoch_high": 20,
    "lr_reference_batch_size": 32,
}

DEFAULT_OPTUNA_SPACE_BASE = {
    "lr_low": 1e-5,
    "lr_high": 8e-5,
    "batch_sizes": [8, 16, 32],
    "wd_low": 0.0,
    "wd_high": 0.1,
    "schedulers": ["linear", "cosine", "polynomial"],
    "epoch_low": 5,
    "epoch_high": 30,
    "lr_reference_batch_size": 32,
}

DEFAULT_OPTUNA_SPACE_PEFT = {
    "lr_low": 1e-5,
    "lr_high": 1e-4,
    "batch_sizes": [8, 16, 32],
    "wd_low": 0.0,
    "wd_high": 0.01,
    "schedulers": ["linear", "cosine"],
    "epoch_low": 5,
    "epoch_high": 20,
    "lr_reference_batch_size": 32,
}

MINIMAL_HPO = {"optuna_space": CUSTOM_OPTUNA_SPACE}


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


class TestSettingsInfrastructure:
    """Tests for shared layered-settings helpers and generic resolution behavior."""

    def test_unset_has_stable_repr(self) -> None:
        """UNSET should render predictably for debugging."""
        assert repr(UNSET) == "UNSET"

    def test_unset_has_no_truth_value(self) -> None:
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
    def test_deep_merge(
        self,
        base: dict[str, object],
        incoming: dict[str, object],
        expected: dict[str, object],
    ) -> None:
        """deep_merge should recursively merge higher-precedence settings."""
        merged = deep_merge(base, incoming)
        assert merged == expected

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
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
    def test_prune_unset(self, value: object, expected: object) -> None:
        """prune_unset should recursively remove UNSET values."""
        assert prune_unset(value) == expected

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (
                dedent(
                    """
                        foo: 1
                        bar: hello
                        nested:
                          alpha: 10
                          beta: 20
                        """
                ).strip(),
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
        self,
        tmp_path: Path,
        content: str,
        expected: dict[str, object],
    ) -> None:
        """load_config_file should return parsed YAML mapping data."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(content, encoding="utf-8")

        assert load_config_file(config_path) == expected

    def test_load_config_file_raises_for_missing_file(self, tmp_path: Path) -> None:
        """load_config_file should fail clearly for missing config files."""
        missing = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file does not exist"):
            load_config_file(missing)

    def test_load_config_file_raises_for_non_mapping_yaml_root(self, tmp_path: Path) -> None:
        """load_config_file should reject YAML files whose root is not a mapping."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- one\n- two\n", encoding="utf-8")

        with pytest.raises(TypeError, match="Config file root must be a mapping"):
            load_config_file(config_path)

    def test_resolve_uses_config_only(self) -> None:
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

    def test_resolve_applies_precedence_config_then_env_then_overrides(self) -> None:
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

    def test_resolve_prunes_unset_override_values(self) -> None:
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

    def test_resolve_rejects_extra_keys(self) -> None:
        """Resolve should respect extra='forbid' on the resolved model."""
        with pytest.raises(ValidationError):
            ExampleSettings.resolve(
                config={
                    "foo": 1,
                    "nested": {"alpha": 10},
                    "unknown": "boom",
                }
            )

    def test_resolve_requires_required_fields(self) -> None:
        """Resolve should fail validation when required fields are missing."""
        with pytest.raises(ValidationError):
            ExampleSettings.resolve(
                config={
                    "bar": "hello",
                    "nested": {"alpha": 10},
                }
            )

    def test_resolve_rejects_extra_keys_in_nested_models(self) -> None:
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


class TestBundleSettings:
    """Tests for bundle defaults, immutability, assignment validation, and field constraints."""

    def test_model_settings_defaults(self) -> None:
        """ModelSettings should expose the package defaults."""
        settings = ModelSettings()

        assert settings.target_name == "Target"
        assert settings.proxy_checkpoint == "microsoft/deberta-v3-small"
        assert settings.checkpoint == "microsoft/deberta-v3-base"
        assert settings.sequence_length == 128

    def test_model_settings_defaults_proxy_to_explicit_checkpoint(self) -> None:
        """ModelSettings should use an explicit checkpoint as proxy when proxy is omitted."""
        settings = ModelSettings(checkpoint="custom/target")

        assert settings.checkpoint == "custom/target"
        assert settings.proxy_checkpoint == "custom/target"

    def test_model_settings_preserves_explicit_proxy_checkpoint(self) -> None:
        """ModelSettings should preserve an explicit proxy checkpoint."""
        settings = ModelSettings(
            checkpoint="custom/target",
            proxy_checkpoint="custom/proxy",
        )

        assert settings.checkpoint == "custom/target"
        assert settings.proxy_checkpoint == "custom/proxy"

    def test_split_settings_defaults(self) -> None:
        """SplitSettings should expose the package defaults."""
        settings = SplitSettings()

        assert settings.validation_size == 0.15
        assert settings.test_size == 0.15
        assert settings.random_seed == 2469

    def test_workflow_settings_defaults(self) -> None:
        """WorkflowSettings should expose the package defaults."""
        settings = WorkflowSettings()

        assert settings.hyperparameter_tuning is True
        assert settings.threshold_optimization is True
        assert settings.transfer_learning is True
        assert settings.scale_learning_rate is False
        assert settings.wrap_peft is True

    def test_training_settings_defaults(self) -> None:
        """TrainingSettings should expose the package defaults."""
        settings = TrainingSettings()

        assert settings.batch_size == 16
        assert settings.train_epochs == 20
        assert settings.weight_decay == 0.01
        assert settings.learning_rate == 2e-5
        assert settings.lr_scheduler == "linear"
        assert settings.best_model_metric == "roc_auc_macro"
        assert settings.early_stopping_patience == 10

    def test_threshold_settings_defaults(self) -> None:
        """ThresholdSettings should expose the package defaults."""
        settings = ThresholdSettings()

        assert settings.threshold_type == "label"
        assert settings.best_threshold_metric == "f1_macro"

    def test_peft_settings_defaults(self) -> None:
        """PeftSettings should expose the package defaults."""
        settings = PeftSettings()

        assert settings.lora_r == 8
        assert settings.lora_alpha == 32
        assert settings.lora_dropout == 0.1
        assert settings.lora_bias == "none"

    def test_hardware_settings_defaults(self) -> None:
        """HardwareSettings should expose the package defaults."""
        settings = HardwareSettings()

        assert settings.use_cpu is False

    def test_runtime_settings_defaults(self) -> None:
        """RuntimeSettings should expose the package runtime-output defaults."""
        settings = RuntimeSettings()

        assert settings.verbosity == "progress"

    @pytest.mark.parametrize(
        "factory",
        [
            ModelSettings,
            SplitSettings,
            WorkflowSettings,
            TrainingSettings,
            ThresholdSettings,
            PeftSettings,
            HardwareSettings,
            RuntimeSettings,
        ],
    )
    def test_frozen_settings_reject_assignment(self, factory: type[BaseModel]) -> None:
        """Frozen settings bundles should reject mutation."""
        settings = factory()

        field_name = next(iter(type(settings).model_fields))
        current_value = getattr(settings, field_name)

        with pytest.raises(ValidationError):
            setattr(settings, field_name, current_value)

    @pytest.mark.parametrize(
        ("model_cls", "kwargs", "pattern"),
        [
            (ModelSettings, {"sequence_length": 0}, "sequence_length"),
            (SplitSettings, {"validation_size": 1.0}, "validation_size"),
            (SplitSettings, {"test_size": 0.0}, "test_size"),
            (TrainingSettings, {"learning_rate": 0.0}, "learning_rate"),
            (TrainingSettings, {"weight_decay": -0.1}, "weight_decay"),
            (PeftSettings, {"lora_dropout": 1.0}, "lora_dropout"),
            (PeftSettings, {"lora_r": 0}, "lora_r"),
            (RuntimeSettings, {"verbosity": "verbose"}, "verbosity"),
            (
                OptunaSpaceSettings,
                {
                    **CUSTOM_OPTUNA_SPACE,
                    "lr_reference_batch_size": 0,
                },
                "lr_reference_batch_size",
            ),
        ],
    )
    def test_settings_bundles_reject_invalid_values(
        self,
        model_cls: type[BaseModel],
        kwargs: dict[str, object],
        pattern: str,
    ) -> None:
        """Representative bundle validations should fail clearly for invalid values."""
        with pytest.raises(ValidationError, match=pattern):
            model_cls(**kwargs)

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (ModelSettings, {"unknown": 1}),
            (TrainingSettings, {"unknown": "boom"}),
            (PeftSettings, {"unknown": True}),
            (RuntimeSettings, {"unknown": "boom"}),
        ],
    )
    def test_settings_bundles_reject_extra_keys(
        self,
        model_cls: type[BaseModel],
        kwargs: dict[str, object],
    ) -> None:
        """Settings bundles should enforce extra='forbid'."""
        with pytest.raises(ValidationError):
            model_cls(**kwargs)


class TestRunSettings:
    """Tests for top-level run settings construction, nested defaults, and layered overrides."""

    def test_run_settings_minimal_construction_uses_nested_defaults(self) -> None:
        """RunSettings should construct from the minimal required inputs and apply nested defaults."""
        settings = RunSettings(raw_csv="train.csv")

        assert settings.raw_csv == Path("train.csv")
        assert settings.raw_test_csv is None
        assert settings.work_dir == Path.cwd()
        assert settings.run_id is None

        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.split, SplitSettings)
        assert isinstance(settings.workflow, WorkflowSettings)
        assert isinstance(settings.training, TrainingSettings)
        assert isinstance(settings.threshold, ThresholdSettings)
        assert isinstance(settings.hpo, HpoSettings)
        assert isinstance(settings.hpo.optuna_space, OptunaSpaceSettings)
        assert isinstance(settings.peft, PeftSettings)
        assert isinstance(settings.hardware, HardwareSettings)
        assert isinstance(settings.runtime, RuntimeSettings)

        assert settings.model.target_name == "Target"
        assert settings.split.validation_size == 0.15
        assert settings.workflow.wrap_peft is True
        assert settings.training.batch_size == 16
        assert settings.threshold.threshold_type == "label"
        assert settings.peft.lora_r == 8
        assert settings.hardware.use_cpu is False
        assert settings.runtime.verbosity == "progress"
        assert settings.hpo.tuning_trials == 10
        assert settings.hpo.optuna_space.model_dump(mode="python") == DEFAULT_OPTUNA_SPACE_PEFT

    def test_run_settings_defaults_work_dir_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """RunSettings should default work_dir to the current working directory."""
        monkeypatch.chdir(tmp_path)

        settings = RunSettings(raw_csv="train.csv")

        assert settings.work_dir == tmp_path

    def test_run_settings_accepts_explicit_work_dir_and_run_id(self, tmp_path: Path) -> None:
        """RunSettings should preserve explicit work_dir and run_id values."""
        settings = RunSettings(
            raw_csv="train.csv",
            work_dir=tmp_path / "workspace",
            run_id="manual-run",
        )

        assert settings.work_dir == tmp_path / "workspace"
        assert settings.run_id == "manual-run"

    def test_run_settings_resolve_applies_nested_overrides_and_preserves_defaults(self) -> None:
        """RunSettings.resolve should merge nested overrides while preserving untouched defaults."""
        settings = RunSettings.resolve(
            config={
                "raw_csv": "train.csv",
                "model": {
                    "checkpoint": "microsoft/deberta-v3-large",
                },
                "split": {
                    "validation_size": 0.2,
                },
                "hpo": {
                    "optuna_space": {
                        "batch_sizes": [16],
                        "epoch_high": 25,
                    }
                },
                "runtime": {
                    "verbosity": "quiet",
                },
            },
            overrides={
                "model": {
                    "sequence_length": 256,
                },
                "workflow": {
                    "wrap_peft": False,
                },
                "training": {
                    "batch_size": 32,
                },
                "runtime": {
                    "verbosity": "progress",
                },
            },
        )

        assert settings.raw_csv == Path("train.csv")
        assert settings.model.target_name == "Target"
        assert settings.model.checkpoint == "microsoft/deberta-v3-large"
        assert settings.model.proxy_checkpoint == "microsoft/deberta-v3-large"
        assert settings.model.sequence_length == 256
        assert settings.split.validation_size == 0.2
        assert settings.split.test_size == 0.15
        assert settings.workflow.wrap_peft is False
        assert settings.workflow.transfer_learning is True
        assert settings.training.batch_size == 32
        assert settings.training.learning_rate == 2e-5
        assert settings.hpo.optuna_space.model_dump(mode="python") == {
            **DEFAULT_OPTUNA_SPACE_BASE,
            "batch_sizes": [16],
            "epoch_high": 25,
        }
        assert settings.runtime.verbosity == "progress"

    def test_run_settings_resolve_prunes_unset_in_nested_overrides(self) -> None:
        """RunSettings.resolve should ignore nested override values explicitly marked as UNSET."""
        settings = RunSettings.resolve(
            config={
                "raw_csv": "train.csv",
                "model": {
                    "target_name": "My task",
                    "sequence_length": 256,
                },
                "training": {
                    "batch_size": 32,
                },
                "hpo": MINIMAL_HPO,
                "runtime": {
                    "verbosity": "quiet",
                },
            },
            overrides={
                "model": {
                    "target_name": UNSET,
                    "sequence_length": UNSET,
                },
                "training": {
                    "batch_size": UNSET,
                    "learning_rate": 1e-4,
                },
                "runtime": {
                    "verbosity": UNSET,
                },
            },
        )

        assert settings.model.target_name == "My task"
        assert settings.model.sequence_length == 256
        assert settings.training.batch_size == 32
        assert settings.training.learning_rate == 1e-4
        assert settings.runtime.verbosity == "quiet"

    def test_run_settings_requires_raw_csv(self) -> None:
        """RunSettings should require raw_csv."""
        with pytest.raises(ValidationError):
            RunSettings(
                hpo=MINIMAL_HPO,
            )

    def test_run_settings_synthesizes_hpo_defaults_when_omitted(self) -> None:
        """RunSettings should synthesize HPO defaults when hpo is omitted."""
        settings = RunSettings(raw_csv="train.csv")

        assert settings.hpo.tuning_trials == 10
        assert isinstance(settings.hpo.optuna_space, OptunaSpaceSettings)
        assert settings.hpo.optuna_space.model_dump(mode="python") == DEFAULT_OPTUNA_SPACE_PEFT

    @pytest.mark.parametrize(
        ("payload", "pattern"),
        [
            (
                {
                    "raw_csv": "train.csv",
                    "hpo": MINIMAL_HPO,
                    "unknown": "boom",
                },
                "unknown",
            ),
            (
                {
                    "raw_csv": "train.csv",
                    "model": {"unknown": "boom"},
                    "hpo": MINIMAL_HPO,
                },
                "unknown",
            ),
        ],
    )
    def test_run_settings_rejects_extra_keys(self, payload: dict[str, object], pattern: str) -> None:
        """RunSettings should enforce extra='forbid' at both root and nested levels."""
        with pytest.raises(ValidationError, match=pattern):
            RunSettings(**payload)

    def test_run_settings_preserves_full_explicit_optuna_space(self) -> None:
        """RunSettings should preserve a fully specified explicit Optuna space."""
        settings = RunSettings(
            raw_csv="train.csv",
            hpo=MINIMAL_HPO,
        )

        assert isinstance(settings.hpo.optuna_space, OptunaSpaceSettings)
        assert settings.hpo.optuna_space.model_dump(mode="python") == CUSTOM_OPTUNA_SPACE

    def test_run_settings_rejects_non_mapping_optuna_space_override(self) -> None:
        """RunSettings should reject non-mapping Optuna-space overrides before merge."""
        with pytest.raises(TypeError, match="hpo.optuna_space must be a mapping"):
            RunSettings(
                raw_csv="train.csv",
                hpo={"optuna_space": "boom"},
            )


class TestPredictionSettings:
    """Tests for top-level prediction settings construction and layered overrides."""

    def test_prediction_settings_minimal_construction_uses_defaults(self) -> None:
        """PredictionSettings should construct from the required prediction CSV and apply defaults."""
        settings = PredictionSettings(prediction_csv="predict.csv")

        assert settings.prediction_csv == Path("predict.csv")
        assert settings.work_dir == Path.cwd()
        assert settings.run_id is None
        assert settings.batch_size == 32
        assert isinstance(settings.hardware, HardwareSettings)
        assert settings.hardware.use_cpu is False
        assert isinstance(settings.runtime, RuntimeSettings)
        assert settings.runtime.verbosity == "progress"

    def test_prediction_settings_accepts_explicit_values(self, tmp_path: Path) -> None:
        """PredictionSettings should preserve explicit prediction runtime values."""
        settings = PredictionSettings(
            prediction_csv=tmp_path / "predict.csv",
            work_dir=tmp_path / "workspace",
            run_id="manual-run",
            batch_size=64,
            hardware={"use_cpu": True},
            runtime={"verbosity": "quiet"},
        )

        assert settings.prediction_csv == tmp_path / "predict.csv"
        assert settings.work_dir == tmp_path / "workspace"
        assert settings.run_id == "manual-run"
        assert settings.batch_size == 64
        assert settings.hardware.use_cpu is True
        assert settings.runtime.verbosity == "quiet"

    def test_prediction_settings_resolve_applies_overrides_and_preserves_defaults(self) -> None:
        """PredictionSettings.resolve should merge prediction overrides while preserving defaults."""
        settings = PredictionSettings.resolve(
            config={
                "prediction_csv": "from-config.csv",
                "work_dir": "from-config-workdir",
                "run_id": "from-config-run",
                "batch_size": 16,
                "runtime": {
                    "verbosity": "quiet",
                },
            },
            overrides={
                "prediction_csv": "from-override.csv",
                "run_id": None,
                "hardware": {
                    "use_cpu": True,
                },
            },
        )

        assert settings.prediction_csv == Path("from-override.csv")
        assert settings.work_dir == Path("from-config-workdir")
        assert settings.run_id is None
        assert settings.batch_size == 16
        assert settings.hardware.use_cpu is True
        assert settings.runtime.verbosity == "quiet"

    def test_prediction_settings_resolve_prunes_unset_values(self) -> None:
        """PredictionSettings.resolve should ignore override values explicitly marked as UNSET."""
        settings = PredictionSettings.resolve(
            config={
                "prediction_csv": "predict.csv",
                "work_dir": "configured-workdir",
                "run_id": "configured-run",
                "batch_size": 64,
                "hardware": {
                    "use_cpu": True,
                },
                "runtime": {
                    "verbosity": "quiet",
                },
            },
            overrides={
                "work_dir": UNSET,
                "run_id": UNSET,
                "batch_size": UNSET,
                "hardware": {
                    "use_cpu": UNSET,
                },
                "runtime": {
                    "verbosity": UNSET,
                },
            },
        )

        assert settings.prediction_csv == Path("predict.csv")
        assert settings.work_dir == Path("configured-workdir")
        assert settings.run_id == "configured-run"
        assert settings.batch_size == 64
        assert settings.hardware.use_cpu is True
        assert settings.runtime.verbosity == "quiet"

    def test_prediction_settings_requires_prediction_csv(self) -> None:
        """PredictionSettings should require prediction_csv."""
        with pytest.raises(ValidationError):
            PredictionSettings()

    def test_prediction_settings_rejects_invalid_batch_size(self) -> None:
        """PredictionSettings should reject non-positive prediction batch sizes."""
        with pytest.raises(ValidationError, match="batch_size"):
            PredictionSettings(
                prediction_csv="predict.csv",
                batch_size=0,
            )

    @pytest.mark.parametrize(
        ("payload", "pattern"),
        [
            (
                {
                    "prediction_csv": "predict.csv",
                    "unknown": "boom",
                },
                "unknown",
            ),
            (
                {
                    "prediction_csv": "predict.csv",
                    "hardware": {"unknown": "boom"},
                },
                "unknown",
            ),
            (
                {
                    "prediction_csv": "predict.csv",
                    "runtime": {"unknown": "boom"},
                },
                "unknown",
            ),
        ],
    )
    def test_prediction_settings_rejects_extra_keys(
        self,
        payload: dict[str, object],
        pattern: str,
    ) -> None:
        """PredictionSettings should enforce extra='forbid' at both root and nested levels."""
        with pytest.raises(ValidationError, match=pattern):
            PredictionSettings(**payload)
