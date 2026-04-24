"""Tests for the run_tlmtc library entrypoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import tlmtc.run as run_mod

DATA_PIPELINE_FLUENT: tuple[str, ...] = (
    "split_data",
    "get_multi_hot_vectors",
    "create_hf_dataset",
)

FINETUNE_PIPELINE_FLUENT: tuple[str, ...] = (
    "load_pretrained",
    "tune_hyperparameters",
    "fine_tune_pretrained",
    "tune_thresholds",
    "save_pretrained",
)


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """Create a minimal multi-label CSV for testing."""
    df = pd.DataFrame({"text": ["hello world", "foo bar"], "label_a": [1, 0], "label_b": [0, 1]})
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def raw_test_csv(tmp_path: Path) -> Path:
    """Create a minimal multi-label test CSV for testing."""
    df = pd.DataFrame({"text": ["test row 1", "test row 2"], "label_a": [0, 1], "label_b": [1, 0]})
    path = tmp_path / "raw_test_input.csv"
    df.to_csv(path, index=False)
    return path


def _chainable_mock(fluent_methods: tuple[str, ...]) -> MagicMock:
    """Create a chainable mock where fluent methods return self."""
    inst = MagicMock()
    for meth in fluent_methods:
        getattr(inst, meth).return_value = inst
    return inst


def test_run_tlmtc_returns_run_result_and_creates_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
) -> None:
    """Ensure run_tlmtc returns a RunResult and creates the expected run directories."""
    tokenized = object()

    dp_inst = _chainable_mock(DATA_PIPELINE_FLUENT)
    dp_inst.tokenize_data.return_value = SimpleNamespace(tokenized_dataset=tokenized)
    dp_cls = MagicMock(return_value=dp_inst)

    ft_inst = _chainable_mock(FINETUNE_PIPELINE_FLUENT)
    ft_cls = MagicMock(return_value=ft_inst)

    monkeypatch.setattr(run_mod, "DataPipeline", dp_cls)
    monkeypatch.setattr(run_mod, "FinetunePipeline", ft_cls)

    result = run_mod.run_tlmtc(raw_csv, work_dir=tmp_path, run_id="run_123")

    assert isinstance(result, run_mod.RunResult)
    assert result.paths.run_id == "run_123"

    # Observable side effects.
    assert result.paths.data_dir.exists()
    assert result.paths.logs_dir.exists()
    assert result.paths.model_dir.exists()

    # Minimal contract: tokenized dataset flows into finetuning + outputs are wired.
    _, dp_kwargs = dp_cls.call_args
    assert dp_kwargs["paths"] == result.paths

    _, ft_kwargs = ft_cls.call_args
    assert ft_kwargs["tokenized_dataset"] is tokenized
    assert ft_kwargs["paths"] == result.paths


@pytest.mark.parametrize("provide_raw_test", [False, True])
def test_run_tlmtc_resolves_raw_test_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
    raw_test_csv: Path,
    provide_raw_test: bool,
) -> None:
    """Ensure run_tlmtc resolves the raw test CSV path from argument or default."""
    tokenized = object()

    dp_inst = _chainable_mock(DATA_PIPELINE_FLUENT)
    dp_inst.tokenize_data.return_value = SimpleNamespace(tokenized_dataset=tokenized)
    monkeypatch.setattr(run_mod, "DataPipeline", MagicMock(return_value=dp_inst))

    ft_inst = _chainable_mock(FINETUNE_PIPELINE_FLUENT)
    monkeypatch.setattr(run_mod, "FinetunePipeline", MagicMock(return_value=ft_inst))

    result = run_mod.run_tlmtc(
        raw_csv,
        raw_test_csv=raw_test_csv if provide_raw_test else None,
        work_dir=tmp_path,
        run_id="run_abc",
    )

    if provide_raw_test:
        assert result.paths.raw_test_data_path == raw_test_csv.resolve()
    else:
        assert result.paths.raw_test_data_path.parent == raw_csv.resolve().parent
        assert result.paths.raw_test_data_path.name == "raw_test.csv"


def test_run_tlmtc_resolves_selected_settings_from_config_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
) -> None:
    """Ensure run_tlmtc loads YAML config values through RunSettings."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
run_id: config_run

model:
  target_name: Config Target
  sequence_length: 64

split:
  random_seed: 123

workflow:
  wrap_peft: false

hpo:
  optuna_space:
    batch_sizes: [4, 8]
""",
        encoding="utf-8",
    )

    tokenized = object()

    dp_inst = _chainable_mock(DATA_PIPELINE_FLUENT)
    dp_inst.tokenize_data.return_value = SimpleNamespace(tokenized_dataset=tokenized)
    dp_cls = MagicMock(return_value=dp_inst)

    ft_inst = _chainable_mock(FINETUNE_PIPELINE_FLUENT)
    ft_cls = MagicMock(return_value=ft_inst)

    monkeypatch.setattr(run_mod, "DataPipeline", dp_cls)
    monkeypatch.setattr(run_mod, "FinetunePipeline", ft_cls)

    result = run_mod.run_tlmtc(
        raw_csv,
        work_dir=tmp_path,
        config_path=config_path,
    )

    assert result.paths.run_id == "config_run"

    _, dp_kwargs = dp_cls.call_args
    assert dp_kwargs["model"].target_name == "Config Target"
    assert dp_kwargs["model"].sequence_length == 64
    assert dp_kwargs["split"].random_seed == 123

    _, ft_kwargs = ft_cls.call_args
    assert ft_kwargs["workflow"].wrap_peft is False
    assert ft_kwargs["hpo"].optuna_space.batch_sizes == [4, 8]


def test_run_tlmtc_propagates_data_pipeline_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
) -> None:
    """Ensure run_tlmtc propagates failures raised during data preparation."""
    dp_inst = _chainable_mock(DATA_PIPELINE_FLUENT)
    dp_inst.split_data.side_effect = ValueError("split failed")
    monkeypatch.setattr(run_mod, "DataPipeline", MagicMock(return_value=dp_inst))

    ft_cls = MagicMock()
    monkeypatch.setattr(run_mod, "FinetunePipeline", ft_cls)

    with pytest.raises(ValueError, match="split failed"):
        run_mod.run_tlmtc(raw_csv, work_dir=tmp_path, run_id="run_fail")

    ft_cls.assert_not_called()
