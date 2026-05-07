"""Tests for the train_tlmtc library entrypoint."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import tlmtc.api as train_mod
from tlmtc.data_contracts import InputMode
from tlmtc.meta import read_run_meta

DATA_PIPELINE_METHODS: tuple[str, ...] = (
    "split_data",
    "get_multi_hot_vectors",
    "create_hf_dataset",
    "tokenize_data",
)

FINETUNE_PIPELINE_METHODS: tuple[str, ...] = (
    "load_pretrained",
    "tune_hyperparameters",
    "fine_tune_pretrained",
    "tune_thresholds",
    "save_pretrained",
)

EVALUATION_PIPELINE_METHODS: tuple[str, ...] = (
    "run_evaluation",
    "save_metrics",
    "render_tables",
    "render_figures",
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


def _chainable_mock(methods: tuple[str, ...]) -> MagicMock:
    """Create a mock whose pipeline methods return self."""
    inst = MagicMock()
    for method in methods:
        getattr(inst, method).return_value = inst
    return inst


def _mock_successful_pipelines(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock, MagicMock, object, object, object]:
    """Mock successful data, fine-tuning, and evaluation pipelines."""
    tokenized_dataset = object()
    updated_trainer = object()
    tuned_threshold = np.array([0.42, 0.61], dtype=float)

    data_pipeline = _chainable_mock(DATA_PIPELINE_METHODS)
    data_pipeline.tokenized_dataset = tokenized_dataset
    data_pipeline.input_mode = InputMode.SINGLE_TEXT
    data_pipeline_cls = MagicMock(return_value=data_pipeline)

    finetune_pipeline = _chainable_mock(FINETUNE_PIPELINE_METHODS)
    finetune_pipeline.updated_trainer = updated_trainer
    finetune_pipeline.tuned_threshold = tuned_threshold
    finetune_pipeline_cls = MagicMock(return_value=finetune_pipeline)

    evaluation_pipeline = _chainable_mock(EVALUATION_PIPELINE_METHODS)
    evaluation_pipeline.label_names = ["a", "b"]
    evaluation_pipeline_cls = MagicMock(return_value=evaluation_pipeline)

    monkeypatch.setattr(train_mod, "DataPipeline", data_pipeline_cls)
    monkeypatch.setattr(train_mod, "FinetunePipeline", finetune_pipeline_cls)
    monkeypatch.setattr(train_mod, "EvaluationPipeline", evaluation_pipeline_cls)

    return (
        data_pipeline_cls,
        finetune_pipeline_cls,
        evaluation_pipeline_cls,
        tokenized_dataset,
        updated_trainer,
        tuned_threshold,
    )


def test_train_tlmtc_returns_train_result_and_creates_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
) -> None:
    (
        data_pipeline_cls,
        finetune_pipeline_cls,
        evaluation_pipeline_cls,
        tokenized_dataset,
        updated_trainer,
        tuned_threshold,
    ) = _mock_successful_pipelines(monkeypatch)

    result = train_mod.train_tlmtc(raw_csv, work_dir=tmp_path, run_id="run_123")

    assert isinstance(result, train_mod.TrainResult)
    assert result.paths.run_id == "run_123"

    assert result.paths.data_dir.exists()
    assert result.paths.logs_dir.exists()
    assert result.paths.model_dir.exists()
    assert result.paths.eval_dir.exists()
    assert result.paths.train_run_meta_path.exists()

    _, data_kwargs = data_pipeline_cls.call_args
    assert data_kwargs["paths"] == result.paths

    _, finetune_kwargs = finetune_pipeline_cls.call_args
    assert finetune_kwargs["tokenized_dataset"] is tokenized_dataset
    assert finetune_kwargs["paths"] == result.paths

    _, evaluation_kwargs = evaluation_pipeline_cls.call_args
    assert evaluation_kwargs["tokenized_dataset"] is tokenized_dataset
    assert evaluation_kwargs["updated_trainer"] is updated_trainer
    assert evaluation_kwargs["paths"] == result.paths
    assert evaluation_kwargs["tuned_threshold"] is tuned_threshold

    run_meta = read_run_meta(result.paths.train_run_meta_path)
    assert run_meta.run_id == "run_123"
    assert run_meta.target_name == "Target"
    assert run_meta.input_mode is InputMode.SINGLE_TEXT
    assert run_meta.label_names == ["a", "b"]
    assert run_meta.threshold_type == "label"
    assert run_meta.thresholds == [0.42, 0.61]
    assert run_meta.transfer_learning is True
    assert run_meta.hyperparameter_tuning is True
    assert run_meta.threshold_optimization is True
    assert run_meta.scale_learning_rate is False
    assert run_meta.wrap_peft is True


def test_train_tlmtc_preserves_explicit_raw_test_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
    raw_test_csv: Path,
) -> None:
    _mock_successful_pipelines(monkeypatch)

    result = train_mod.train_tlmtc(
        raw_csv,
        raw_test_csv=raw_test_csv,
        work_dir=tmp_path,
        run_id="run_abc",
    )

    assert result.paths.raw_test_data_path == raw_test_csv.resolve()


def test_train_tlmtc_leaves_raw_test_path_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
) -> None:
    _mock_successful_pipelines(monkeypatch)

    result = train_mod.train_tlmtc(
        raw_csv,
        work_dir=tmp_path,
        run_id="run_abc",
    )

    assert result.paths.raw_test_data_path is None


def test_train_tlmtc_resolves_selected_settings_from_config_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
) -> None:
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

    data_pipeline_cls, finetune_pipeline_cls, _, _, _, _ = _mock_successful_pipelines(monkeypatch)

    result = train_mod.train_tlmtc(
        raw_csv,
        work_dir=tmp_path,
        config_path=config_path,
    )

    assert result.paths.run_id == "config_run"

    _, data_kwargs = data_pipeline_cls.call_args
    assert data_kwargs["model"].target_name == "Config Target"
    assert data_kwargs["model"].sequence_length == 64
    assert data_kwargs["split"].random_seed == 123

    _, finetune_kwargs = finetune_pipeline_cls.call_args
    assert finetune_kwargs["workflow"].wrap_peft is False
    assert finetune_kwargs["hpo"].optuna_space.batch_sizes == [4, 8]


def test_train_tlmtc_propagates_data_pipeline_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_csv: Path,
) -> None:
    """Ensure train_tlmtc propagates failures raised during data preparation."""
    data_pipeline = _chainable_mock(DATA_PIPELINE_METHODS)
    data_pipeline.split_data.side_effect = ValueError("split failed")
    monkeypatch.setattr(train_mod, "DataPipeline", MagicMock(return_value=data_pipeline))

    finetune_pipeline_cls = MagicMock()
    evaluation_pipeline_cls = MagicMock()
    monkeypatch.setattr(train_mod, "FinetunePipeline", finetune_pipeline_cls)
    monkeypatch.setattr(train_mod, "EvaluationPipeline", evaluation_pipeline_cls)

    with pytest.raises(ValueError, match="split failed"):
        train_mod.train_tlmtc(raw_csv, work_dir=tmp_path, run_id="run_fail")

    finetune_pipeline_cls.assert_not_called()
    evaluation_pipeline_cls.assert_not_called()
