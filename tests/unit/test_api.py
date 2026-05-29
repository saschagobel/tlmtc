"""Tests for the tlmtc library entrypoints."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Literal
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest

import tlmtc.api as api_mod
from tlmtc.data_contracts import InputMode
from tlmtc.meta import TrainRunMeta, read_run_meta, write_run_meta

LABEL_NAMES = ["a", "b"]
TRAIN_THRESHOLDS = np.array([0.42, 0.61], dtype=float)
PERSISTED_THRESHOLDS = [0.5, 0.6]
PROBABILITIES = np.array(
    [
        [0.2, 0.7],
        [0.9, 0.1],
    ],
    dtype=float,
)
BINARY_PREDICTIONS = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=int,
)

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


@dataclass(frozen=True, slots=True)
class MockedPipelines:
    """Mocked training pipeline objects and sentinel state."""

    data_pipeline_cls: MagicMock
    finetune_pipeline_cls: MagicMock
    evaluation_pipeline_cls: MagicMock

    data_pipeline: MagicMock
    finetune_pipeline: MagicMock
    evaluation_pipeline: MagicMock

    tokenized_dataset: object
    updated_trainer: object
    tuned_threshold: np.ndarray


@dataclass(frozen=True, slots=True)
class MockedPredictionOps:
    """Mocked prediction operations and sentinel state."""

    read_prediction_csv: MagicMock
    create_prediction_dataset: MagicMock
    tokenize_prediction_dataset: MagicMock
    load_prediction_model: MagicMock
    predict_probabilities: MagicMock

    input_df: pd.DataFrame
    prediction_dataset: object
    tokenized_dataset: object
    model: object
    probabilities: np.ndarray


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """Create a minimal multi-label CSV for testing."""
    df = pd.DataFrame(
        {
            "text": ["hello world", "foo bar"],
            "label_a": [1, 0],
            "label_b": [0, 1],
        }
    )
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def raw_test_csv(tmp_path: Path) -> Path:
    """Create a minimal multi-label test CSV for testing."""
    df = pd.DataFrame(
        {
            "text": ["test row 1", "test row 2"],
            "label_a": [0, 1],
            "label_b": [1, 0],
        }
    )
    path = tmp_path / "raw_test_input.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def prediction_csv(tmp_path: Path) -> Path:
    """Create a minimal unlabeled prediction CSV for testing."""
    df = pd.DataFrame(
        {
            "text": ["prediction row 1", "prediction row 2"],
            "source_id": ["one", "two"],
        }
    )
    path = tmp_path / "prediction.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def paired_prediction_csv(tmp_path: Path) -> Path:
    """Create a minimal paired-text prediction CSV for testing."""
    df = pd.DataFrame(
        {
            "text": ["query one", "query two"],
            "text_pair": ["candidate one", "candidate two"],
            "source_id": ["one", "two"],
        }
    )
    path = tmp_path / "paired_prediction.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def configure_runtime_output_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock runtime-output configuration for API orchestration tests."""
    mock = MagicMock()
    monkeypatch.setattr(api_mod, "configure_runtime_output", mock)
    return mock

@pytest.fixture(autouse=True)
def distributed_context_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock distributed runtime context for API orchestration tests."""
    context = MagicMock()
    context.is_main_process = True

    distributed_context_cls = MagicMock()
    distributed_context_cls.create.return_value = context

    monkeypatch.setattr(api_mod, "DistributedContext", distributed_context_cls)
    return context


def _write_yaml(path: Path, content: str) -> None:
    """Write dedented YAML test config."""
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _chainable_mock(methods: tuple[str, ...]) -> MagicMock:
    """Create a mock whose pipeline methods return self."""
    instance = MagicMock()

    for method in methods:
        getattr(instance, method).return_value = instance

    return instance


def _mock_successful_pipelines(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_mode: InputMode = InputMode.SINGLE_TEXT,
    label_names: list[str] | None = None,
    tuned_threshold: np.ndarray | None = None,
) -> MockedPipelines:
    """Mock successful data, fine-tuning, and evaluation pipelines."""
    tokenized_dataset = object()
    updated_trainer = object()
    threshold = TRAIN_THRESHOLDS if tuned_threshold is None else tuned_threshold

    data_pipeline = _chainable_mock(DATA_PIPELINE_METHODS)
    data_pipeline.tokenized_dataset = tokenized_dataset
    data_pipeline.input_mode = input_mode
    data_pipeline_cls = MagicMock(return_value=data_pipeline)

    finetune_pipeline = _chainable_mock(FINETUNE_PIPELINE_METHODS)
    finetune_pipeline.updated_trainer = updated_trainer
    finetune_pipeline.tuned_threshold = threshold
    finetune_pipeline_cls = MagicMock(return_value=finetune_pipeline)

    evaluation_pipeline = _chainable_mock(EVALUATION_PIPELINE_METHODS)
    evaluation_pipeline.label_names = LABEL_NAMES if label_names is None else label_names
    evaluation_pipeline_cls = MagicMock(return_value=evaluation_pipeline)

    monkeypatch.setattr(api_mod, "DataPipeline", data_pipeline_cls)
    monkeypatch.setattr(api_mod, "FinetunePipeline", finetune_pipeline_cls)
    monkeypatch.setattr(api_mod, "EvaluationPipeline", evaluation_pipeline_cls)

    return MockedPipelines(
        data_pipeline_cls=data_pipeline_cls,
        finetune_pipeline_cls=finetune_pipeline_cls,
        evaluation_pipeline_cls=evaluation_pipeline_cls,
        data_pipeline=data_pipeline,
        finetune_pipeline=finetune_pipeline,
        evaluation_pipeline=evaluation_pipeline,
        tokenized_dataset=tokenized_dataset,
        updated_trainer=updated_trainer,
        tuned_threshold=threshold,
    )


def _write_prediction_ready_train_run(
    work_dir: Path,
    *,
    run_id: str,
    created_at: datetime | None = None,
    input_mode: InputMode = InputMode.SINGLE_TEXT,
    label_names: list[str] | None = None,
    threshold_type: Literal["global", "label"] = "label",
    thresholds: list[float] | None = None,
    transfer_learning: bool = True,
    wrap_peft: bool = False,
) -> None:
    """Write minimal training artifacts required by predict_tlmtc."""
    train_run_dir = work_dir / "train_outputs" / run_id
    model_dir = train_run_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("placeholder", encoding="utf-8")

    meta_kwargs: dict[str, object] = {}
    if created_at is not None:
        meta_kwargs["created_at"] = created_at

    write_run_meta(
        meta=TrainRunMeta(
            run_id=run_id,
            target_name="Target",
            checkpoint="test-checkpoint",
            proxy_checkpoint="test-proxy-checkpoint",
            sequence_length=16,
            input_mode=input_mode,
            label_names=LABEL_NAMES if label_names is None else label_names,
            threshold_type=threshold_type,
            thresholds=PERSISTED_THRESHOLDS if thresholds is None else thresholds,
            transfer_learning=transfer_learning,
            hyperparameter_tuning=False,
            threshold_optimization=True,
            scale_learning_rate=False,
            wrap_peft=wrap_peft,
            **meta_kwargs,
        ),
        path=train_run_dir / "train_run_meta.json",
    )


def _prediction_input_frame(
    input_mode: InputMode,
) -> pd.DataFrame:
    """Create a prediction input frame matching the selected input mode."""
    data: dict[str, list[str]] = {
        "text": ["prediction row 1", "prediction row 2"],
    }

    if input_mode is InputMode.PAIRED_TEXT:
        data["text_pair"] = ["paired row 1", "paired row 2"]

    data["source_id"] = ["one", "two"]
    return pd.DataFrame(data)


def _mock_prediction_operations(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_mode: InputMode = InputMode.SINGLE_TEXT,
    probabilities: np.ndarray | None = None,
) -> MockedPredictionOps:
    """Mock prediction operations called by predict_tlmtc."""
    input_df = _prediction_input_frame(input_mode)
    prediction_dataset = object()
    tokenized_dataset = object()
    model = object()
    probability_values = PROBABILITIES if probabilities is None else probabilities

    read_prediction_csv_mock = MagicMock(return_value=input_df)
    create_prediction_dataset_mock = MagicMock(return_value=prediction_dataset)
    tokenize_prediction_dataset_mock = MagicMock(return_value=tokenized_dataset)
    load_prediction_model_mock = MagicMock(return_value=model)
    predict_probabilities_mock = MagicMock(return_value=probability_values)

    monkeypatch.setattr(api_mod, "read_prediction_csv", read_prediction_csv_mock)
    monkeypatch.setattr(api_mod, "create_prediction_dataset", create_prediction_dataset_mock)
    monkeypatch.setattr(api_mod, "tokenize_prediction_dataset", tokenize_prediction_dataset_mock)
    monkeypatch.setattr(api_mod, "load_prediction_model", load_prediction_model_mock)
    monkeypatch.setattr(api_mod, "predict_probabilities", predict_probabilities_mock)

    return MockedPredictionOps(
        read_prediction_csv=read_prediction_csv_mock,
        create_prediction_dataset=create_prediction_dataset_mock,
        tokenize_prediction_dataset=tokenize_prediction_dataset_mock,
        load_prediction_model=load_prediction_model_mock,
        predict_probabilities=predict_probabilities_mock,
        input_df=input_df,
        prediction_dataset=prediction_dataset,
        tokenized_dataset=tokenized_dataset,
        model=model,
        probabilities=probability_values,
    )


def _assert_training_pipeline_call_order(pipelines: MockedPipelines) -> None:
    """Assert the public API runs training pipeline stages in order."""
    assert pipelines.data_pipeline.method_calls == [
        call.split_data(),
        call.get_multi_hot_vectors(),
        call.create_hf_dataset(),
        call.tokenize_data(),
    ]
    assert pipelines.finetune_pipeline.method_calls == [
        call.tune_hyperparameters(),
        call.load_pretrained(),
        call.fine_tune_pretrained(),
        call.tune_thresholds(),
        call.save_pretrained(),
    ]
    assert pipelines.evaluation_pipeline.method_calls == [
        call.run_evaluation(),
        call.save_metrics(),
        call.render_tables(),
        call.render_figures(),
    ]


def _assert_default_train_meta(path: Path) -> None:
    """Assert persisted training metadata for the default mocked run."""
    run_meta = read_run_meta(path)

    assert run_meta.run_id == "run_123"
    assert run_meta.target_name == "Target"
    assert run_meta.input_mode is InputMode.SINGLE_TEXT
    assert run_meta.label_names == LABEL_NAMES
    assert run_meta.threshold_type == "label"
    assert run_meta.thresholds == TRAIN_THRESHOLDS.tolist()
    assert run_meta.transfer_learning is True
    assert run_meta.hyperparameter_tuning is True
    assert run_meta.threshold_optimization is True
    assert run_meta.scale_learning_rate is False
    assert run_meta.wrap_peft is True


def _assert_prediction_operations_called(
    result: api_mod.PredictResult,
    ops: MockedPredictionOps,
    *,
    input_mode: InputMode,
    batch_size: int,
    use_cpu: bool,
) -> None:
    """Assert predict_tlmtc wires prediction operations with resolved metadata and settings."""
    ops.read_prediction_csv.assert_called_once_with(
        df_path=result.paths.input_data_path,
        expected_input_mode=input_mode,
    )

    _, create_dataset_kwargs = ops.create_prediction_dataset.call_args
    assert create_dataset_kwargs["df"] is ops.input_df
    assert create_dataset_kwargs["input_mode"] is input_mode

    ops.tokenize_prediction_dataset.assert_called_once_with(
        dataset=ops.prediction_dataset,
        checkpoint="test-checkpoint",
        input_mode=input_mode,
        sequence_length=16,
    )
    ops.load_prediction_model.assert_called_once_with(
        model_dir=result.paths.train_run_model_dir,
        checkpoint="test-checkpoint",
        num_labels=len(LABEL_NAMES),
        wrap_peft=False,
    )
    ops.predict_probabilities.assert_called_once_with(
        model=ops.model,
        dataset=ops.tokenized_dataset,
        batch_size=batch_size,
        use_cpu=use_cpu,
    )


def _assert_prediction_outputs(
    result: api_mod.PredictResult,
    ops: MockedPredictionOps,
    *,
    expected_binary_predictions: np.ndarray = BINARY_PREDICTIONS,
) -> None:
    """Assert probability and binary prediction CSV artifacts."""
    probability_df = pd.read_csv(result.paths.probabilities_path)
    prediction_df = pd.read_csv(result.paths.predictions_path)

    pd.testing.assert_frame_equal(
        probability_df[ops.input_df.columns],
        ops.input_df,
    )
    pd.testing.assert_frame_equal(
        prediction_df[ops.input_df.columns],
        ops.input_df,
    )

    np.testing.assert_allclose(
        probability_df[LABEL_NAMES].to_numpy(),
        ops.probabilities,
    )
    np.testing.assert_array_equal(
        prediction_df[LABEL_NAMES].to_numpy(),
        expected_binary_predictions,
    )


class TestTrainTlmtc:
    """Tests for the public training entrypoint."""

    def test_returns_train_result_creates_dirs_and_wires_pipelines(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_csv: Path,
    ) -> None:
        pipelines = _mock_successful_pipelines(monkeypatch)

        result = api_mod.train_tlmtc(raw_csv, work_dir=tmp_path, run_id="run_123")

        assert isinstance(result, api_mod.TrainResult)
        assert result.paths.run_id == "run_123"

        assert result.paths.data_dir.exists()
        assert result.paths.logs_dir.exists()
        assert result.paths.model_dir.exists()
        assert result.paths.eval_dir.exists()
        assert result.paths.train_run_meta_path.exists()

        _, data_kwargs = pipelines.data_pipeline_cls.call_args
        assert data_kwargs["paths"] == result.paths

        _, finetune_kwargs = pipelines.finetune_pipeline_cls.call_args
        assert finetune_kwargs["tokenized_dataset"] is pipelines.tokenized_dataset
        assert finetune_kwargs["paths"] == result.paths

        _, evaluation_kwargs = pipelines.evaluation_pipeline_cls.call_args
        assert evaluation_kwargs["tokenized_dataset"] is pipelines.tokenized_dataset
        assert evaluation_kwargs["updated_trainer"] is pipelines.updated_trainer
        assert evaluation_kwargs["paths"] == result.paths
        assert evaluation_kwargs["tuned_threshold"] is pipelines.tuned_threshold

        _assert_training_pipeline_call_order(pipelines)
        _assert_default_train_meta(result.paths.train_run_meta_path)

    def test_preserves_explicit_raw_test_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_csv: Path,
        raw_test_csv: Path,
    ) -> None:
        _mock_successful_pipelines(monkeypatch)

        result = api_mod.train_tlmtc(
            raw_csv,
            raw_test_csv=raw_test_csv,
            work_dir=tmp_path,
            run_id="run_abc",
        )

        assert result.paths.raw_test_data_path == raw_test_csv.resolve()

    def test_leaves_raw_test_path_absent_when_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_csv: Path,
    ) -> None:
        _mock_successful_pipelines(monkeypatch)

        result = api_mod.train_tlmtc(
            raw_csv,
            work_dir=tmp_path,
            run_id="run_abc",
        )

        assert result.paths.raw_test_data_path is None

    def test_resolves_selected_settings_from_config_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_csv: Path,
    ) -> None:
        config_path = tmp_path / "config.yml"
        _write_yaml(
            config_path,
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
        )

        pipelines = _mock_successful_pipelines(monkeypatch)

        result = api_mod.train_tlmtc(
            raw_csv,
            work_dir=tmp_path,
            config_path=config_path,
        )

        assert result.paths.run_id == "config_run"

        _, data_kwargs = pipelines.data_pipeline_cls.call_args
        assert data_kwargs["model"].target_name == "Config Target"
        assert data_kwargs["model"].sequence_length == 64
        assert data_kwargs["split"].random_seed == 123

        _, finetune_kwargs = pipelines.finetune_pipeline_cls.call_args
        assert finetune_kwargs["workflow"].wrap_peft is False
        assert finetune_kwargs["hpo"].optuna_space.batch_sizes == [4, 8]

    def test_explicit_arguments_override_config_path_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_csv: Path,
    ) -> None:
        config_path = tmp_path / "config.yml"
        _write_yaml(
            config_path,
            """
            run_id: config_run

            model:
              target_name: Config Target
              sequence_length: 64

            training:
              batch_size: 4

            workflow:
              wrap_peft: false
            """,
        )

        pipelines = _mock_successful_pipelines(monkeypatch)

        result = api_mod.train_tlmtc(
            raw_csv,
            work_dir=tmp_path,
            config_path=config_path,
            run_id="explicit_run",
            target_name="Explicit Target",
            sequence_length=32,
            batch_size=8,
            wrap_peft=True,
        )

        assert result.paths.run_id == "explicit_run"

        _, data_kwargs = pipelines.data_pipeline_cls.call_args
        assert data_kwargs["model"].target_name == "Explicit Target"
        assert data_kwargs["model"].sequence_length == 32

        _, finetune_kwargs = pipelines.finetune_pipeline_cls.call_args
        assert finetune_kwargs["training"].batch_size == 8
        assert finetune_kwargs["workflow"].wrap_peft is True

    def test_persists_paired_text_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_csv: Path,
    ) -> None:
        _mock_successful_pipelines(monkeypatch, input_mode=InputMode.PAIRED_TEXT)

        result = api_mod.train_tlmtc(raw_csv, work_dir=tmp_path, run_id="paired_run")

        run_meta = read_run_meta(result.paths.train_run_meta_path)
        assert run_meta.input_mode is InputMode.PAIRED_TEXT

    def test_propagates_data_pipeline_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_csv: Path,
    ) -> None:
        data_pipeline = _chainable_mock(DATA_PIPELINE_METHODS)
        data_pipeline.split_data.side_effect = ValueError("split failed")
        monkeypatch.setattr(api_mod, "DataPipeline", MagicMock(return_value=data_pipeline))

        finetune_pipeline_cls = MagicMock()
        evaluation_pipeline_cls = MagicMock()
        monkeypatch.setattr(api_mod, "FinetunePipeline", finetune_pipeline_cls)
        monkeypatch.setattr(api_mod, "EvaluationPipeline", evaluation_pipeline_cls)

        with pytest.raises(ValueError, match="split failed"):
            api_mod.train_tlmtc(raw_csv, work_dir=tmp_path, run_id="run_fail")

        finetune_pipeline_cls.assert_not_called()
        evaluation_pipeline_cls.assert_not_called()

    def test_configures_runtime_output_from_explicit_argument(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_csv: Path,
        configure_runtime_output_mock: MagicMock,
        distributed_context_mock: MagicMock,
    ) -> None:
        _mock_successful_pipelines(monkeypatch)

        api_mod.train_tlmtc(
            raw_csv,
            work_dir=tmp_path,
            run_id="quiet_run",
            verbosity="quiet",
        )

        configure_runtime_output_mock.assert_called_once_with("quiet", is_main_process=True)
        distributed_context_mock.warn_if_multi_gpu_without_launcher.assert_called_once_with(use_cpu=False)

class TestPredictTlmtc:
    """Tests for the public prediction entrypoint."""

    def test_returns_predict_result_and_writes_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        prediction_csv: Path,
    ) -> None:
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="run_123",
        )
        ops = _mock_prediction_operations(monkeypatch)

        result = api_mod.predict_tlmtc(
            prediction_csv,
            work_dir=tmp_path,
            run_id="run_123",
            batch_size=8,
            use_cpu=True,
        )

        assert isinstance(result, api_mod.PredictResult)
        assert result.paths.run_id == "run_123"
        assert result.paths.probabilities_path.exists()
        assert result.paths.predictions_path.exists()

        _assert_prediction_operations_called(
            result=result,
            ops=ops,
            input_mode=InputMode.SINGLE_TEXT,
            batch_size=8,
            use_cpu=True,
        )
        _assert_prediction_outputs(result=result, ops=ops)

    def test_resolves_selected_settings_from_config_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        prediction_csv: Path,
    ) -> None:
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="config_run",
        )
        config_path = tmp_path / "predict_config.yml"
        _write_yaml(
            config_path,
            f"""
            work_dir: {tmp_path.as_posix()}
            run_id: config_run
            batch_size: 4
            hardware:
              use_cpu: true
            """,
        )
        ops = _mock_prediction_operations(monkeypatch)

        result = api_mod.predict_tlmtc(
            prediction_csv,
            config_path=config_path,
        )

        assert result.paths.run_id == "config_run"

        _, predict_kwargs = ops.predict_probabilities.call_args
        assert predict_kwargs["batch_size"] == 4
        assert predict_kwargs["use_cpu"] is True

    def test_explicit_arguments_override_config_path_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        prediction_csv: Path,
    ) -> None:
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="config_run",
        )
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="explicit_run",
        )
        config_path = tmp_path / "predict_config.yml"
        _write_yaml(
            config_path,
            f"""
            work_dir: {tmp_path.as_posix()}
            run_id: config_run
            batch_size: 4
            hardware:
              use_cpu: false
            """,
        )
        ops = _mock_prediction_operations(monkeypatch)

        result = api_mod.predict_tlmtc(
            prediction_csv,
            config_path=config_path,
            run_id="explicit_run",
            batch_size=8,
            use_cpu=True,
        )

        assert result.paths.run_id == "explicit_run"

        _, predict_kwargs = ops.predict_probabilities.call_args
        assert predict_kwargs["batch_size"] == 8
        assert predict_kwargs["use_cpu"] is True

    def test_selects_latest_completed_training_run_when_run_id_is_omitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        prediction_csv: Path,
    ) -> None:
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="older_run",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="newer_run",
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        _mock_prediction_operations(monkeypatch)

        result = api_mod.predict_tlmtc(
            prediction_csv,
            work_dir=tmp_path,
        )

        assert result.paths.run_id == "newer_run"

    def test_uses_paired_text_training_contract_for_prediction(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        paired_prediction_csv: Path,
    ) -> None:
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="paired_run",
            input_mode=InputMode.PAIRED_TEXT,
        )
        ops = _mock_prediction_operations(monkeypatch, input_mode=InputMode.PAIRED_TEXT)

        result = api_mod.predict_tlmtc(
            paired_prediction_csv,
            work_dir=tmp_path,
            run_id="paired_run",
        )

        _assert_prediction_operations_called(
            result=result,
            ops=ops,
            input_mode=InputMode.PAIRED_TEXT,
            batch_size=32,
            use_cpu=False,
        )
        _assert_prediction_outputs(result=result, ops=ops)

    def test_applies_global_prediction_threshold(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        prediction_csv: Path,
    ) -> None:
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="global_threshold_run",
            thresholds=[0.5],
        )
        ops = _mock_prediction_operations(monkeypatch)

        result = api_mod.predict_tlmtc(
            prediction_csv,
            work_dir=tmp_path,
            run_id="global_threshold_run",
        )

        _assert_prediction_outputs(
            result=result,
            ops=ops,
            expected_binary_predictions=BINARY_PREDICTIONS,
        )

    def test_rejects_training_run_without_transfer_learning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        prediction_csv: Path,
    ) -> None:
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="data_only_run",
            transfer_learning=False,
        )
        read_prediction_csv_mock = MagicMock()
        monkeypatch.setattr(api_mod, "read_prediction_csv", read_prediction_csv_mock)

        with pytest.raises(RuntimeError, match="transfer_learning=True"):
            api_mod.predict_tlmtc(
                prediction_csv,
                work_dir=tmp_path,
                run_id="data_only_run",
            )

        read_prediction_csv_mock.assert_not_called()

    def test_configures_runtime_output_from_explicit_argument(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        prediction_csv: Path,
        configure_runtime_output_mock: MagicMock,
    ) -> None:
        _write_prediction_ready_train_run(
            work_dir=tmp_path,
            run_id="quiet_prediction_run",
        )
        _mock_prediction_operations(monkeypatch)

        api_mod.predict_tlmtc(
            prediction_csv,
            work_dir=tmp_path,
            run_id="quiet_prediction_run",
            verbosity="quiet",
        )

        configure_runtime_output_mock.assert_called_once_with("quiet", is_main_process=True)