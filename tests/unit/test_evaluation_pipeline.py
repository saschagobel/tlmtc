"""Tests for EvaluationPipeline."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict
from matplotlib.figure import Figure

from tlmtc.data_contracts import InputMode
from tlmtc.evaluation_pipeline import EvaluationPipeline
from tlmtc.paths import RunPaths, resolve_paths
from tlmtc.settings import ModelSettings, TrainingSettings, WorkflowSettings


@pytest.fixture
def paths(tmp_path: Path) -> RunPaths:
    """Create run paths with persisted train/test label splits."""
    run_paths = resolve_paths(
        labeled_data=tmp_path / "labeled.csv",
        raw_test_csv=None,
        work_dir=tmp_path,
        run_id="test-run",
    ).ensure_dirs()

    train_df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label_x": [1, 0, 1, 0],
            "label_y": [0, 1, 1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "text": ["e", "f", "g", "h"],
            "label_x": [1, 0, 1, 0],
            "label_y": [0, 1, 1, 0],
        }
    )

    train_df.to_parquet(run_paths.train_data_path, index=False)
    test_df.to_parquet(run_paths.test_data_path, index=False)
    return run_paths


@pytest.fixture
def tokenized_dataset() -> DatasetDict:
    """Minimal tokenized dataset with a test split."""
    test = Dataset.from_dict(
        {
            "input_ids": [[0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 0]],
            "attention_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "labels": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        }
    )
    return DatasetDict({"test": test})


@pytest.fixture
def fake_trainer() -> MagicMock:
    """Trainer-like object with deterministic predictions and log history."""
    trainer = MagicMock()
    trainer.predict.return_value = SimpleNamespace(
        predictions=np.array(
            [
                [3.0, -3.0],
                [-3.0, 3.0],
                [2.0, 2.0],
                [-2.0, -2.0],
            ],
            dtype=float,
        )
    )
    trainer.state = SimpleNamespace(
        log_history=[
            {"epoch": 1, "loss": 0.7},
            {"epoch": 1, "eval_loss": 0.6, "eval_f1_macro": 0.5},
            {"epoch": 2, "loss": 0.4},
            {"epoch": 2, "eval_loss": 0.3, "eval_f1_macro": 0.8},
        ]
    )
    return trainer


@pytest.fixture
def model_settings() -> ModelSettings:
    """Return model settings for evaluation reports."""
    return ModelSettings(
        target_name="Test Target",
        proxy_checkpoint="dummy-proxy",
        checkpoint="dummy-checkpoint",
        sequence_length=16,
    )


@pytest.fixture
def training_settings() -> TrainingSettings:
    """Return training settings used for best-epoch selection."""
    return TrainingSettings(
        batch_size=2,
        train_epochs=2,
        weight_decay=0.0,
        learning_rate=1e-3,
        lr_scheduler="linear",
        best_model_metric="f1_macro",
        early_stopping_patience=1,
    )


@pytest.fixture
def pipeline_factory(
    tokenized_dataset: DatasetDict,
    fake_trainer: MagicMock,
    paths: RunPaths,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
):
    """Factory for EvaluationPipeline instances."""

    def _factory(
        *,
        transfer_learning: bool = True,
        hyperparameter_tuning: bool = False,
        tuned_threshold: np.ndarray | None = None,
        input_mode: InputMode = InputMode.SINGLE_TEXT,
    ) -> EvaluationPipeline:
        workflow = WorkflowSettings(
            hyperparameter_tuning=hyperparameter_tuning,
            threshold_optimization=True,
            transfer_learning=transfer_learning,
            scale_learning_rate=False,
            wrap_peft=False,
        )
        return EvaluationPipeline(
            tokenized_dataset=tokenized_dataset,
            updated_trainer=fake_trainer,
            paths=paths,
            model=model_settings,
            workflow=workflow,
            training=training_settings,
            tuned_threshold=np.array([0.5], dtype=float) if tuned_threshold is None else tuned_threshold,
            input_mode=input_mode,
        )

    return _factory


class TestRunEvaluation:
    """Test suite for EvaluationPipeline.run_evaluation."""

    def test_populates_transfer_learning_outputs(self, pipeline_factory, fake_trainer):
        """Ensure run_evaluation computes and stores test-set evaluation artifacts."""
        pipeline = pipeline_factory()

        result = pipeline.run_evaluation()

        assert result is pipeline
        fake_trainer.predict.assert_called_once_with(pipeline.tokenized_dataset["test"])

        assert pipeline.label_names == ["x", "y"]
        assert pipeline.probabilities is not None
        assert pipeline.probabilities.shape == (4, 2)
        assert pipeline.true_labels is not None
        assert pipeline.pred_labels is not None
        assert pipeline.pred_labels.shape == (4, 2)

        assert pipeline.global_eval_metrics is not None
        assert pipeline.label_eval_metrics is not None
        assert set(pipeline.label_eval_metrics) == {"x", "y"}
        assert pipeline.roc_curves is not None
        assert pipeline.co_occurrence is not None
        assert pipeline.losses is not None
        assert pipeline.best_epoch == 2

    def test_returns_self_without_transfer_learning(self, pipeline_factory, fake_trainer):
        """Ensure run_evaluation is a no-op for transfer-learning outputs when disabled."""
        pipeline = pipeline_factory(transfer_learning=False, hyperparameter_tuning=False)

        result = pipeline.run_evaluation()

        assert result is pipeline
        fake_trainer.predict.assert_not_called()
        assert pipeline.global_eval_metrics is None
        assert pipeline.label_eval_metrics is None
        assert pipeline.hp_objective_values is None

    def test_loads_hpo_objective_values(self, pipeline_factory, paths, monkeypatch):
        """Ensure run_evaluation loads Optuna objective values when HPO is enabled."""
        paths.optuna_trials_path.touch()

        fake_study = MagicMock()
        fake_study.trials_dataframe.return_value = pd.DataFrame(
            {
                "number": [0, 1],
                "value": [0.4, 0.7],
            }
        )
        load_study_mock = MagicMock(return_value=fake_study)
        monkeypatch.setattr("tlmtc.evaluation_pipeline.optuna.load_study", load_study_mock)

        pipeline = pipeline_factory(transfer_learning=False, hyperparameter_tuning=True)

        result = pipeline.run_evaluation()

        assert result is pipeline
        load_study_mock.assert_called_once_with(
            study_name="Test_Target_optuna_study",
            storage=f"sqlite:///{paths.optuna_trials_path.as_posix()}",
        )
        pd.testing.assert_frame_equal(
            pipeline.hp_objective_values,
            pd.DataFrame({"number": [0, 1], "value": [0.4, 0.7]}),
        )

    def test_raises_when_hpo_database_missing(self, pipeline_factory):
        """Ensure run_evaluation raises when HPO is enabled but no Optuna DB exists."""
        pipeline = pipeline_factory(transfer_learning=False, hyperparameter_tuning=True)

        with pytest.raises(RuntimeError, match="Optuna study database not found"):
            pipeline.run_evaluation()

    def test_raises_when_hpo_study_has_no_objective_values(self, pipeline_factory, paths, monkeypatch):
        """Ensure run_evaluation raises when the Optuna study has no objective values."""
        paths.optuna_trials_path.touch()

        fake_study = MagicMock()
        fake_study.trials_dataframe.return_value = pd.DataFrame({"number": [], "value": []})
        monkeypatch.setattr("tlmtc.evaluation_pipeline.optuna.load_study", MagicMock(return_value=fake_study))

        pipeline = pipeline_factory(transfer_learning=False, hyperparameter_tuning=True)

        with pytest.raises(RuntimeError, match="does not contain any completed trials"):
            pipeline.run_evaluation()

    def test_applies_label_specific_thresholds(self, pipeline_factory) -> None:
        """Ensure run_evaluation supports one threshold per label."""
        pipeline = pipeline_factory(tuned_threshold=np.array([0.9, 0.1]))

        pipeline.run_evaluation()

        expected = np.array(
            [
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
            ]
        )
        np.testing.assert_array_equal(pipeline.pred_labels, expected)


class TestSaveMetrics:
    """Test suite for EvaluationPipeline.save_metrics."""

    def test_writes_metric_json_files(self, pipeline_factory, paths):
        """Ensure save_metrics writes global and label-specific metric JSON files."""
        pipeline = pipeline_factory()
        pipeline.run_evaluation().save_metrics()

        assert paths.global_metrics_path.exists()
        assert paths.label_metrics_path.exists()

        global_metrics = json.loads(paths.global_metrics_path.read_text(encoding="utf-8"))
        label_metrics = json.loads(paths.label_metrics_path.read_text(encoding="utf-8"))

        assert "f1_micro" in global_metrics
        assert set(label_metrics) == {"x", "y"}

    def test_noop_when_transfer_learning_disabled(self, pipeline_factory, paths):
        """Ensure save_metrics is a no-op when transfer learning is disabled."""
        pipeline = pipeline_factory(transfer_learning=False)

        result = pipeline.save_metrics()

        assert result is pipeline
        assert not paths.global_metrics_path.exists()
        assert not paths.label_metrics_path.exists()

    def test_raises_if_metrics_missing(self, pipeline_factory):
        """Ensure save_metrics requires run_evaluation first."""
        pipeline = pipeline_factory()

        with pytest.raises(RuntimeError, match="Evaluation metrics not found"):
            pipeline.save_metrics()


class FakeTable:
    """Minimal Great Tables stand-in."""

    def __init__(self, content: str) -> None:
        """Initialize the fake table with HTML content."""
        self.content = content

    def write_raw_html(self, path: Path) -> None:
        """Write fake HTML content."""
        path.write_text(self.content, encoding="utf-8")


class TestRenderTables:
    """Test suite for EvaluationPipeline.render_tables."""

    def test_writes_report_tables(self, pipeline_factory, paths, monkeypatch):
        """Ensure render_tables writes all expected HTML report tables."""
        make_global_metrics_table_mock = MagicMock(return_value=FakeTable("global"))
        make_label_metrics_table_mock = MagicMock(return_value=FakeTable("label"))
        make_hyperparameters_table_mock = MagicMock(return_value=FakeTable("hyperparameters"))

        monkeypatch.setattr(
            "tlmtc.evaluation_pipeline.make_global_metrics_table",
            make_global_metrics_table_mock,
        )
        monkeypatch.setattr(
            "tlmtc.evaluation_pipeline.make_label_metrics_table",
            make_label_metrics_table_mock,
        )
        monkeypatch.setattr(
            "tlmtc.evaluation_pipeline.make_hyperparameters_table",
            make_hyperparameters_table_mock,
        )

        pipeline = pipeline_factory()
        pipeline.run_evaluation().render_tables()

        assert paths.global_metrics_table_path.read_text(encoding="utf-8") == "global"
        assert paths.label_metrics_table_path.read_text(encoding="utf-8") == "label"
        assert paths.hyperparameters_table_path.read_text(encoding="utf-8") == "hyperparameters"

        assert make_global_metrics_table_mock.call_args.kwargs["input_mode"] == "Single text"
        assert make_label_metrics_table_mock.call_args.kwargs["input_mode"] == "Single text"
        assert make_hyperparameters_table_mock.call_args.kwargs["input_mode"] == "Single text"

    def test_passes_paired_input_mode_to_report_tables(self, pipeline_factory, monkeypatch):
        """Ensure render_tables labels paired-text input mode in table metadata."""
        make_global_metrics_table_mock = MagicMock(return_value=FakeTable("global"))
        make_label_metrics_table_mock = MagicMock(return_value=FakeTable("label"))
        make_hyperparameters_table_mock = MagicMock(return_value=FakeTable("hyperparameters"))

        monkeypatch.setattr(
            "tlmtc.evaluation_pipeline.make_global_metrics_table",
            make_global_metrics_table_mock,
        )
        monkeypatch.setattr(
            "tlmtc.evaluation_pipeline.make_label_metrics_table",
            make_label_metrics_table_mock,
        )
        monkeypatch.setattr(
            "tlmtc.evaluation_pipeline.make_hyperparameters_table",
            make_hyperparameters_table_mock,
        )

        pipeline = pipeline_factory(input_mode=InputMode.PAIRED_TEXT)
        pipeline.run_evaluation().render_tables()

        assert make_global_metrics_table_mock.call_args.kwargs["input_mode"] == "Paired text"
        assert make_label_metrics_table_mock.call_args.kwargs["input_mode"] == "Paired text"
        assert make_hyperparameters_table_mock.call_args.kwargs["input_mode"] == "Paired text"

    def test_noop_when_transfer_learning_disabled(self, pipeline_factory, paths):
        """Ensure render_tables is a no-op when transfer learning is disabled."""
        pipeline = pipeline_factory(transfer_learning=False)

        result = pipeline.render_tables()

        assert result is pipeline
        assert not paths.global_metrics_table_path.exists()
        assert not paths.label_metrics_table_path.exists()
        assert not paths.hyperparameters_table_path.exists()

    def test_raises_if_metrics_missing(self, pipeline_factory):
        """Ensure render_tables requires run_evaluation first."""
        pipeline = pipeline_factory()

        with pytest.raises(RuntimeError, match="Evaluation metrics not found"):
            pipeline.render_tables()


class TestRenderFigures:
    """Test suite for EvaluationPipeline.render_figures."""

    def test_writes_transfer_learning_figures(self, pipeline_factory, paths, monkeypatch):
        """Ensure render_figures writes ROC, co-occurrence, and loss plots."""
        monkeypatch.setattr("tlmtc.evaluation_pipeline.make_roc_curves_plot", MagicMock(return_value=Figure()))
        monkeypatch.setattr(
            "tlmtc.evaluation_pipeline.make_cooccurrence_heatmaps_plot", MagicMock(return_value=Figure())
        )
        monkeypatch.setattr("tlmtc.evaluation_pipeline.make_loss_curves_plot", MagicMock(return_value=Figure()))

        pipeline = pipeline_factory()
        pipeline.run_evaluation().render_figures()

        assert paths.roc_plot_path.exists()
        assert paths.co_occurrence_plot_path.exists()
        assert paths.loss_plot_path.exists()

    def test_writes_hpo_objective_figure(self, pipeline_factory, paths, monkeypatch):
        """Ensure render_figures writes the HPO objective-value plot when HPO is enabled."""
        monkeypatch.setattr("tlmtc.evaluation_pipeline.make_objective_values_plot", MagicMock(return_value=Figure()))

        pipeline = pipeline_factory(transfer_learning=False, hyperparameter_tuning=True)
        pipeline.hp_objective_values = pd.DataFrame({"number": [0, 1], "value": [0.4, 0.7]})

        pipeline.render_figures()

        assert paths.objective_values_plot_path.exists()

    def test_raises_if_hpo_objective_values_missing(self, pipeline_factory):
        """Ensure render_figures requires HPO objective values when HPO is enabled."""
        pipeline = pipeline_factory(transfer_learning=False, hyperparameter_tuning=True)

        with pytest.raises(RuntimeError, match="Hyperparameter tuning results not found"):
            pipeline.render_figures()

    def test_raises_if_figure_data_missing(self, pipeline_factory):
        """Ensure render_figures requires run_evaluation first for transfer-learning plots."""
        pipeline = pipeline_factory()

        with pytest.raises(RuntimeError, match="Evaluation figure data not found"):
            pipeline.render_figures()
