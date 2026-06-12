"""Tests for FinetunePipeline."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
import torch
from datasets import Dataset, DatasetDict
from optuna.trial import FixedTrial
from transformers import TrainingArguments

from tlmtc.finetune_pipeline import FinetunePipeline
from tlmtc.paths import resolve_paths
from tlmtc.settings import (
    HardwareSettings,
    HpoSettings,
    ModelSettings,
    PeftSettings,
    ThresholdSettings,
    TrainingSettings,
    WorkflowSettings,
)


@pytest.fixture
def dummy_train_parquet(tmp_path):
    """Minimal train parquet file with multi-label columns."""
    df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "label_x": [1, 0],
            "label_y": [0, 1],
        }
    )
    path = tmp_path / "train.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def pipeline_factory(tmp_path, base_search_space):
    """Factory for constructing a FinetunePipeline with minimal test settings."""

    def _factory(
        train_path,
        *,
        tokenized_dataset=None,
        wrap_peft: bool = False,
        transfer_learning: bool = True,
        hyperparameter_tuning: bool = False,
        threshold_optimization: bool = False,
        scale_learning_rate: bool = False,
        optuna_space=None,
        tuning_trials: int = 1,
        model_checkpoint: str = "dummy",
        target_name: str = "dummy",
    ):
        paths = resolve_paths(
            raw_csv=tmp_path / "raw.csv",
            raw_test_csv=tmp_path / "raw_test.csv",
            work_dir=tmp_path,
            run_id="test-run",
        )

        paths = replace(paths, train_data_path=train_path).ensure_dirs()

        model = ModelSettings(
            target_name=target_name,
            proxy_checkpoint="dummy_proxy",
            checkpoint=model_checkpoint,
            sequence_length=16,
        )

        workflow = WorkflowSettings(
            hyperparameter_tuning=hyperparameter_tuning,
            threshold_optimization=threshold_optimization,
            transfer_learning=transfer_learning,
            scale_learning_rate=scale_learning_rate,
            wrap_peft=wrap_peft,
        )

        peft = PeftSettings(lora_r=1, lora_alpha=1, lora_dropout=0.1, lora_bias="none")

        training = TrainingSettings(
            batch_size=2,
            train_epochs=1,
            weight_decay=0.0,
            learning_rate=1e-3,
            lr_scheduler="linear",
            best_model_metric="f1_macro",
            early_stopping_patience=1,
        )

        hpo = HpoSettings(
            tuning_trials=tuning_trials,
            optuna_space=base_search_space if optuna_space is None else optuna_space,
        )
        threshold = ThresholdSettings(threshold_type="global", best_threshold_metric="f1_macro")
        hardware = HardwareSettings(use_cpu=True)

        return FinetunePipeline(
            tokenized_dataset=tokenized_dataset,
            paths=paths,
            model=model,
            workflow=workflow,
            peft=peft,
            training=training,
            hpo=hpo,
            threshold=threshold,
            hardware=hardware,
        )

    return _factory


@pytest.fixture
def tokenized_dataset():
    """Minimal, Trainer-compatible tokenized dataset."""
    train = Dataset.from_dict(
        {
            "input_ids": [[0, 1, 2]],
            "attention_mask": [[1, 1, 1]],
            "labels": [[1.0, 0.0]],
        }
    )
    val = Dataset.from_dict(
        {
            "input_ids": [[2, 1, 0]],
            "attention_mask": [[1, 1, 1]],
            "labels": [[0.0, 1.0]],
        }
    )
    return DatasetDict({"train": train, "validation": val})


@pytest.fixture
def base_search_space():
    """Baseline Optuna search space for hyperparameter tuning."""
    return {
        "lr_low": 1e-5,
        "lr_high": 1e-3,
        "batch_sizes": [8, 16],
        "wd_low": 0.0,
        "wd_high": 0.1,
        "schedulers": ["linear"],
        "epoch_low": 1,
        "epoch_high": 3,
        "lr_reference_batch_size": 16,
    }


@pytest.fixture
def pipeline_with_tokenized_hpo(pipeline_factory, dummy_train_parquet, tokenized_dataset, base_search_space):
    """FinetunePipeline configured for hyperparameter tuning with a tokenized dataset."""
    return pipeline_factory(
        train_path=dummy_train_parquet,
        tokenized_dataset=tokenized_dataset,
        hyperparameter_tuning=True,
        optuna_space=base_search_space,
    )


@pytest.fixture
def patch_hf_hyperparameter_search(monkeypatch):
    """Patch to disable real HF hyperparameter search and Optuna runs."""

    def _fake_hp_search(*_args, **_kwargs):
        return SimpleNamespace(
            hyperparameters={
                "learning_rate": 1e-4,
                "lr_scheduler_type": "linear",
                "per_device_train_batch_size": 8,
                "weight_decay": 0.0,
                "num_train_epochs": 1,
            }
        )

    monkeypatch.setattr(
        "transformers.Trainer.hyperparameter_search",
        _fake_hp_search,
        raising=True,
    )


@pytest.fixture
def patch_model_init(monkeypatch):
    """Patch to replace make_model_init with a tiny torch model factory."""

    def _fake_model_init(*_args, **kwargs):
        num_labels = kwargs.get("num_labels", 2)

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_labels = num_labels

            def forward(self, input_ids=None, **_):
                batch = 1 if input_ids is None else input_ids.shape[0]
                logits = torch.zeros((batch, self.num_labels), dtype=torch.float32)
                return SimpleNamespace(logits=logits)

        return lambda *_: _Model()

    monkeypatch.setattr(
        "tlmtc.finetune_pipeline.make_model_init",
        _fake_model_init,
        raising=True,
    )


@pytest.fixture
def fake_trainer():
    """Fake Trainer that records hyperparameter_search calls."""
    trainer = MagicMock()

    trainer.train.return_value = None
    trainer.evaluate.return_value = {"eval_f1_macro": 0.5}
    trainer.predict.return_value = SimpleNamespace(predictions=[[0.0]])

    trainer.hp_search_calls = []

    def _fake_hp_search(*_args, **kwargs):
        trainer.hp_search_calls.append(kwargs)
        return SimpleNamespace(
            hyperparameters={
                "learning_rate": 1e-4,
                "lr_scheduler_type": "linear",
                "per_device_train_batch_size": 8,
                "weight_decay": 0.01,
                "num_train_epochs": 2,
            }
        )

    trainer.hyperparameter_search.side_effect = _fake_hp_search

    return trainer


@pytest.mark.usefixtures("patch_hf_hyperparameter_search", "patch_model_init")
class TestTuneHyperparameters:
    """Test suite for FinetunePipeline.tune_hyperparameters."""

    def test_returns_self_when_disabled(self, pipeline_factory, dummy_train_parquet):
        """Ensure tune_hyperparameters is a no-op and returns self when hyperparameter_tuning is False."""
        pipeline = pipeline_factory(dummy_train_parquet, tokenized_dataset=None, hyperparameter_tuning=False)

        assert pipeline.tokenized_dataset is None
        assert pipeline.num_labels is None

        result = pipeline.tune_hyperparameters()

        assert result is pipeline
        assert pipeline.tokenized_dataset is None
        assert pipeline.num_labels is None

    def test_requires_tokenized_dataset(self, pipeline_factory, dummy_train_parquet):
        """Ensure tune_hyperparameters raises a RuntimeError when no tokenized dataset is set."""
        pipeline = pipeline_factory(dummy_train_parquet, tokenized_dataset=None, hyperparameter_tuning=True)

        with pytest.raises(RuntimeError, match="Tokenized dataset not found"):
            pipeline.tune_hyperparameters()

    def test_sets_num_labels_from_train_data(self, pipeline_with_tokenized_hpo):
        """Ensure num_labels is inferred from the multilabel train data when unset."""
        pipeline = pipeline_with_tokenized_hpo
        pipeline.num_labels = None

        pipeline.tune_hyperparameters()

        assert pipeline.num_labels == 2

    @pytest.mark.parametrize("wrap_peft", [False, True])
    def test_configures_hp_search_from_hpo_settings(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        base_search_space,
        fake_trainer,
        wrap_peft,
    ):
        """Ensure hp_search kwargs reflect HpoSettings and storage paths."""
        pipeline = pipeline_factory(
            dummy_train_parquet,
            tokenized_dataset=tokenized_dataset,
            hyperparameter_tuning=True,
            wrap_peft=wrap_peft,
            optuna_space=base_search_space,
            tuning_trials=7,
        )

        def trainer_factory(*_args, **_kwargs):
            return fake_trainer

        result = pipeline.tune_hyperparameters(trainer=trainer_factory)
        assert result is pipeline

        assert fake_trainer.hp_search_calls, "hyperparameter_search was never called"
        assert len(fake_trainer.hp_search_calls) == 1

        hp_search_kwargs = fake_trainer.hp_search_calls[0]
        hp_space_fn = hp_search_kwargs["hp_space"]

        assert callable(hp_space_fn)

        assert hp_search_kwargs["direction"] == "maximize"
        assert hp_search_kwargs["backend"] == "optuna"
        assert hp_search_kwargs["n_trials"] == pipeline.hpo.tuning_trials
        assert hp_search_kwargs["study_name"] == f"{pipeline.model.target_name.replace(' ', '_')}_optuna_study"

        expected_storage = f"sqlite:///{pipeline.paths.optuna_trials_path.as_posix()}"
        assert hp_search_kwargs["storage"] == expected_storage

        assert callable(hp_search_kwargs["compute_objective"])
        assert hp_search_kwargs["load_if_exists"] is True
        assert hp_search_kwargs["catch"] == (ValueError,)

    def test_instantiates_trainer_with_expected_arguments(
        self,
        pipeline_with_tokenized_hpo,
        fake_trainer,
    ):
        """Ensure tune_hyperparameters instantiates the Trainer with expected inputs."""
        pipeline = pipeline_with_tokenized_hpo
        recorded: dict = {}

        def trainer_factory(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return fake_trainer

        pipeline.tune_hyperparameters(trainer=trainer_factory)

        assert recorded, "Trainer was never instantiated by tune_hyperparameters"
        assert recorded["args"] == ()
        kwargs = recorded["kwargs"]

        assert kwargs["model"] is None
        assert isinstance(kwargs["args"], TrainingArguments)

        assert kwargs["train_dataset"] is pipeline.tokenized_dataset["train"]
        assert kwargs["eval_dataset"] is pipeline.tokenized_dataset["validation"]

        assert callable(kwargs["compute_metrics"])

        assert isinstance(kwargs["class_weights"], torch.Tensor)
        assert callable(kwargs["model_init"])

        model_instance = kwargs["model_init"]()
        assert isinstance(model_instance, torch.nn.Module)
        assert getattr(model_instance, "num_labels") == 2

        training_args = kwargs["args"]

        assert training_args.output_dir == str(pipeline.paths.hpo_checkpoints_dir)

    @pytest.mark.parametrize("scale_learning_rate", [False, True])
    def test_updates_runtime_training_hyperparameters_from_best_run(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        base_search_space,
        fake_trainer,
        monkeypatch,
        scale_learning_rate,
    ):
        """Ensure best hyperparameters are applied to runtime training state, not resolved settings."""
        pipeline = pipeline_factory(
            dummy_train_parquet,
            tokenized_dataset=tokenized_dataset,
            hyperparameter_tuning=True,
            optuna_space=base_search_space,
            scale_learning_rate=scale_learning_rate,
            wrap_peft=False,
        )

        pipeline.runtime_training.learning_rate = 9.9
        pipeline.runtime_training.lr_scheduler = "cosine"
        pipeline.runtime_training.batch_size = 999
        pipeline.runtime_training.weight_decay = 0.5
        pipeline.runtime_training.train_epochs = 10

        original_training = pipeline.training.model_copy(deep=True)

        scaled_lr = 5e-5
        mock_get_scaled_lr = MagicMock(return_value=scaled_lr)
        monkeypatch.setattr(
            "tlmtc.finetune_pipeline.get_scaled_lr",
            mock_get_scaled_lr,
            raising=True,
        )

        def trainer_factory(*_args, **_kwargs):
            return fake_trainer

        pipeline.tune_hyperparameters(trainer=trainer_factory)

        if scale_learning_rate:
            assert pipeline.runtime_training.learning_rate == scaled_lr
            mock_get_scaled_lr.assert_called_once_with(
                learning_rate=1e-4,
                checkpoint=pipeline.model.checkpoint,
                proxy_checkpoint=pipeline.model.proxy_checkpoint,
                peft=pipeline.workflow.wrap_peft,
            )
        else:
            assert pipeline.runtime_training.learning_rate == 1e-4
            mock_get_scaled_lr.assert_not_called()

        assert pipeline.runtime_training.lr_scheduler == "linear"
        assert pipeline.runtime_training.batch_size == 8
        assert pipeline.runtime_training.weight_decay == 0.01
        assert pipeline.runtime_training.train_epochs == 2

        assert pipeline.training == original_training

    def test_suppresses_trainer_console_callbacks(
        self,
        pipeline_with_tokenized_hpo,
        fake_trainer,
        monkeypatch,
    ):
        """Ensure HPO Trainer console callbacks are suppressed after Trainer construction."""
        pipeline = pipeline_with_tokenized_hpo

        suppress_mock = MagicMock(side_effect=lambda trainer: trainer)
        monkeypatch.setattr(
            "tlmtc.finetune_pipeline.suppress_trainer_console_callbacks",
            suppress_mock,
            raising=True,
        )

        def trainer_factory(*_args, **_kwargs):
            return fake_trainer

        result = pipeline.tune_hyperparameters(trainer=trainer_factory)

        assert result is pipeline
        suppress_mock.assert_called_once_with(fake_trainer)
        fake_trainer.hyperparameter_search.assert_called_once()

    def test_hpo_hp_space_emits_trial_progress_once_per_trial(
        self,
        pipeline_with_tokenized_hpo,
        fake_trainer,
        monkeypatch,
    ):
        """Ensure HPO hp_space emits per-trial progress."""
        monkeypatch.setattr(
            "tlmtc.finetune_pipeline.get_existing_trial_count",
            MagicMock(return_value=3),
            raising=True,
        )

        emit_progress_mock = MagicMock()
        monkeypatch.setattr(
            "tlmtc.finetune_pipeline.emit_progress",
            emit_progress_mock,
            raising=True,
        )

        pipeline_with_tokenized_hpo.tune_hyperparameters(
            trainer=lambda **_: fake_trainer,
        )

        hp_space_fn = fake_trainer.hp_search_calls[0]["hp_space"]

        trial = FixedTrial(
            {
                "learning_rate": 1e-4,
                "per_device_train_batch_size": 8,
                "weight_decay": 0.01,
                "lr_scheduler_type": "linear",
                "num_train_epochs": 2,
            },
            number=3,
        )

        hp_space_fn(trial)
        hp_space_fn(trial)

        emit_progress_mock.assert_any_call("HPO trial 4/4 started")

        assert emit_progress_mock.mock_calls.count(call("HPO trial 4/4 started")) == 1

    def test_uses_broadcast_hyperparameters_when_hpo_returns_none(
        self,
        pipeline_with_tokenized_hpo,
        fake_trainer,
    ):
        """Ensure non-main DDP ranks apply rank-zero best hyperparameters after broadcast."""
        pipeline = pipeline_with_tokenized_hpo

        fake_trainer.hyperparameter_search.return_value = None
        fake_trainer.hyperparameter_search.side_effect = None

        broadcast_value = MagicMock(
            return_value={
                "learning_rate": 2e-4,
                "lr_scheduler_type": "cosine",
                "per_device_train_batch_size": 16,
                "weight_decay": 0.02,
                "num_train_epochs": 3,
            }
        )

        pipeline.tune_hyperparameters(
            trainer=lambda **_: fake_trainer,
            broadcast_value=broadcast_value,
        )

        broadcast_value.assert_called_once_with(None)
        assert pipeline.runtime_training.learning_rate == 2e-4
        assert pipeline.runtime_training.lr_scheduler == "cosine"
        assert pipeline.runtime_training.batch_size == 16
        assert pipeline.runtime_training.weight_decay == 0.02
        assert pipeline.runtime_training.train_epochs == 3


@pytest.mark.usefixtures("patch_model_init")
class TestFineTunePretrained:
    """Test suite for FinetunePipeline.fine_tune_pretrained."""

    @pytest.mark.parametrize("transfer_learning, expected_call", [(True, True), (False, False)])
    def test_noop_when_transfer_learning_disabled(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        fake_trainer,
        monkeypatch,
        transfer_learning,
        expected_call,
    ):
        """Ensure fine_tune_pretrained is a no-op when transfer_learning is disabled."""
        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            tokenized_dataset=tokenized_dataset,
            transfer_learning=transfer_learning,
            hyperparameter_tuning=False,
        )

        mock_get_class_weights = MagicMock(return_value=torch.ones(2))
        monkeypatch.setattr("tlmtc.finetune_pipeline.get_class_weights", mock_get_class_weights, raising=True)

        recorded: dict[str, Any] = {}

        def trainer_factory(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return fake_trainer

        result = pipeline.fine_tune_pretrained(trainer=trainer_factory)
        assert result is pipeline

        if expected_call:
            mock_get_class_weights.assert_called_once()
            fake_trainer.train.assert_called_once()
            assert "kwargs" in recorded
            assert pipeline.updated_trainer is fake_trainer
        else:
            mock_get_class_weights.assert_not_called()
            fake_trainer.train.assert_not_called()
            assert recorded == {}
            assert pipeline.updated_trainer is None

    def test_requires_tokenized_dataset(self, pipeline_factory, dummy_train_parquet):
        """..."""
        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            tokenized_dataset=None,
            transfer_learning=True,
        )

        with pytest.raises(RuntimeError, match="Tokenized dataset not found"):
            pipeline.fine_tune_pretrained()

    def test_instantiates_trainer_with_expected_arguments(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        fake_trainer,
    ):
        """Ensure fine_tune_pretrained instantiates Trainer with expected arguments."""
        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            tokenized_dataset=tokenized_dataset,
            transfer_learning=True,
            hyperparameter_tuning=False,
        )

        recorded: dict[str, Any] = {}

        def trainer_factory(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return fake_trainer

        pipeline.fine_tune_pretrained(trainer=trainer_factory)

        assert recorded, "Trainer was never instantiated by fine_tune_pretrained"
        assert recorded["args"] == ()
        kwargs = recorded["kwargs"]

        assert kwargs["model"] is None
        assert callable(kwargs["model_init"])

        model_instance = kwargs["model_init"]()
        assert isinstance(model_instance, torch.nn.Module)
        assert getattr(model_instance, "num_labels") == 2

        assert kwargs["train_dataset"] is pipeline.tokenized_dataset["train"]
        assert kwargs["eval_dataset"] is pipeline.tokenized_dataset["validation"]

        training_args = kwargs["args"]
        assert isinstance(training_args, TrainingArguments)

        assert training_args.learning_rate == pytest.approx(pipeline.runtime_training.learning_rate)
        assert training_args.num_train_epochs == pipeline.runtime_training.train_epochs
        assert training_args.per_device_train_batch_size == pipeline.runtime_training.batch_size
        assert training_args.weight_decay == pipeline.runtime_training.weight_decay
        assert training_args.lr_scheduler_type == pipeline.runtime_training.lr_scheduler
        assert training_args.metric_for_best_model == pipeline.training.best_model_metric
        assert training_args.use_cpu == pipeline.hardware.use_cpu

        assert callable(kwargs["compute_metrics"])
        assert isinstance(kwargs["class_weights"], torch.Tensor)

        callbacks = kwargs["callbacks"]
        assert isinstance(callbacks, list)
        assert callbacks, "Expected at least one callback for early stopping"
        assert callbacks[0].early_stopping_patience == pipeline.training.early_stopping_patience

        fake_trainer.train.assert_called_once()
        assert pipeline.updated_trainer is fake_trainer

    def test_uses_runtime_training_state_for_training_arguments(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        fake_trainer,
    ):
        """Ensure fine_tune_pretrained consumes runtime training state rather than resolved training settings."""
        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            tokenized_dataset=tokenized_dataset,
            transfer_learning=True,
            hyperparameter_tuning=False,
        )

        pipeline.runtime_training.learning_rate = 5e-5
        pipeline.runtime_training.train_epochs = 3
        pipeline.runtime_training.batch_size = 8
        pipeline.runtime_training.weight_decay = 0.02
        pipeline.runtime_training.lr_scheduler = "cosine"

        recorded: dict[str, Any] = {}

        def trainer_factory(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return fake_trainer

        pipeline.fine_tune_pretrained(trainer=trainer_factory)

        training_args = recorded["kwargs"]["args"]

        assert training_args.learning_rate == pytest.approx(5e-5)
        assert training_args.num_train_epochs == 3
        assert training_args.per_device_train_batch_size == 8
        assert training_args.weight_decay == 0.02
        assert training_args.lr_scheduler_type == "cosine"

        assert training_args.metric_for_best_model == pipeline.training.best_model_metric

    def test_suppresses_trainer_console_callbacks_without_removing_early_stopping(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        fake_trainer,
        monkeypatch,
    ):
        """Ensure final Trainer console callbacks are suppressed while preserving early stopping."""
        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            tokenized_dataset=tokenized_dataset,
            transfer_learning=True,
            hyperparameter_tuning=False,
        )

        suppress_mock = MagicMock(side_effect=lambda trainer: trainer)
        monkeypatch.setattr(
            "tlmtc.finetune_pipeline.suppress_trainer_console_callbacks",
            suppress_mock,
            raising=True,
        )

        recorded: dict[str, Any] = {}

        def trainer_factory(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return fake_trainer

        result = pipeline.fine_tune_pretrained(trainer=trainer_factory)

        assert result is pipeline
        suppress_mock.assert_called_once_with(fake_trainer)

        callbacks = recorded["kwargs"]["callbacks"]
        assert isinstance(callbacks, list)
        assert callbacks, "Expected early stopping callback to be preserved"
        assert callbacks[0].early_stopping_patience == pipeline.training.early_stopping_patience

        fake_trainer.train.assert_called_once()
        assert pipeline.updated_trainer is fake_trainer


class TestTuneThresholds:
    """Test suite for FinetunePipeline.tune_thresholds."""

    @pytest.mark.parametrize(
        "threshold_optimization, transfer_learning",
        [(False, True), (True, False), (False, False)],
    )
    def test_returns_self_when_disabled(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        monkeypatch,
        threshold_optimization,
        transfer_learning,
    ):
        """Ensure tune_thresholds is a no-op unless threshold_optimization and transfer_learning are enabled."""
        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            tokenized_dataset=tokenized_dataset,
            threshold_optimization=threshold_optimization,
            transfer_learning=transfer_learning,
        )

        pipeline.updated_trainer = MagicMock()
        find_mock = MagicMock()
        monkeypatch.setattr("tlmtc.finetune_pipeline.find_optimal_threshold", find_mock, raising=True)

        result = pipeline.tune_thresholds()

        assert result is pipeline
        pipeline.updated_trainer.predict.assert_not_called()
        find_mock.assert_not_called()

    @pytest.mark.parametrize(
        "tokenized_dataset_present, updated_trainer, expected_msg",
        [
            (False, MagicMock(), "Tokenized dataset not found"),
            (True, None, "Trained model not found"),
        ],
    )
    def test_requires_prerequisites(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        tokenized_dataset_present,
        updated_trainer,
        expected_msg,
    ):
        """Ensure tune_thresholds raises when prerequisites are missing."""
        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            tokenized_dataset=tokenized_dataset if tokenized_dataset_present else None,
            threshold_optimization=True,
            transfer_learning=True,
        )
        pipeline.updated_trainer = updated_trainer

        with pytest.raises(RuntimeError, match=expected_msg):
            pipeline.tune_thresholds()

    def test_predicts_on_validation_and_sets_tuned_threshold(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        monkeypatch,
    ):
        """Ensure tune_thresholds predicts on validation and stores the tuned threshold."""
        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            tokenized_dataset=tokenized_dataset,
            threshold_optimization=True,
            transfer_learning=True,
        )

        fake_trainer = MagicMock()
        fake_trainer.predict.return_value = SimpleNamespace(predictions=[[0.0, 0.0]])
        pipeline.updated_trainer = fake_trainer

        tuned = object()
        find_mock = MagicMock(return_value=tuned)
        monkeypatch.setattr("tlmtc.finetune_pipeline.find_optimal_threshold", find_mock, raising=True)

        result = pipeline.tune_thresholds()

        assert result is pipeline
        fake_trainer.predict.assert_called_once_with(pipeline.tokenized_dataset["validation"])
        assert pipeline.tuned_threshold is tuned

        _, kwargs = find_mock.call_args
        assert kwargs["best_threshold_metric"] == pipeline.threshold.best_threshold_metric
        assert kwargs["threshold_type"] == pipeline.threshold.threshold_type
        assert "y_true" in kwargs
        assert "y_prob" in kwargs


class TestSavePretrained:
    """Test suite for FinetunePipeline.save_pretrained."""

    def test_noop_when_transfer_learning_disabled(self, pipeline_factory, dummy_train_parquet):
        """Ensure save_pretrained is a no-op when transfer_learning is disabled."""
        pipeline = pipeline_factory(dummy_train_parquet, transfer_learning=False)

        assert pipeline.paths.model_dir.exists()
        assert list(pipeline.paths.model_dir.iterdir()) == []

        result = pipeline.save_pretrained()

        assert result is pipeline
        assert pipeline.updated_trainer is None
        assert list(pipeline.paths.model_dir.iterdir()) == []

    def test_raises_if_trainer_missing(self, pipeline_factory, dummy_train_parquet):
        """Ensure save_pretrained raises when the fine-tuned Trainer is missing."""
        pipeline = pipeline_factory(dummy_train_parquet)
        assert pipeline.workflow.transfer_learning is True
        assert pipeline.updated_trainer is None

        with pytest.raises(RuntimeError, match="Instantiated Trainer after fine-tuning not found"):
            pipeline.save_pretrained()

    def test_delegates_to_trainer_save_model_with_output_path(self, pipeline_factory, dummy_train_parquet):
        """Ensure save_pretrained delegates model artifact writing to Trainer."""
        pipeline = pipeline_factory(dummy_train_parquet)

        fake_trainer = MagicMock()
        pipeline.updated_trainer = fake_trainer

        assert pipeline.paths.model_dir.exists()
        assert list(pipeline.paths.model_dir.iterdir()) == []

        result = pipeline.save_pretrained()

        assert result is pipeline
        fake_trainer.save_model.assert_called_once_with(str(pipeline.paths.model_dir))
