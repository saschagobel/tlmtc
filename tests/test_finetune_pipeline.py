"""Tests for the WeightedTrainer and FinetunePipeline class."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import TrainingArguments

from tlmtc.finetune_pipeline import FinetunePipeline, WeightedTrainer
from tlmtc.hpo import optuna_hp_space


class DummyModel(nn.Module):
    """Minimal feed-forward classifier used for testing the `WeightedTrainer`."""

    def __init__(self, num_labels=3):
        """Initialize the dummy model."""
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(4, num_labels)

    def forward(self, input_ids=None):
        """Compute logits for a batch of inputs."""
        logits = self.linear(input_ids.float())
        return type("Output", (), {"logits": logits})


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
def pipeline_factory(tmp_path):
    """Return a factory for constructing FinetunePipeline instances for tests."""

    def _factory(train_path, wrap_peft=False, transfer_learning=True):
        """Construct a FinetunePipeline with controlled configuration for testing."""
        return FinetunePipeline(
            tokenized_dataset=None,
            train_data_path=train_path,
            val_data_path=tmp_path / "val.parquet",
            output_logging_path=tmp_path / "logs",
            output_model_path=tmp_path / "model",
            target_name="dummy",
            proxy_checkpoint="dummy_proxy",
            checkpoint="dummy",
            transfer_learning=transfer_learning,
            hyperparameter_tuning=False,
            threshold_optimization=False,
            threshold_type="global",
            scale_learning_rate=False,
            wrap_peft=wrap_peft,
            optuna_space_default_base={},
            optuna_space_default_peft={},
            tuning_trials=1,
            batch_size=2,
            weight_decay=0.0,
            learning_rate=1e-3,
            lr_scheduler="linear",
            epochs=1,
            best_model_metric="f1_macro",
            best_threshold_metric="f1_macro",
            early_stopping_patience=1,
            lora_r=1,
            lora_alpha=1,
            lora_dropout=0.1,
            lora_bias="none",
            use_cpu=True,
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
    }


@pytest.fixture
def pipeline_with_tokenized(pipeline_factory, dummy_train_parquet, tokenized_dataset, base_search_space):
    """Construct a pipeline ready for tuning."""
    pipeline = pipeline_factory(dummy_train_parquet)
    pipeline.tokenized_dataset = tokenized_dataset
    pipeline.hyperparameter_tuning = True
    pipeline.optuna_space_default_base = base_search_space
    return pipeline


@pytest.fixture(autouse=True)
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


@pytest.fixture(autouse=True)
def patch_model_init(monkeypatch):
    """Patch to replace _make_model_init with a DummyModel factory."""

    def _fake_model_init(*_args, **kwargs):
        return lambda *_: DummyModel(num_labels=kwargs.get("num_labels", 2))

    monkeypatch.setattr(
        "tlmtc.finetune_pipeline.make_model_init",
        _fake_model_init,
        raising=True,
    )


@pytest.fixture(autouse=True)
def patch_class_weights(monkeypatch):
    """Patch to stub _get_class_weights with unit weights per label."""
    monkeypatch.setattr(
        "tlmtc.finetune_pipeline.get_class_weights",
        lambda *args, **kwargs: torch.ones(2),
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


class TestWeightedTrainer:
    """Test suite for the WeightedTrainer class."""

    def test_loss_respects_class_weights(self, tmp_path):
        """Ensure that applying class weights increases the computed BCE loss."""
        model = DummyModel(num_labels=3)

        inputs = {
            "labels": torch.tensor([[1, 0, 1], [0, 1, 0]]).float(),
            "input_ids": torch.zeros((2, 4)),
        }

        args = TrainingArguments(
            output_dir=str(tmp_path / "trainer_output"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            report_to="none",
        )

        unweighted = WeightedTrainer(model=model, args=args, class_weights=None)
        loss_unweighted = unweighted.compute_loss(model, dict(inputs))

        cw = torch.tensor([5.0, 5.0, 5.0])
        weighted = WeightedTrainer(model=model, args=args, class_weights=cw)
        loss_weighted = weighted.compute_loss(model, dict(inputs))

        assert loss_weighted > loss_unweighted

    def test_returns_loss_and_outputs_tuple(self, tmp_path):
        """Test that compute_loss returns (loss, outputs) when return_outputs is True."""
        model = DummyModel(num_labels=3)
        args = TrainingArguments(output_dir=str(tmp_path / "trainer_output"), report_to="none")

        trainer = WeightedTrainer(model=model, args=args)
        inputs = {"labels": torch.zeros((1, 3)), "input_ids": torch.zeros((1, 4))}

        loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

        assert isinstance(loss, torch.Tensor)
        assert hasattr(outputs, "logits")

    def test_reads_num_labels_from_model_module(self, tmp_path):
        """Ensure that compute_loss reads num_labels correctly from model.module when present."""
        model = torch.nn.DataParallel(DummyModel(num_labels=4))
        args = TrainingArguments(output_dir=str(tmp_path / "trainer_output"), report_to="none")

        trainer = WeightedTrainer(model=model, args=args)
        inputs = {"labels": torch.zeros((1, 4)), "input_ids": torch.zeros((1, 4))}

        loss = trainer.compute_loss(model, inputs)
        assert isinstance(loss, torch.Tensor)


class TestLoadPretrained:
    """Test suite for FinetunePipeline.load_pretrained."""

    @pytest.mark.parametrize("transfer_learning, expected_call", [(True, True), (False, False)])
    def test_respects_transfer_learning_flag(
        self,
        pipeline_factory,
        dummy_train_parquet,
        monkeypatch,
        transfer_learning,
        expected_call,
    ):
        """Ensure load_pretrained skips or performs model loading based on transfer_learning."""
        fake_model = SimpleNamespace()

        mock_from_pretrained = MagicMock(return_value=fake_model)
        monkeypatch.setattr(
            "tlmtc.finetune_pipeline.AutoModelForSequenceClassification.from_pretrained",
            mock_from_pretrained,
        )

        pipeline = pipeline_factory(
            train_path=dummy_train_parquet,
            wrap_peft=False,
            transfer_learning=transfer_learning,
        )

        assert pipeline.pretrained_model is None
        assert pipeline.num_labels is None

        pipeline.load_pretrained()

        if not expected_call:
            mock_from_pretrained.assert_not_called()
            assert pipeline.pretrained_model is None
            assert pipeline.num_labels is None
        else:
            mock_from_pretrained.assert_called_once_with(
                "dummy",
                num_labels=2,
                problem_type="multi_label_classification",
            )
            assert pipeline.pretrained_model is fake_model
            assert pipeline.num_labels == 2

    def test_raises_when_train_file_missing(self, pipeline_factory, tmp_path):
        """Ensure that load_pretrained raises a RuntimeError when the train file is missing."""
        missing_path = tmp_path / "train.parquet"
        assert not missing_path.exists()

        pipeline = pipeline_factory(train_path=missing_path)

        with pytest.raises(RuntimeError, match="Train data not found"):
            pipeline.load_pretrained()

    @pytest.mark.parametrize("wrap_peft", [True, False])
    def test_applies_peft_wrapping_conditionally(self, pipeline_factory, dummy_train_parquet, monkeypatch, wrap_peft):
        """Ensure that load_pretrained applies PEFT wrapping only when wrap_peft is True."""
        fake_model = SimpleNamespace(name="original")
        fake_peft_model = SimpleNamespace(name="wrapped")

        monkeypatch.setattr(
            "tlmtc.finetune_pipeline.AutoModelForSequenceClassification.from_pretrained",
            lambda *_, **__: fake_model,
        )

        wrap_mock = MagicMock(return_value=fake_peft_model)
        monkeypatch.setattr("tlmtc.finetune_pipeline.wrap_model_with_peft", wrap_mock)

        pipeline = pipeline_factory(dummy_train_parquet, wrap_peft=wrap_peft)

        pipeline.load_pretrained()

        if wrap_peft:
            wrap_mock.assert_called_once_with(
                model=fake_model,
                lora_r=pipeline.lora_r,
                lora_alpha=pipeline.lora_alpha,
                lora_dropout=pipeline.lora_dropout,
                lora_bias=pipeline.lora_bias,
            )
            assert pipeline.pretrained_model is fake_peft_model
        else:
            wrap_mock.assert_not_called()
            assert pipeline.pretrained_model is fake_model


class TestTuneHyperparameters:
    """Test suite for FinetunePipeline.tune_hyperparameters."""

    def test_returns_self_when_disabled(self, pipeline_factory, dummy_train_parquet):
        """Ensure tune_hyperparameters is a no-op and returns self when hyperparameter_tuning is False."""
        pipeline = pipeline_factory(dummy_train_parquet)
        assert pipeline.tokenized_dataset is None
        assert pipeline.num_labels is None

        pipeline.hyperparameter_tuning = False

        result = pipeline.tune_hyperparameters()

        assert result is pipeline
        assert pipeline.tokenized_dataset is None
        assert pipeline.num_labels is None

    def test_requires_tokenized_dataset(self, pipeline_factory, dummy_train_parquet):
        """Ensure tune_hyperparameters raises a RuntimeError when no tokenized dataset is set."""
        pipeline = pipeline_factory(dummy_train_parquet)

        pipeline.hyperparameter_tuning = True
        assert pipeline.tokenized_dataset is None

        with pytest.raises(RuntimeError, match="Tokenized dataset not found"):
            pipeline.tune_hyperparameters()

    def test_sets_num_labels_from_train_data(self, pipeline_with_tokenized):
        """Ensure num_labels is inferred from the multilabel train data when unset."""
        pipeline = pipeline_with_tokenized
        pipeline.num_labels = None

        pipeline.tune_hyperparameters()

        assert pipeline_with_tokenized.num_labels == 2

    @pytest.mark.parametrize("wrap_peft", [False, True])
    def test_merges_optuna_spaces_and_configures_hp_search(
        self,
        pipeline_with_tokenized,
        base_search_space,
        fake_trainer,
        wrap_peft,
    ):
        """Ensure user Optuna space overrides defaults and hyperparameter_search is configured correctly."""
        pipeline = pipeline_with_tokenized
        pipeline.wrap_peft = wrap_peft

        peft_space = {
            "lr_low": 1e-6,
            "lr_high": 1e-4,
            "batch_sizes": [4, 8],
            "wd_low": 0.0,
            "wd_high": 0.05,
            "schedulers": ["cosine"],
            "epoch_low": 2,
            "epoch_high": 4,
        }
        pipeline.optuna_space_default_peft = peft_space

        user_space = {
            "lr_high": 5e-4,
            "epoch_high": 10,
            "extra_param": "foobar",
        }
        pipeline.optuna_space_user = user_space

        def trainer_factory(*_args, **_kwargs):
            return fake_trainer

        result = pipeline.tune_hyperparameters(trainer=trainer_factory)

        assert result is pipeline

        assert fake_trainer.hp_search_calls, "hyperparameter_search was never called"
        assert len(fake_trainer.hp_search_calls) == 1

        hp_search_kwargs = fake_trainer.hp_search_calls[0]
        hp_space_fn = hp_search_kwargs["hp_space"]

        assert hp_space_fn.func is optuna_hp_space

        default_space = peft_space if wrap_peft else base_search_space
        actual_space = hp_space_fn.keywords["space"]
        expected_space = {**default_space, **user_space}
        assert actual_space == expected_space

        assert hp_search_kwargs["direction"] == "maximize"
        assert hp_search_kwargs["backend"] == "optuna"
        assert hp_search_kwargs["n_trials"] == pipeline.tuning_trials
        assert hp_search_kwargs["study_name"] == f"{pipeline.target_name.replace(' ', '_')}_optuna_study"

        expected_storage = f"sqlite:///{pipeline.output_logging_path.as_posix()}/optuna_trials.db"
        assert hp_search_kwargs["storage"] == expected_storage

        compute_objective = hp_search_kwargs["compute_objective"]
        assert callable(compute_objective)
        assert hp_search_kwargs["load_if_exists"] is True

    def test_instantiates_trainer_with_expected_arguments(
        self,
        pipeline_with_tokenized,
        fake_trainer,
    ):
        """Ensure Trainer is instantiated with the expected core arguments in tune_hyperparameters."""
        pipeline = pipeline_with_tokenized
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
        assert isinstance(model_instance, DummyModel)
        assert model_instance.num_labels == 2

    @pytest.mark.parametrize("scale_learning_rate", [False, True])
    def test_updates_pipeline_hyperparameters_from_best_run(
        self,
        pipeline_with_tokenized,
        fake_trainer,
        monkeypatch,
        scale_learning_rate,
    ):
        """Ensure pipeline hyperparameters reflect the best Optuna run and optionally use a scaled learning rate."""
        pipeline = pipeline_with_tokenized
        pipeline.scale_learning_rate = scale_learning_rate

        pipeline.learning_rate = 9.9
        pipeline.lr_scheduler = "cosine"
        pipeline.batch_size = 999
        pipeline.weight_decay = 0.5
        pipeline.epochs = 10

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
            assert pipeline.learning_rate == scaled_lr
            mock_get_scaled_lr.assert_called_once_with(
                learning_rate=1e-4,
                checkpoint=pipeline.checkpoint,
                proxy_checkpoint=pipeline.proxy_checkpoint,
                peft=pipeline.wrap_peft,
            )
        else:
            assert pipeline.learning_rate == 1e-4
            mock_get_scaled_lr.assert_not_called()

        assert pipeline.lr_scheduler == "linear"
        assert pipeline.batch_size == 8
        assert pipeline.weight_decay == 0.01
        assert pipeline.epochs == 2


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
        """Ensure fine_tune_pretrained is a no-op unless transfer_learning is True."""
        pipeline = pipeline_factory(train_path=dummy_train_parquet, transfer_learning=transfer_learning)
        pipeline.tokenized_dataset = tokenized_dataset
        pipeline.pretrained_model = DummyModel(num_labels=2)

        mock_get_class_weights = MagicMock(return_value=torch.ones(2))
        monkeypatch.setattr(
            "tlmtc.finetune_pipeline.get_class_weights",
            mock_get_class_weights,
            raising=True,
        )

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

    def test_requires_tokenized_dataset(
        self,
        pipeline_factory,
        dummy_train_parquet,
    ):
        """Ensure a RuntimeError is raised when tokenized_dataset is missing."""
        pipeline = pipeline_factory(train_path=dummy_train_parquet)
        pipeline.transfer_learning = True
        pipeline.tokenized_dataset = None
        pipeline.pretrained_model = DummyModel(num_labels=2)

        with pytest.raises(RuntimeError, match="Tokenized dataset not found"):
            pipeline.fine_tune_pretrained()

    def test_requires_pretrained_model(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
    ):
        """Ensure a RuntimeError is raised when pretrained_model is missing."""
        pipeline = pipeline_factory(train_path=dummy_train_parquet)
        pipeline.transfer_learning = True
        pipeline.tokenized_dataset = tokenized_dataset
        pipeline.pretrained_model = None

        with pytest.raises(RuntimeError, match="Pretrained model not loaded"):
            pipeline.fine_tune_pretrained()

    def test_instantiates_trainer_with_expected_arguments(
        self,
        pipeline_factory,
        dummy_train_parquet,
        tokenized_dataset,
        fake_trainer,
    ):
        """Test that Trainer receives the expected model, datasets, hyperparameters, and callbacks."""
        pipeline = pipeline_factory(train_path=dummy_train_parquet)
        pipeline.tokenized_dataset = tokenized_dataset
        pipeline.pretrained_model = DummyModel(num_labels=2)
        pipeline.hyperparameter_tuning = False  # train split only

        recorded: dict[str, Any] = {}

        def trainer_factory(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return fake_trainer

        pipeline.fine_tune_pretrained(trainer=trainer_factory)

        assert recorded, "Trainer was never instantiated by fine_tune_pretrained"
        assert recorded["args"] == ()
        kwargs = recorded["kwargs"]

        assert kwargs["model"] is pipeline.pretrained_model
        assert kwargs["train_dataset"] is pipeline.tokenized_dataset["train"]
        assert kwargs["eval_dataset"] is pipeline.tokenized_dataset["validation"]

        training_args = kwargs["args"]
        assert isinstance(training_args, TrainingArguments)
        assert training_args.learning_rate == pytest.approx(pipeline.learning_rate)
        assert training_args.num_train_epochs == pipeline.epochs
        assert training_args.per_device_train_batch_size == pipeline.batch_size
        assert training_args.weight_decay == pipeline.weight_decay
        assert training_args.lr_scheduler_type == pipeline.lr_scheduler
        assert training_args.metric_for_best_model == pipeline.best_model_metric
        assert training_args.use_cpu == pipeline.use_cpu

        assert callable(kwargs["compute_metrics"])
        assert isinstance(kwargs["class_weights"], torch.Tensor)

        callbacks = kwargs["callbacks"]
        assert isinstance(callbacks, list)
        assert callbacks, "Expected at least one callback for early stopping"

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
        pipeline_with_tokenized,
        monkeypatch,
        threshold_optimization,
        transfer_learning,
    ):
        """Ensure tune_thresholds is a no-op unless threshold optimization and transfer learning are both enabled."""
        pipeline = pipeline_with_tokenized
        pipeline.threshold_optimization = threshold_optimization
        pipeline.transfer_learning = transfer_learning

        pipeline.updated_trainer = MagicMock()
        find_mock = MagicMock()
        monkeypatch.setattr("tlmtc.finetune_pipeline.find_optimal_threshold", find_mock, raising=True)

        result = pipeline.tune_thresholds()

        assert result is pipeline
        pipeline.updated_trainer.predict.assert_not_called()
        find_mock.assert_not_called()

    @pytest.mark.parametrize(
        "tokenized_dataset, updated_trainer, expected_msg",
        [
            (None, MagicMock(), "Tokenized dataset not found"),
            ("__present__", None, "Trained model not found"),
        ],
    )
    def test_requires_prerequisites(
        self,
        pipeline_with_tokenized,
        tokenized_dataset,
        updated_trainer,
        expected_msg,
    ):
        """Ensure tune_thresholds raises when tokenized data or a trained model is missing."""
        pipeline = pipeline_with_tokenized
        pipeline.threshold_optimization = True
        pipeline.transfer_learning = True

        pipeline.tokenized_dataset = None if tokenized_dataset is None else pipeline.tokenized_dataset
        pipeline.updated_trainer = updated_trainer

        with pytest.raises(RuntimeError, match=expected_msg):
            pipeline.tune_thresholds()

    def test_predicts_on_validation_and_sets_tuned_threshold(
        self,
        pipeline_with_tokenized,
        monkeypatch,
    ):
        """Ensure tune_thresholds predicts on validation, calls the threshold finder, and stores the tuned threshold."""
        pipeline = pipeline_with_tokenized
        pipeline.threshold_optimization = True
        pipeline.transfer_learning = True

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
        assert kwargs["best_threshold_metric"] == pipeline.best_threshold_metric
        assert kwargs["threshold_type"] == pipeline.threshold_type
        assert "y_true" in kwargs
        assert "y_prob" in kwargs


class TestSavePretrained:
    """Test suite for FinetunePipeline.save_pretrained."""

    def test_noop_when_transfer_learning_disabled(
        self,
        pipeline_factory,
        dummy_train_parquet,
    ):
        """Ensure save_pretrained is a no-op when transfer_learning is False."""
        pipeline = pipeline_factory(dummy_train_parquet, transfer_learning=False)

        assert not pipeline.output_model_path.exists()

        result = pipeline.save_pretrained()

        assert result is pipeline
        assert not pipeline.output_model_path.exists()
        assert pipeline.updated_trainer is None

    def test_raises_if_trainer_missing(
        self,
        pipeline_factory,
        dummy_train_parquet,
    ):
        """Ensure save_pretrained raises an error when called before fine_tune_pretrained."""
        pipeline = pipeline_factory(dummy_train_parquet)
        assert pipeline.transfer_learning is True
        assert pipeline.updated_trainer is None

        with pytest.raises(RuntimeError, match="Instantiated Trainer after fine-tuning not found"):
            pipeline.save_pretrained()

    def test_delegates_to_model_with_output_path(
        self,
        pipeline_factory,
        dummy_train_parquet,
    ):
        """Test that save_pretrained calls model.save_pretrained with output_model_path."""
        pipeline = pipeline_factory(dummy_train_parquet)

        fake_model = MagicMock()
        pipeline.updated_trainer = SimpleNamespace(model=fake_model)

        assert not pipeline.output_model_path.exists()

        result = pipeline.save_pretrained()

        assert result is pipeline
        fake_model.save_pretrained.assert_called_once_with(pipeline.output_model_path)
