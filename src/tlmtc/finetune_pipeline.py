"""Fine-tuning pipeline for Hugging Face multi-label text classification."""

from functools import partial
from tempfile import TemporaryDirectory
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback, PreTrainedModel, Trainer

from tlmtc.data_contracts import LABEL_PREFIX
from tlmtc.evaluation import find_optimal_threshold
from tlmtc.hpo import make_compute_objective, make_model_init, optuna_hp_space
from tlmtc.paths import RunPaths
from tlmtc.runtime_output import emit_progress
from tlmtc.settings import (
    HardwareSettings,
    HpoSettings,
    ModelSettings,
    PeftSettings,
    ThresholdSettings,
    TrainingSettings,
    WorkflowSettings,
)
from tlmtc.training import (
    TrainingRuntimeState,
    WeightedTrainer,
    compute_metrics,
    get_class_weights,
    get_scaled_lr,
    get_training_args,
    wrap_model_with_peft,
)


class TrainerFactory(Protocol):
    """Factory protocol for Trainer-compatible instances."""

    def __call__(self, **kwargs: Any) -> Trainer:
        """Create a configured Trainer instance."""
        ...


class FinetunePipeline:
    """Stateful fine-tuning pipeline for multi-label text classification.

    Attributes:
        tokenized_dataset: Tokenized Hugging Face dataset with train, validation, and test splits.
        paths: Run-specific filesystem layout for data splits, logs, model outputs, and artifacts.
        model: Model configuration, including checkpoint identifiers and classification target name.
        workflow: Workflow configuration controlling HPO, learning-rate scaling, fine-tuning, and thresholds.
        peft: PEFT/LoRA configuration.
        training: Resolved immutable training settings.
        runtime_training: Mutable effective training state used during HPO and fine-tuning.
        hpo: Hyperparameter optimization settings, including the resolved Optuna search space.
        threshold: Threshold optimization settings.
        hardware: Hardware settings controlling device selection.
        pretrained_model: Loaded model instance before Trainer fine-tuning.
        updated_trainer: Trainer instance after fine-tuning.
        num_labels: Number of labels in the multi-label classification task.
        tuned_threshold: Global or per-label decision thresholds for multi-label prediction.
    """

    def __init__(
        self,
        tokenized_dataset: DatasetDict,
        paths: RunPaths,
        model: ModelSettings,
        workflow: WorkflowSettings,
        peft: PeftSettings,
        training: TrainingSettings,
        hpo: HpoSettings,
        threshold: ThresholdSettings,
        hardware: HardwareSettings,
    ) -> None:
        """Initialize the fine-tuning pipeline.

        Args:
            tokenized_dataset: Tokenized Hugging Face dataset with train, validation, and test splits.
            paths: Run-specific filesystem layout for data splits, logs, model outputs, and artifacts.
            model: Model configuration, including checkpoint identifiers and classification target name.
            workflow: Workflow configuration controlling HPO, learning-rate scaling, fine-tuning, and thresholds.
            peft: PEFT/LoRA configuration.
            training: Resolved immutable training settings.
            hpo: Hyperparameter optimization settings, including the resolved Optuna search space.
            threshold: Threshold optimization settings.
            hardware: Hardware settings controlling device selection.
        """
        self.tokenized_dataset = tokenized_dataset
        self.paths = paths
        self.model = model
        self.workflow = workflow
        self.hpo = hpo
        self.training = training
        self.runtime_training: TrainingRuntimeState = TrainingRuntimeState.from_settings(training)
        self.peft = peft
        self.threshold = threshold
        self.hardware = hardware
        self.pretrained_model: PreTrainedModel | None = None
        self.updated_trainer: Trainer | None = None
        self.num_labels: int | None = None
        self.tuned_threshold: np.ndarray = np.array([0.5], dtype=float)

    def load_pretrained(
        self,
    ) -> Self:
        """Load the target sequence-classification model and optionally apply PEFT/LoRA.

        Returns:
            Updated pipeline instance.

        Raises:
            RuntimeError: If the prepared training split is missing.
        """
        if not self.workflow.transfer_learning:
            return self

        if not self.paths.train_data_path.exists():
            raise RuntimeError("Train data not found. Run DataPipeline class first.")

        emit_progress("Loading pretrained transformer model")

        self.num_labels = sum(
            1 for col in pd.read_parquet(self.paths.train_data_path).columns if col.startswith(LABEL_PREFIX)
        )
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            self.model.checkpoint, num_labels=self.num_labels, problem_type="multi_label_classification"
        )

        if self.workflow.wrap_peft:
            self.pretrained_model = wrap_model_with_peft(  # type: ignore[assignment]
                model=self.pretrained_model,
                lora_r=self.peft.lora_r,
                lora_alpha=self.peft.lora_alpha,
                lora_dropout=self.peft.lora_dropout,
                lora_bias=self.peft.lora_bias,
            )
        return self

    def tune_hyperparameters(
        self,
        trainer: TrainerFactory = WeightedTrainer,
    ) -> Self:
        """Run Optuna hyperparameter tuning on the proxy checkpoint.

        Args:
            trainer: Trainer-compatible factory used for hyperparameter search.

        Returns:
            Updated pipeline instance.

        Raises:
            RuntimeError: If tokenized data is missing or Optuna returns multiple best runs.
        """
        if not self.workflow.hyperparameter_tuning:
            return self
        if self.tokenized_dataset is None:
            raise RuntimeError("Tokenized dataset not found. Run DataPipeline class first.")
        if self.num_labels is None:
            self.num_labels = sum(
                1 for col in pd.read_parquet(self.paths.train_data_path).columns if col.startswith(LABEL_PREFIX)
            )

        hp_space_fn = partial(
            optuna_hp_space,
            space=self.hpo.optuna_space,
        )

        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as output_dir:
            model_init = make_model_init(
                checkpoint=self.model.proxy_checkpoint,
                num_labels=self.num_labels,
                wrap_peft=self.workflow.wrap_peft,
                lora_r=self.peft.lora_r,
                lora_alpha=self.peft.lora_alpha,
                lora_dropout=self.peft.lora_dropout,
                lora_bias=self.peft.lora_bias,
            )
            compute_objective = make_compute_objective(best_model_metric=self.training.best_model_metric)
            training_args = get_training_args(
                logging_path=output_dir,
                batch_size=self.runtime_training.batch_size,
                epochs=self.runtime_training.train_epochs,
                weight_decay=self.runtime_training.weight_decay,
                learning_rate=self.runtime_training.learning_rate,
                lr_scheduler=self.runtime_training.lr_scheduler,
                best_model_metric=self.training.best_model_metric,
                use_cpu=self.hardware.use_cpu,
            )
            trainer_instance = trainer(
                model=None,
                args=training_args,
                train_dataset=self.tokenized_dataset["train"],
                eval_dataset=self.tokenized_dataset["validation"],
                compute_metrics=compute_metrics,
                class_weights=get_class_weights(train_data_path=self.paths.train_data_path),
                model_init=model_init,
            )

            emit_progress("Loading pretrained proxy transformer model")

            emit_progress("Running hyperparameter optimization")

            best_run = trainer_instance.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                hp_space=hp_space_fn,
                n_trials=self.hpo.tuning_trials,
                study_name=f"{self.model.target_name.replace(' ', '_')}_optuna_study",
                storage=f"sqlite:///{self.paths.optuna_trials_path.as_posix()}",
                compute_objective=compute_objective,
                load_if_exists=True,
            )
        if isinstance(best_run, list):
            raise RuntimeError("Expected a single best run from single-objective HPO, but received a list.")

        if self.workflow.scale_learning_rate:
            self.runtime_training.learning_rate = get_scaled_lr(
                learning_rate=best_run.hyperparameters["learning_rate"],
                checkpoint=self.model.checkpoint,
                proxy_checkpoint=self.model.proxy_checkpoint,
                peft=self.workflow.wrap_peft,
            )
        else:
            self.runtime_training.learning_rate = best_run.hyperparameters["learning_rate"]
        self.runtime_training.lr_scheduler = best_run.hyperparameters["lr_scheduler_type"]
        self.runtime_training.batch_size = best_run.hyperparameters["per_device_train_batch_size"]
        self.runtime_training.weight_decay = best_run.hyperparameters["weight_decay"]
        self.runtime_training.train_epochs = best_run.hyperparameters["num_train_epochs"]
        return self

    def fine_tune_pretrained(
        self,
        trainer: TrainerFactory = WeightedTrainer,
    ) -> Self:
        """Fine-tune the loaded target model on the training split.

        Args:
            trainer: Trainer-compatible factory used for fine-tuning.

        Returns:
            Updated pipeline instance.

        Raises:
            RuntimeError: If tokenized data or the loaded pretrained model is missing.
        """
        if not self.workflow.transfer_learning:
            return self
        if self.tokenized_dataset is None:
            raise RuntimeError("Tokenized dataset not found. Run DataPipeline class first.")
        if self.pretrained_model is None:
            raise RuntimeError("Pretrained model not loaded. Run load_pretrained() first.")

        emit_progress("Fine-tuning model")

        training_args = get_training_args(
            logging_path=self.paths.logs_dir,
            batch_size=self.runtime_training.batch_size,
            epochs=self.runtime_training.train_epochs,
            weight_decay=self.runtime_training.weight_decay,
            learning_rate=self.runtime_training.learning_rate,
            lr_scheduler=self.runtime_training.lr_scheduler,
            best_model_metric=self.training.best_model_metric,
            use_cpu=self.hardware.use_cpu,
        )
        trainer_instance = trainer(
            model=self.pretrained_model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.training.early_stopping_patience)],
            class_weights=get_class_weights(train_data_path=self.paths.train_data_path),
        )
        trainer_instance.train()
        self.updated_trainer = trainer_instance
        return self

    def tune_thresholds(self) -> Self:
        """Tune decision thresholds on validation-set predictions.

        Returns:
            Updated pipeline instance.

        Raises:
            RuntimeError: If tokenized data or the fine-tuned trainer is missing.
        """
        if not self.workflow.threshold_optimization or not self.workflow.transfer_learning:
            return self

        if self.tokenized_dataset is None:
            raise RuntimeError("Tokenized dataset not found. Run DataPipeline class first.")
        if self.updated_trainer is None:
            raise RuntimeError("Trained model not found. Run fine_tune_pretrained() first.")

        emit_progress("Optimizing decision thresholds")

        preds = self.updated_trainer.predict(self.tokenized_dataset["validation"])
        logits = preds.predictions
        probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
        true_labels = np.array(self.tokenized_dataset["validation"]["labels"])

        self.tuned_threshold = find_optimal_threshold(
            y_true=true_labels,
            y_prob=probabilities,
            best_threshold_metric=self.threshold.best_threshold_metric,
            threshold_type=self.threshold.threshold_type,
        )
        return self

    def save_pretrained(
        self,
    ) -> Self:
        """Save the fine-tuned model artifacts.

        Returns:
            Updated pipeline instance.

        Raises:
            RuntimeError: If the fine-tuned trainer is missing.
        """
        if not self.workflow.transfer_learning:
            return self

        if self.updated_trainer is None:
            raise RuntimeError("Instantiated Trainer after fine-tuning not found. Run fine_tune_pretrained() first.")

        emit_progress("Saving model artifacts")

        self.updated_trainer.model.save_pretrained(self.paths.model_dir)  # type: ignore[operator,union-attr]
        return self
