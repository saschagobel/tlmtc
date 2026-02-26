"""
Transfer Learning for Multi-Label Text Classification.

Hyperparameter tuning and model fine-tuning.
"""

from __future__ import annotations

from functools import partial
from tempfile import TemporaryDirectory
from typing import Type, Union

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback, PreTrainedModel, Trainer

from tlmtc.evaluation import find_optimal_threshold
from tlmtc.hpo import make_compute_objective, make_model_init, optuna_hp_space
from tlmtc.paths import RunPaths
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
    WeightedTrainer,
    compute_metrics,
    get_class_weights,
    get_scaled_lr,
    get_training_args,
    wrap_model_with_peft,
)


class FinetunePipeline:
    """Fine-tune a pretrained model and optionally run HPO and threshold optimization.

    Attributes:
        tokenized_dataset: Tokenized Hugging Face dataset ready for PyTorch
        paths: Resolved filesystem locations for persisted train and validation splits, logs, and model outputs.
        model: Model configuration (proxy-checkpoint and checkpoint).
        workflow: High-level workflow toggles (HPO, learning rate scaling, transfer learning, threshold optimization).
        peft: PEFT/LoRA configuration.
        training: Training hyperparameters (updated after HPO).
        hpo: Hyperparameter optimization settings, including the resolved Optuna space.
        threshold: Threshold optimization settings.
        hardware: Hardware settings (forcing CPU execution).
        pretrained_model: Loaded model instance ready for fine-tuning.
        updated_trainer: The instantiated Trainer after fine-tuning.
        num_labels: Number of labels in the multi-label classification task.
        tuned_threshold: Tuned global or label-specific thresholds for multi-label classification
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
            tokenized_dataset : Tokenized dataset ready for PyTorch with "train"/"validation"/"test" splits.
            paths: Run-specific filesystem layout for reading data splits and writing artifacts.
            model: Model settings (checkpoints).
            workflow: Workflow flags controlling which stages to run.
            hpo: Hyperparameter tuning configuration, including the resolved Optuna search space.
            training: Training hyperparameters (updated by HPO).
            peft: LoRA/PEFT settings.
            threshold: Threshold optimization settings.
            hardware: Hardware settings (forcing CPU execution).
        """
        self.tokenized_dataset = tokenized_dataset
        self.paths = paths
        self.model = model
        self.workflow = workflow
        self.hpo = hpo
        self.training = training
        self.peft = peft
        self.threshold = threshold
        self.hardware = hardware
        self.pretrained_model: PreTrainedModel | None = None
        self.updated_trainer: Trainer | None = None
        self.num_labels: int | None = None
        self.tuned_threshold: Union[float, np.ndarray] = 0.5

    def load_pretrained(
        self,
    ) -> FinetunePipeline:
        """
        Load a pretrained Hugging Face model for multi-label classification and optionally wrap with peft.

        Returns
        -------
        FinetunePipeline
        """
        if not self.workflow.transfer_learning:
            return self

        if not self.paths.train_data_path.exists():
            raise RuntimeError("Train data not found. Run DataPipeline class first.")

        self.num_labels = sum(
            1 for col in pd.read_parquet(self.paths.train_data_path).columns if col.startswith("label_")
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
        trainer: Type[Trainer] = WeightedTrainer,
    ) -> FinetunePipeline:
        """
        Run automated hyperparameter optimization on the pretrained Hugging Face proxy model using Optuna.

        Parameters
        ----------
        trainer: Type[transformers.Trainer], default=WeightedTrainer
            Custom Hugging Face Trainer for handling class imbalances in multi-label classification

        Returns
        -------
        FinetunePipeline
        """
        if not self.workflow.hyperparameter_tuning:
            return self
        if self.tokenized_dataset is None:
            raise RuntimeError("Tokenized dataset not found. Run DataPipeline class first.")
        if self.num_labels is None:
            self.num_labels = sum(
                1 for col in pd.read_parquet(self.paths.train_data_path).columns if col.startswith("label_")
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
                batch_size=self.training.batch_size,
                epochs=self.training.train_epochs,
                weight_decay=self.training.weight_decay,
                learning_rate=self.training.learning_rate,
                lr_scheduler=self.training.lr_scheduler,
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
            best_run = trainer_instance.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                hp_space=hp_space_fn,
                n_trials=self.hpo.tuning_trials,
                study_name=f"{self.model.target_name.replace(' ', '_')}_optuna_study",
                storage=f"sqlite:///{self.paths.logs_dir.as_posix()}/optuna_trials.db",
                compute_objective=compute_objective,
                load_if_exists=True,
            )
        if self.workflow.scale_learning_rate:
            self.training.learning_rate = get_scaled_lr(
                learning_rate=best_run.hyperparameters["learning_rate"],
                checkpoint=self.model.checkpoint,
                proxy_checkpoint=self.model.proxy_checkpoint,
                peft=self.workflow.wrap_peft,
            )
        else:
            self.training.learning_rate = best_run.hyperparameters["learning_rate"]
        self.training.lr_scheduler = best_run.hyperparameters["lr_scheduler_type"]
        self.training.batch_size = best_run.hyperparameters["per_device_train_batch_size"]
        self.training.weight_decay = best_run.hyperparameters["weight_decay"]
        self.training.train_epochs = best_run.hyperparameters["num_train_epochs"]
        return self

    def fine_tune_pretrained(
        self,
        trainer: Type[Trainer] = WeightedTrainer,
    ) -> FinetunePipeline:
        """
        Fine-tune a pretrained Hugging Face model for multi-label classification.

        Parameters
        ----------
        trainer: Type[transformers.Trainer], default=WeightedTrainer
            Custom Hugging Face Trainer for handling class imbalances in multi-label classification

        Returns
        -------
        FinetunePipeline
        """
        if not self.workflow.transfer_learning:
            return self
        if self.tokenized_dataset is None:
            raise RuntimeError("Tokenized dataset not found. Run DataPipeline class first.")
        if self.pretrained_model is None:
            raise RuntimeError("Pretrained model not loaded. Run load_pretrained() first.")

        training_args = get_training_args(
            logging_path=self.paths.logs_dir,
            batch_size=self.training.batch_size,
            epochs=self.training.train_epochs,
            weight_decay=self.training.weight_decay,
            learning_rate=self.training.learning_rate,
            lr_scheduler=self.training.lr_scheduler,
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

    def tune_thresholds(self) -> FinetunePipeline:
        """
        Tune decision threshold(s) on the validation split using the trained model.

        This step is post-training calibration: it does not update model weights.
        The tuned threshold(s) are stored in `self.tuned_threshold`.

        Returns
        -------
        FinetunePipeline
        """
        if not self.workflow.threshold_optimization or not self.workflow.transfer_learning:
            return self

        if self.tokenized_dataset is None:
            raise RuntimeError("Tokenized dataset not found. Run DataPipeline class first.")
        if self.updated_trainer is None:
            raise RuntimeError("Trained model not found. Run fine_tune_pretrained() first.")

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
    ) -> FinetunePipeline:
        """
        Save a fine-tuned Hugging Face model for multi-label classification.

        Returns
        -------
        FinetunePipeline
        """
        if not self.workflow.transfer_learning:
            return self

        if self.updated_trainer is None:
            raise RuntimeError("Instantiated Trainer after fine-tuning not found. Run fine_tune_pretrained() first.")

        self.updated_trainer.model.save_pretrained(self.paths.model_dir)  # type: ignore[operator,union-attr]
        return self
