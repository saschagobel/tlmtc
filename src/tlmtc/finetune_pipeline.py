"""
Transfer Learning for Multi-Label Text Classification.

Hyperparameter tuning and model fine-tuning.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from torch import Tensor
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback, PreTrainedModel, Trainer
from transformers.modeling_outputs import ModelOutput  # type: ignore[attr-defined]

from tlmtc.utils import (
    _compute_metrics,
    _find_optimal_threshold,
    _get_class_weights,
    _get_scaled_lr,
    _get_training_args,
    _make_compute_objective,
    _make_model_init,
    _optuna_hp_space,
    _wrap_peft,
)


class WeightedTrainer(Trainer):
    """
    Custom Hugging Face Trainer class-balanced loss weighting for multi-label classification.

    Parameters
    ----------
    *args : Any
        Positional arguments passed to the parent 'Trainer'
    class_weights : Optional[torch.FloatTensor], default=None
        A 1D tensor of positive weights for each class. Used as 'pos_weight' in 'torch.nn.BCEWithLogitsLoss' to
        up- or down-weight positive examples depending on class imbalance
    **kwargs : Any
        Keyword arguments passed to the parent 'Trainer'

    Attributes
    ----------
    loss_fct : torch.nn.BCEWithLogitsLoss
        Loss function configured with optional class weights
    """

    def __init__(
        self,
        *args: Any,
        class_weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)

        self.loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=class_weights,
            reduction="mean",
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, ModelOutput]:
        """
        Compute the weighted BCE loss for multi-label classification with multi-GPU support.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained
        inputs : dict
            Input batch, including 'labels' and other model-specific input tensors
        return_outputs : bool, optional
            If True, return a tuple (loss, outputs), default is False
        num_items_in_batch : int, optional
            Number of items in the current batch, default is None

        Returns
        -------
        loss : torch.Tensor
            The computed loss value
        outputs : ModelOutput, optional
            Returned only if 'return_outputs' is True, contains model outputs
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        num_labels = getattr(model, "num_labels", None)
        if num_labels is None:
            num_labels = getattr(model.module, "num_labels")

        loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))

        return (loss, outputs) if return_outputs else loss


class FinetunePipeline:
    """
    Set up and perform hyperparameter tuning and/or transfer learning on a pretrained model.

    Attributes
    ----------
    tokenized_dataset : DatasetDict
        Tokenized dataset ready for PyTorch
    train_data_path : str or Path
        Path to the training split
    val_data_path : str or Path
        Path where the validation split will be saved
    output_logging_path : str or Path
        Path where intermediate checkpoints and logs will be saved
    output_model_path: str or Path
        Path where fine-tuned models will be saved
    target_name : str
        Short name or description of the classification target
    proxy_checkpoint: str
        Name of the pretrained model checkpoint on the Hugging Face Hub, used for hyperparameter tuning
    checkpoint : str
        Name of the pretrained model checkpoint on the Hugging Face Hub, used for transfer learning
    transfer_learning : bool
        Flag whether transfer learning should be performed
    hyperparameter_tuning: bool
        Flag whether hyperparameter tuning should be performed
    threshold_optimization: bool
        Flag whether threshold optimization should be performed
    threshold_type: str
        Type of threshold to compute, 'global' or 'label'
    scale_learning_rate: bool
        Flag whether learning rate should be scaled
    wrap_peft: bool
        Flag whether to wrap model in parameter-efficient fine-tuning
    optuna_space_default_base : Dict[str, Any]
        Default Optuna search space used when *not* using PEFT.
    optuna_space_default_peft : Dict[str, Any]
        Default Optuna search space used when PEFT is enabled.
    optuna_space_user : dict or None
        User-supplied overrides for the hyperparameter search space.
        Ignored hyperparameters fall back to the corresponding default space.
    tuning_trials: int
        Number of trials to perform during hyperparameter tuning
    batch_size : int
        Batch size for training and evaluation
    weight_decay: float
        Strength of weight decay regularization applied to model parameters
    learning_rate: float
        Initial learning rate for optimizer
    lr_scheduler: str
        Type of learning rate scheduler to use
    epochs : int
        Maximum number of training epochs
    best_model_metric : str
        Metric to monitor for selecting the best-performing model checkpoint
    best_threshold_metric: str
        Metric to monitor for selecting the best-performing global threshold
    early_stopping_patience : int
        Number of evaluation steps without improvement before triggering early stopping
    lora_r : int
        Rank of the LoRA matrices. Controls adapter capacity
    lora_alpha : int
        Scaling factor for the LoRA updates
    lora_dropout : float
        Dropout probability for LoRA layers
    lora_bias : str
        Whether to train bias terms, 'none', 'all', or 'lora_only'
    use_cpu : bool
        Flag whether to force training on CPU instead of GPU
    pretrained_model: transformers.PreTrainedModel
        Pretrained model ready for fine-tuning
    updated_trainer: transformers.Trainer
        The instantiated Trainer after fine-tuning
    num_labels : int
         Number of labels in the multi-label classification task
    tuned_threshold : np.ndarray
        Optimal global or label-specific thresholds for multi-label classification

    Methods
    -------
    load_pretrained():
        Load a pretrained Hugging Face model for multi-label classification
    tune_hyperparameters():
        Run automated hyperparameter optimization on the pretrained Hugging Face proxy model using Optuna
    fine_tune_pretrained():
        Fine-tune a pretrained Hugging Face model for multi-label classification
    save_pretrained():
        Save a fine-tuned Hugging Face model for multi-label classification
    """

    def __init__(
        self,
        tokenized_dataset: DatasetDict,
        train_data_path: Union[str, Path],
        val_data_path: Union[str, Path],
        output_logging_path: Path,
        output_model_path: Union[str, Path],
        target_name: str,
        proxy_checkpoint: str,
        checkpoint: str,
        transfer_learning: bool,
        hyperparameter_tuning: bool,
        threshold_optimization: bool,
        threshold_type: str,
        scale_learning_rate: bool,
        wrap_peft: bool,
        optuna_space_default_base: Dict[str, Any],
        optuna_space_default_peft: Dict[str, Any],
        tuning_trials: int,
        batch_size: int,
        weight_decay: float,
        learning_rate: float,
        lr_scheduler: str,
        epochs: int,
        best_model_metric: str,
        best_threshold_metric: str,
        early_stopping_patience: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_bias: Literal["none", "all", "lora_only"],
        use_cpu: bool,
        optuna_space_user: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize configuration.

        Parameters
        ----------
        tokenized_dataset : DatasetDict
            Tokenized dataset ready for PyTorch
        train_data_path : str or Path
            Path to the training split
        val_data_path : str or Path
            Path where the validation split will be saved
        output_logging_path : str or Path
            Path where intermediate checkpoints and logs will be saved
        output_model_path: str or Path
            Path where fine-tuned models will be saved
        target_name : str
            Short name or description of the classification target
        proxy_checkpoint: str
            Name of the pretrained model checkpoint on the Hugging Face Hub, used for hyperparameter tuning
        checkpoint : str
            Name of the pretrained model checkpoint on the Hugging Face Hub, used for transfer learning
        transfer_learning : bool
            Flag whether transfer learning should be performed
        hyperparameter_tuning: bool
            Flag whether hyperparameter tuning should be performed
        threshold_optimization: bool
            Flag whether threshold optimization should be performed
        threshold_type: str
            Type of threshold to compute, 'global' or 'label'
        scale_learning_rate: bool
            Flag whether learning rate should be scaled
        wrap_peft: bool
            Flag whether to wrap model in parameter-efficient fine-tuning
        optuna_space_default_base : Dict[str, Any]
            Default Optuna search space used when *not* using PEFT.
        optuna_space_default_peft : Dict[str, Any]
            Default Optuna search space used when PEFT is enabled.
        optuna_space_user : dict or None
            User-supplied overrides for the hyperparameter search space.
            Ignored hyperparameters fall back to the corresponding default space.
        tuning_trials: int
            Number of trials to perform during hyperparameter tuning
        batch_size : int
            Batch size for training and evaluation
        weight_decay: float
            Strength of weight decay regularization applied to model parameters
        learning_rate: float
            Initial learning rate for optimizer
        lr_scheduler: str
            Type of learning rate scheduler to use
        epochs : int
            Maximum number of training epochs
        best_model_metric : str
            Metric to monitor for selecting the best-performing model checkpoint
        best_threshold_metric: str
            Metric to monitor for selecting the best-performing global threshold
        early_stopping_patience : int
            Number of evaluation steps without improvement before triggering early stopping
        lora_r : int
            Rank of the LoRA matrices. Controls adapter capacity
        lora_alpha : int
            Scaling factor for the LoRA updates
        lora_dropout : float
            Dropout probability for LoRA layers
        lora_bias : str
            Whether to train bias terms, 'none', 'all', or 'lora_only'
        use_cpu : bool
            Flag whether to force training on CPU instead of GPU
        """
        self.tokenized_dataset = tokenized_dataset
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.output_logging_path = output_logging_path
        self.output_model_path = output_model_path
        self.target_name = target_name
        self.proxy_checkpoint = proxy_checkpoint
        self.checkpoint = checkpoint
        self.transfer_learning = transfer_learning
        self.hyperparameter_tuning = hyperparameter_tuning
        self.threshold_optimization = threshold_optimization
        self.threshold_type = threshold_type
        self.scale_learning_rate = scale_learning_rate
        self.wrap_peft = wrap_peft
        self.optuna_space_default_base = optuna_space_default_base
        self.optuna_space_default_peft = optuna_space_default_peft
        self.tuning_trials = tuning_trials
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.best_model_metric = best_model_metric
        self.best_threshold_metric = best_threshold_metric
        self.early_stopping_patience = early_stopping_patience
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.use_cpu = use_cpu
        self.optuna_space_user = optuna_space_user
        self.pretrained_model: Optional[PreTrainedModel] = None
        self.updated_trainer: Optional[Trainer] = None
        self.num_labels: Optional[int] = None
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
        if not self.transfer_learning:
            return self

        if not os.path.exists(self.train_data_path):
            raise RuntimeError("Train data not found. Run DataPipeline class first.")

        self.num_labels = sum(1 for col in pd.read_parquet(self.train_data_path).columns if col.startswith("label_"))
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, num_labels=self.num_labels, problem_type="multi_label_classification"
        )

        if self.wrap_peft:
            self.pretrained_model = _wrap_peft(
                model=self.pretrained_model,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_bias=self.lora_bias,
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
        if not self.hyperparameter_tuning:
            return self
        if self.tokenized_dataset is None:
            raise RuntimeError("Tokenized dataset not found. Run DataPipeline class first.")
        if self.num_labels is None:
            self.num_labels = sum(
                1 for col in pd.read_parquet(self.train_data_path).columns if col.startswith("label_")
            )

        default_space = self.optuna_space_default_peft if self.wrap_peft else self.optuna_space_default_base

        if self.optuna_space_user:
            resolved_space = {**default_space, **self.optuna_space_user}
        else:
            resolved_space = default_space

        hp_space_fn = partial(
            _optuna_hp_space,
            space=resolved_space,
        )

        self.output_logging_path.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as output_dir:
            model_init = _make_model_init(
                checkpoint=self.proxy_checkpoint,
                num_labels=self.num_labels,
                wrap_peft=self.wrap_peft,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_bias=self.lora_bias,
            )
            compute_objective = _make_compute_objective(best_model_metric=self.best_model_metric)
            training_args = _get_training_args(
                logging_path=output_dir,
                batch_size=self.batch_size,
                epochs=self.epochs,
                weight_decay=self.weight_decay,
                learning_rate=self.learning_rate,
                lr_scheduler=self.lr_scheduler,
                best_model_metric=self.best_model_metric,
                use_cpu=self.use_cpu,
            )
            trainer_instance = trainer(
                model=None,
                args=training_args,
                train_dataset=self.tokenized_dataset["train"],
                eval_dataset=self.tokenized_dataset["validation"],
                compute_metrics=_compute_metrics,
                class_weights=_get_class_weights(train_data_path=self.train_data_path),
                model_init=model_init,
            )
            best_run = trainer_instance.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                hp_space=hp_space_fn,
                n_trials=self.tuning_trials,
                study_name=f"{self.target_name.replace(' ', '_')}_optuna_study",
                storage=f"sqlite:///{self.output_logging_path.as_posix()}/optuna_trials.db",
                compute_objective=compute_objective,
                load_if_exists=True,
            )
        if self.scale_learning_rate:
            self.learning_rate = _get_scaled_lr(
                learning_rate=best_run.hyperparameters["learning_rate"],
                checkpoint=self.checkpoint,
                proxy_checkpoint=self.proxy_checkpoint,
                peft=self.wrap_peft,
            )
        else:
            self.learning_rate = best_run.hyperparameters["learning_rate"]
        self.lr_scheduler = best_run.hyperparameters["lr_scheduler_type"]
        self.batch_size = best_run.hyperparameters["per_device_train_batch_size"]
        self.weight_decay = best_run.hyperparameters["weight_decay"]
        self.epochs = best_run.hyperparameters["num_train_epochs"]
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
        if not self.transfer_learning:
            return self
        if self.tokenized_dataset is None:
            raise RuntimeError("Tokenized dataset not found. Run DataPipeline class first.")
        if self.pretrained_model is None:
            raise RuntimeError("Pretrained model not loaded. Run load_pretrained() first.")

        training_args = _get_training_args(
            logging_path=self.output_logging_path,
            batch_size=self.batch_size,
            epochs=self.epochs,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            lr_scheduler=self.lr_scheduler,
            best_model_metric=self.best_model_metric,
            use_cpu=self.use_cpu,
        )
        trainer_instance = trainer(
            model=self.pretrained_model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)],
            class_weights=_get_class_weights(train_data_path=self.train_data_path),
        )
        trainer_instance.train()
        self.updated_trainer = trainer_instance
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
        if not self.transfer_learning:
            return self

        if self.updated_trainer is None:
            raise RuntimeError("Instantiated Trainer after fine-tuning not found. Run fine_tune_pretrained() first.")

        self.updated_trainer.model.save_pretrained(self.output_model_path)  # type: ignore[union-attr]
        return self
