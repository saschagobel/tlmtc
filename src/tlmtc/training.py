"""Training components for Hugging Face multi-label text classification."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from sklearn.metrics import f1_score, roc_auc_score
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import ModelOutput  # type: ignore[attr-defined]

from tlmtc.data_contracts import LABEL_PREFIX
from tlmtc.settings import TrainingSettings

RESERVED_TRAINER_ARGS = frozenset(
    {
        "output_dir",
        "logging_dir",
        "eval_strategy",
        "evaluation_strategy",
        "save_strategy",
        "save_total_limit",
        "logging_strategy",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "num_train_epochs",
        "weight_decay",
        "learning_rate",
        "lr_scheduler_type",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
        "disable_tqdm",
        "use_cpu",
        "report_to",
    }
)

SEQUENCE_CLASSIFICATION_MODULES_TO_SAVE = (
    "classifier",
    "classification_head",
    "score",
    "pre_classifier",
    "pooler",
    "head",
)


class TrainingRuntimeState(BaseModel):
    """Mutable effective training hyperparameters for a concrete run.

    Attributes:
        batch_size: Effective training and evaluation batch size.
        train_epochs: Effective number of training epochs.
        weight_decay: Effective weight decay.
        learning_rate: Effective learning rate.
        lr_scheduler: Effective learning-rate scheduler name.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    batch_size: PositiveInt
    train_epochs: PositiveInt
    weight_decay: float = Field(..., ge=0.0)
    learning_rate: float = Field(..., gt=0.0)
    lr_scheduler: str

    @classmethod
    def from_settings(
        cls,
        training: TrainingSettings,
    ) -> Self:
        """Create runtime training state from resolved training settings.

        Args:
            training: Resolved immutable training settings.

        Returns:
            Mutable runtime training state.
        """
        return cls(
            batch_size=training.batch_size,
            train_epochs=training.train_epochs,
            weight_decay=training.weight_decay,
            learning_rate=training.learning_rate,
            lr_scheduler=training.lr_scheduler,
        )


def get_training_args(
    logging_path: str | Path,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    learning_rate: float,
    lr_scheduler: str,
    best_model_metric: str,
    use_cpu: bool,
    trainer_args: dict[str, Any],
) -> TrainingArguments:
    """Create Hugging Face TrainingArguments from effective training parameters.

    Args:
        logging_path: Directory for Trainer checkpoints and logs.
        batch_size: Training and evaluation batch size.
        epochs: Maximum number of training epochs.
        weight_decay: Weight decay applied during optimization.
        learning_rate: Initial optimizer learning rate.
        lr_scheduler: Learning-rate scheduler name.
        best_model_metric: Model-selection metric name as configured in training settings.
        use_cpu: Whether to force CPU execution.
        trainer_args: Additional Hugging Face ``TrainingArguments`` keyword arguments
            forwarded to the Trainer configuration. Arguments managed directly by
            tlmtc, such as batch size, learning rate, output directory, evaluation
            strategy, model-selection settings, and reporting behavior, are rejected.

    Returns:
        Configured TrainingArguments instance.

    Raises:
        ValueError: If ``trainer_args`` attempts to override tlmtc-managed
            ``TrainingArguments``.
    """
    overlapping_keys = RESERVED_TRAINER_ARGS.intersection(trainer_args)

    if overlapping_keys:
        overlapping = ", ".join(sorted(overlapping_keys))
        raise ValueError(
            f"trainer_args must not override tlmtc-managed TrainingArguments: {overlapping}. "
            "Pass these settings via dedicated tlmtc arguments instead."
        )

    init_kwargs: dict[str, Any] = {
        "output_dir": str(logging_path),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "logging_strategy": "epoch",
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler,
        "load_best_model_at_end": True,
        "metric_for_best_model": best_model_metric,
        "greater_is_better": True,
        "disable_tqdm": True,
        "use_cpu": use_cpu,
        "report_to": "none",
    }

    init_kwargs.update(trainer_args)

    return TrainingArguments(**init_kwargs)


def get_scaled_lr(
    learning_rate: float,
    checkpoint: str,
    proxy_checkpoint: str,
    peft: bool,
    trust_remote_code: bool,
) -> float:
    """Scale a proxy-tuned learning rate for a target checkpoint.

    Args:
        learning_rate: Proxy-tuned learning rate.
        checkpoint: Target checkpoint identifier.
        proxy_checkpoint: Proxy checkpoint identifier.
        peft: Whether the target model uses PEFT/LoRA.
        trust_remote_code: Whether Hugging Face config loading may execute custom remote code.

    Returns:
        Conservative target-checkpoint learning rate.
    """
    full_finetune_exponent = 0.5
    peft_exponent = 0.25
    min_scale = 0.5
    max_scale = 1.0

    target_hidden_size = AutoConfig.from_pretrained(
        checkpoint,
        trust_remote_code=trust_remote_code,
    ).hidden_size
    proxy_hidden_size = AutoConfig.from_pretrained(
        proxy_checkpoint,
        trust_remote_code=trust_remote_code,
    ).hidden_size

    hidden_size_ratio = proxy_hidden_size / target_hidden_size
    exponent = peft_exponent if peft else full_finetune_exponent

    scale = hidden_size_ratio**exponent
    bounded_scale = min(max(scale, min_scale), max_scale)

    return learning_rate * bounded_scale


def get_class_weights(
    train_data_path: str | Path,
) -> torch.Tensor:
    """Compute positive-class weights for BCEWithLogitsLoss.

    Args:
        train_data_path: Path to the prepared training split.

    Returns:
        Tensor with one `pos_weight` value per label.
    """
    train_data = pd.read_parquet(train_data_path)

    label_cols = [col for col in train_data.columns if col.startswith(LABEL_PREFIX)]
    labels_array = train_data[label_cols].to_numpy()

    positive_counts = labels_array.sum(axis=0)
    negative_counts = labels_array.shape[0] - positive_counts

    pos_weights = negative_counts / positive_counts
    return torch.tensor(pos_weights, dtype=torch.float)


def get_num_labels(
    train_data_path: str | Path,
) -> int:
    """Count label columns in a prepared multi-label training split.

    Args:
        train_data_path: Path to the prepared training split.

    Returns:
        Number of `label_*` columns.
    """
    train_data = pd.read_parquet(train_data_path)
    return sum(1 for col in train_data.columns if col.startswith(LABEL_PREFIX))


def multi_label_metrics(
    predictions: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Compute Trainer metrics from multi-label logits and labels.

    Args:
        predictions: Model logits with shape `(n_samples, n_labels)`.
        labels: Ground-truth binary label matrix with the same shape as `predictions`.

    Returns:
        F1 and ROC-AUC micro and macro metrics for Trainer evaluation.
    """
    probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    y_true = np.array(labels)
    y_pred = (probs >= 0.5).astype(int)
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    roc_auc_micro = roc_auc_score(y_true, probs, average="micro")
    roc_auc_macro = roc_auc_score(y_true, probs, average="macro")
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "roc_auc_micro": roc_auc_micro,
        "roc_auc_macro": roc_auc_macro,
    }
    return metrics


def compute_metrics(
    p: EvalPrediction,
) -> dict[str, Any]:
    """Compute multi-label metrics from a Hugging Face EvalPrediction.

    Args:
        p: Evaluation predictions produced by Hugging Face Trainer.

    Returns:
        Multi-label evaluation metrics.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
    result = multi_label_metrics(predictions=preds, labels=labels)
    return result


def infer_modules_to_save(
    model: PreTrainedModel,
) -> list[str]:
    """Infer sequence-classification head modules to keep trainable with PEFT.

    Args:
        model: Hugging Face sequence-classification model.

    Returns:
        Top-level module names that should be saved alongside LoRA adapters.
    """
    top_level_modules = dict(model.named_children())
    return [name for name in SEQUENCE_CLASSIFICATION_MODULES_TO_SAVE if name in top_level_modules]


def wrap_model_with_peft(
    model: PreTrainedModel,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: Literal["none", "all", "lora_only"],
) -> PreTrainedModel | PeftModel | PeftMixedModel:
    """Wrap a sequence-classification model with PEFT/LoRA adapters.

    Args:
        model: Pretrained sequence-classification model.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        lora_bias: LoRA bias handling mode.

    Returns:
        Model wrapped for parameter-efficient fine-tuning.
    """
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules="all-linear",
        modules_to_save=infer_modules_to_save(model) or None,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=True,
        init_lora_weights=True,
        bias=lora_bias,
    )
    return get_peft_model(model, peft_config)


def make_model_init(
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: Literal["none", "all", "lora_only"],
    trust_remote_code: bool,
) -> Callable[[object | None], PreTrainedModel | PeftModel | PeftMixedModel]:
    """Create a Trainer-compatible model factory.

    Args:
        checkpoint: Pretrained model checkpoint identifier.
        num_labels: Number of labels in the multi-label classification task.
        wrap_peft: Whether to wrap the model with PEFT/LoRA adapters.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        lora_bias: LoRA bias handling mode.
        trust_remote_code: Whether Hugging Face model loading may execute custom remote code.

    Returns:
        Factory accepted by Hugging Face Trainer as `model_init`.
    """

    def model_init(
        _: object | None = None,
    ) -> PreTrainedModel | PeftModel | PeftMixedModel:
        """Initialize a fresh model.

        The optional positional argument is accepted for Trainer hyperparameter search compatibility.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=trust_remote_code,
        )

        if wrap_peft:
            return wrap_model_with_peft(
                model=model,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
            )

        return model

    return model_init


class WeightedTrainer(Trainer):
    """Trainer variant using class-weighted BCE loss for multi-label classification.

    Args:
        *args: Positional arguments forwarded to Hugging Face Trainer.
        class_weights: Optional positive-class weights passed as `pos_weight` to BCEWithLogitsLoss.
        **kwargs: Keyword arguments forwarded to Hugging Face Trainer.

    Attributes:
        loss_fct: BCEWithLogitsLoss instance configured with optional positive-class weights.
    """

    def __init__(
        self,
        *args: Any,
        class_weights: Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainer with optional positive-class weights.

        Args:
            *args: Positional arguments forwarded to Hugging Face Trainer.
            class_weights: Optional positive-class weights for BCEWithLogitsLoss.
            **kwargs: Keyword arguments forwarded to Hugging Face Trainer.
        """
        super().__init__(*args, **kwargs)

        self.model_accepts_loss_kwargs = False

        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)

        self.loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=class_weights,
            reduction="mean",
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, ModelOutput]:
        """Compute class-weighted BCE loss for a multi-label batch.

        Args:
            model: Model being trained.
            inputs: Input batch containing `labels` and model input tensors.
            return_outputs: Whether to return model outputs together with the loss.
            num_items_in_batch: Batch-size metadata accepted for Trainer compatibility.

        Returns:
            Loss tensor, optionally paired with model outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        num_labels = getattr(model, "num_labels", None)
        if num_labels is None:
            num_labels = getattr(model.module, "num_labels")

        loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))

        return (loss, outputs) if return_outputs else loss
