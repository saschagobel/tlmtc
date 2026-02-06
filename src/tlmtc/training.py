"""Internal helpers for model training and training-time metrics.

Defines utilities for training, class weighting, and evaluation metrics used during training.
"""


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from transformers import AutoConfig, EvalPrediction, PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput  # type: ignore[attr-defined]

from tlmtc.types import LoraBias


def _get_training_args(
    logging_path: str | Path,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    learning_rate: float,
    lr_scheduler: str,
    best_model_metric: str,
    use_cpu: bool,
) -> TrainingArguments:
    """Initialize a TrainingArguments object with set hyperparameters.

    Args:
        logging_path: Path where intermediate checkpoints and logs will be saved.
        batch_size: Batch size for training and evaluation.
        epochs: Maximum number of training epochs.
        weight_decay: Strength of weight decay regularization applied to model parameters.
        learning_rate: Initial learning rate for optimizer.
        lr_scheduler: Type of learning rate scheduler to use.
        best_model_metric: Metric to monitor for selecting the best-performing model checkpoint.
        use_cpu: Flag whether to force training on CPU instead of GPU.

    Returns:
        Configured transformers.TrainingArguments instance for the Trainer class.
    """
    return TrainingArguments(
        output_dir=str(logging_path),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,
        load_best_model_at_end=True,
        metric_for_best_model=best_model_metric,
        greater_is_better=True,
        disable_tqdm=True,
        use_cpu=use_cpu,
        report_to="none",
    )


def _get_scaled_lr(
    learning_rate: float,
    checkpoint: str,
    proxy_checkpoint: str,
    peft: bool,
) -> float:
    """Scale the learning rate by hidden size.

    Args:
        learning_rate: Learning rate for optimizer.
        checkpoint: Name of the pretrained model checkpoint on the Hugging Face Hub.
        proxy_checkpoint: Name of the proxy pretrained model checkpoint on the Hugging Face Hub.
        peft: Flag whether model uses parameter-efficient fine-tuning.

    Returns:
        Scaled learning rate.
    """
    checkpoint_hidden_size = AutoConfig.from_pretrained(checkpoint).hidden_size
    proxy_checkpoint_hidden_size = AutoConfig.from_pretrained(proxy_checkpoint).hidden_size
    if peft:
        return learning_rate * (checkpoint_hidden_size / proxy_checkpoint_hidden_size) ** 0.5
    else:
        return learning_rate * (proxy_checkpoint_hidden_size / checkpoint_hidden_size)


def _get_class_weights(
    train_data_path: str | Path,
    val_data_path: str | Path | None = None,
) -> torch.Tensor:
    """Compute label-specific weights for positive classes.

    Args:
        train_data_path: Path to train split.
        val_data_path: Path to validation split.

    Returns:
        torch.Tensor with class weights for each label.
    """
    train_data = pd.read_parquet(train_data_path)

    if val_data_path is not None:
        val_data = pd.read_parquet(val_data_path)
        train_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)

    label_cols = [col for col in train_data.columns if col.startswith("label_")]
    labels_array = train_data[label_cols].values
    num_labels = labels_array.shape[1]
    class_weights = [
        compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels_array[:, i])[1]
        for i in range(num_labels)
    ]
    return torch.tensor(class_weights, dtype=torch.float)


def _multi_label_metrics(
    predictions: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Compute evaluation metrics for multi-label classification.

    Args:
        predictions: Model outputs (logits) for each sample and label.
        labels: Binary labels.

    Returns:
        Dictionary containing the following metrics:
            - 'f1_micro': Micro-averaged F1 score.
            - 'f1_macro': Macro-averaged F1 score.
            - 'roc_auc_micro': Micro-averaged ROC-AUC score.
            - 'roc_auc_macro': Macro-averaged ROC-AUC score.
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


def _compute_metrics(
    p: EvalPrediction,
) -> dict[str, Any]:
    """Wrap a Hugging Face `EvalPrediction` object to compute multi-label metrics.

    Args:
        p: Evaluation prediction object from Hugging Face Trainer, with attributes:
            - `predictions`: Model output logits.
            - `label_ids`: Ground-truth labels.

    Returns:
        Dictionary of evaluation metrics as returned by `_multi_label_metrics`.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = _multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def _wrap_peft(
    model: PreTrainedModel,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: LoraBias,
) -> PreTrainedModel:
    """Wrap parameter-efficient fine-tuning (LoRA) around a pre-trained model.

    Args:
        model: Pretrained model ready for fine-tuning.
        lora_r: Rank of the LoRA matrices. Controls adapter capacity.
        lora_alpha: Scaling factor for the LoRA updates.
        lora_dropout: Dropout probability for LoRA layers.
        lora_bias: Whether to train bias terms, 'none', 'all', or 'lora_only'.

    Returns:
        model: The model pretrained model wrapped with LoRA adapters, ready for fine-tuning.
    """
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules="all-linear",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=True,
        init_lora_weights=True,
        bias=lora_bias,
    )
    model = get_peft_model(model, peft_config)
    return model


class WeightedTrainer(Trainer):
    """Custom Hugging Face Trainer class-balanced loss weighting for multi-label classification.

    Args:
        *args: Positional arguments passed to the parent 'Trainer'.
        class_weights: A 1D tensor of positive weights for each class. Used as 'pos_weight' in
            'torch.nn.BCEWithLogitsLoss' to up- or down-weight positive examples depending on class imbalance.
        **kwargs: Keyword arguments passed to the parent 'Trainer'.

    Attributes:
        loss_fct: torch.nn.BCEWithLogitsLoss function configured with optional class weights.
    """

    def __init__(
        self,
        *args: Any,
        class_weights: Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainer with optional class weights."""
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
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, ModelOutput]:
        """Compute the weighted BCE loss for multi-label classification with multi-GPU support.

        Args:
            model: Model being trained.
            inputs: Input batch, including 'labels' and other model-specific input tensors.
            return_outputs: If True, return a tuple (loss, outputs), default is False.
            num_items_in_batch: Number of items in the current batch, default is None.

        Returns:
            loss: The computed loss value.
            outputs: Returned only if 'return_outputs' is True, contains model outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        num_labels = getattr(model, "num_labels", None)
        if num_labels is None:
            num_labels = getattr(model.module, "num_labels")

        loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))

        return (loss, outputs) if return_outputs else loss
