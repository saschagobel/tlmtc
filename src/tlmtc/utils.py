"""
Transfer Learning for Multi-Label Text Classification.

Helper functions
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EvalPrediction,
    PreTrainedModel,
    TrainingArguments,
)

from tlmtc.types import OptunaSpace





def _wrap_peft(
    model: PreTrainedModel,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: Literal["none", "all", "lora_only"],
) -> PreTrainedModel:
    """
    Wrap parameter-efficient fine-tuning (LoRA) around a pre-trained model.

    Parameters
    ----------
    model: transformers.PreTrainedModel
        Pretrained model ready for fine-tuning
    lora_r : int
        Rank of the LoRA matrices. Controls adapter capacity
    lora_alpha : int
        Scaling factor for the LoRA updates
    lora_dropout : float
        Dropout probability for LoRA layers
    lora_bias : str
        Whether to train bias terms, 'none', 'all', or 'lora_only'

    Returns
    -------
    model: transformers.PreTrainedModel
        The model pretrained model wrapped with LoRA adapters, ready for fine-tuning.
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


def _make_model_init(
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: Literal["none", "all", "lora_only"],
) -> Callable[[Optional[optuna.trial.Trial]], PreTrainedModel]:
    """
    Create a model initialization function for hyperparameter search.

    Parameters
    ----------
    checkpoint : str
        Name of the pretrained model checkpoint on the Hugging Face Hub
    num_labels : int
        Number of labels in the multi-label classification task
    wrap_peft: bool
        Flag whether to wrap model in parameter-efficient fine-tuning
    lora_r : int
        Rank of the LoRA matrices. Controls adapter capacity
    lora_alpha : int
        Scaling factor for the LoRA updates
    lora_dropout : float
        Dropout probability for LoRA layers
    lora_bias : str
        Whether to train bias terms, 'none', 'all', or 'lora_only'

    Returns
    -------
    model_init : callable
        A function that initializes and returns a model instance during each Optuna trial
    """

    def model_init(trial: Optional[optuna.trial.Trial] = None) -> PreTrainedModel:
        """
        Initialize a new model instance for the current trial.

        Parameters
        ----------
        trial : optuna.trial.Trial, optional
            Current Optuna trial object

        Returns
        -------
        model : transformers.PreTrainedModel
            Pretrained model ready for fine-tuning
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels, problem_type="multi_label_classification"
        )
        if wrap_peft:
            model = _wrap_peft(
                model=model,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
            )
        return model

    return model_init


def _make_compute_objective(
    best_model_metric: str,
) -> Callable[[Dict[str, Any]], float]:
    """
    Create an objective function for Optuna hyperparameter search.

    Parameters
    ----------
    best_model_metric : str
        Metric to monitor for selecting the best-performing model checkpoint

    Returns
    -------
    compute_objective : callable
        A function that extracts and returns the target metric from the Trainer evaluation output
    """

    def compute_objective(metrics: Dict[str, Any]) -> float:
        """
        Extract the objective value for the current Optuna trial.

        Parameters
        ----------
        metrics : dict
            Dictionary of evaluation results returned by the Trainer

        Returns
        -------
        float
            Value of the target metric to be optimized
        """
        return metrics["eval_" + best_model_metric]

    return compute_objective


def _optuna_hp_space(
    trial: optuna.trial.Trial,
    space: OptunaSpace,
) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna tuning.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Current Optuna trial object
    space: dict
        A fully resolved hyperparameter search space dictionary.

    Returns
    -------
    dict
        Dictionary specifying the sampled hyperparameters and their values for the current trial
    """
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            space["lr_low"],
            space["lr_high"],
            log=True,
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size",
            space["batch_sizes"],
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            space["wd_low"],
            space["wd_high"],
        ),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type",
            space["schedulers"],
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", space["epoch_low"], space["epoch_high"]),
    }





def _find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    best_threshold_metric: str,
    threshold_type: str,
) -> np.ndarray:
    """
    Compute the optimal global threshold for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary label matrix of shape (n_samples, n_labels)
    y_prob : np.ndarray
        Predicted probabilities of the same shape as y_true
    best_threshold_metric: str
        Metric to monitor for selecting the best-performing global threshold
    threshold_type: str
        Type of threshold to compute, 'global' or 'label'

    Returns
    -------
    best_threshold or best_thresholds: float
        Optimal global threshold or label-specific threshold
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    num_labels = y_true.shape[1]

    if threshold_type == "global":
        best_threshold, best_score = 0.5, float("-inf")
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            if best_threshold_metric == "f1_micro":
                score = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
            elif best_threshold_metric == "f1_macro":
                score = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
            else:
                raise ValueError("Unsupported metric. Use 'f1_micro' or 'f1_macro' as best_threshold_metric")
            if score > best_score:
                best_threshold, best_score = threshold, score
        return np.array([best_threshold], dtype=float)
    elif threshold_type == "label":
        best_thresholds = np.zeros(num_labels, dtype=float)
        for i in range(num_labels):
            best_threshold, best_score = 0.5, float("-inf")
            for threshold in thresholds:
                y_pred_i = (y_prob[:, i] >= threshold).astype(int)
                score = f1_score(y_true=y_true[:, i], y_pred=y_pred_i, zero_division=0)
                if score > best_score:
                    best_threshold, best_score = threshold, score
            best_thresholds[i] = best_threshold
        return best_thresholds
    else:
        raise ValueError("threshold_type must be 'global' or 'label'.")




def _get_global_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute global evaluation metrics for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary label matrix of shape (n_samples, n_labels)
    y_pred : np.ndarray
        Predicted binary label matrix of the same shape as y_true
    y_prob : np.ndarray
        Predicted probabilities of the same shape as y_true

    Returns
    -------
    metrics : dict
        Dictionary containing global metrics
    """
    metrics = {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc_micro": float(roc_auc_score(y_true, y_prob, average="micro")),
        "roc_auc_macro": float(roc_auc_score(y_true, y_prob, average="macro")),
        "pr_auc_micro": float(average_precision_score(y_true, y_prob, average="micro")),
        "pr_auc_macro": float(average_precision_score(y_true, y_prob, average="macro")),
        "true_cardinality": float(y_true.sum(axis=1).mean()),
        "pred_cardinality": float(y_pred.sum(axis=1).mean()),
    }
    return metrics


def _get_label_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute label-specific evaluation metrics for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary label matrix of shape (n_samples, n_labels)
    y_pred : np.ndarray
        Predicted binary label matrix of the same shape as y_true
    y_prob : np.ndarray
        Predicted probabilities of the same shape as y_true
    label_names : list of str
        Names of the labels

    Returns
    -------
    dict
        Dictionary containing label-specific metrics
    """
    label_f1 = f1_score(y_true, y_pred, average=None)
    label_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    label_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    label_roc = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])]
    label_pr = average_precision_score(y_true, y_prob, average=None)
    metrics = {
        name: {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "true_support": float(y_true[:, i].mean()),
            "pred_support": float(y_pred[:, i].mean()),
        }
        for i, (name, f1, precision, recall, roc, pr) in enumerate(
            zip(label_names, label_f1, label_precision, label_recall, label_roc, label_pr)
        )
    }
    return metrics


def _round_metric_dict(
    metrics: Dict[str, float],
    ndigits: int = 2,
) -> Dict[str, float]:
    """
    Round a dictionary of metric values for presentation.

    Parameters
    ----------
    metrics: dict of str, float
        Dictionary containing label-specific metrics
     ndigits : int, default=2
        Number of decimal places to round to.

    Returns
    -------
    dict of str, float
        Input dictionary with values rounded to `ndigits` decimal places.
    """
    return {k: round(v, ndigits) for k, v in metrics.items()}


def _get_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
) -> Dict[str, Dict[int | str, Any]]:
    """
    Compute ROC curve metrics for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary label matrix of shape (n_samples, n_labels)
    y_prob : np.ndarray
        Predicted probabilities of the same shape as y_true
    label_names : list of str
        Names of the labels

    Returns
    -------
    dict
        Dictionary containing false positive rate, true positive rate, and AUC values
    """
    num_labels = len(label_names)
    fpr: Dict[Union[int, str], np.ndarray] = dict()
    tpr: Dict[Union[int, str], np.ndarray] = dict()
    roc_auc: Dict[Union[int, str], float] = dict()
    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_labels)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_labels):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_labels
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}


def _get_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
) -> Dict[str, Dict[int | str, Any]]:
    """
    Compute PR curve metrics for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary label matrix of shape (n_samples, n_labels)
    y_prob : np.ndarray
        Predicted probabilities of the same shape as y_true
    label_names : list of str
        Names of the labels

    Returns
    -------
    dict
        Dictionary containing precision, recall, and average precision values
    """
    num_labels = len(label_names)
    precision: Dict[int | str, np.ndarray] = dict()
    recall: Dict[int | str, np.ndarray] = dict()
    avg_precision: Dict[int | str, float] = dict()
    for i in range(num_labels):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        avg_precision[i] = average_precision_score(y_true[:, i], y_prob[:, i])
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    avg_precision["micro"] = average_precision_score(y_true, y_prob, average="micro")
    all_recall = np.unique(np.concatenate([recall[i] for i in range(num_labels)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(num_labels):
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
    mean_precision /= num_labels
    recall["macro"] = all_recall
    precision["macro"] = mean_precision
    avg_precision["macro"] = average_precision_score(y_true, y_prob, average="macro")
    return {"precision": precision, "recall": recall, "avg_precision": avg_precision}


def _get_co_occurrence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute label co-occurrence for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary label matrix of shape (n_samples, n_labels)
    y_pred : np.ndarray
        Predicted binary label matrix of the same shape as y_true

    Returns
    -------
    dict
        Dictionary containing absolute and relative label co-occurrence
    """
    co_true_abs = np.dot(y_true.T, y_true)
    diag_true = np.diag(co_true_abs)
    co_true_rel = co_true_abs / np.sqrt(np.outer(diag_true, diag_true))
    np.fill_diagonal(co_true_rel, 1.0)
    co_pred_abs = np.dot(y_pred.T, y_pred)
    diag_pred = np.diag(co_pred_abs)
    co_pred_rel = co_pred_abs / np.sqrt(np.outer(diag_pred, diag_pred))
    np.fill_diagonal(co_pred_rel, 1.0)
    return {
        "co_true_abs": co_true_abs,
        "co_pred_abs": co_pred_abs,
        "co_true_rel": co_true_rel,
        "co_pred_rel": co_pred_rel,
    }


def _get_losses(
    log_history: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Extract per-epoch training and evaluation losses from Trainer.state.log_history.

    Parameters
    ----------
    log_history : list of dict
        The Trainer's state.log_history attribute

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'epoch', 'train_loss', 'eval_loss'
    """
    train_losses = pd.DataFrame([{"epoch": d["epoch"], "train_loss": d["loss"]} for d in log_history if "loss" in d])
    eval_losses = pd.DataFrame(
        [{"epoch": d["epoch"], "eval_loss": d["eval_loss"]} for d in log_history if "eval_loss" in d]
    )
    return pd.merge(train_losses, eval_losses, on="epoch", how="inner")


def _get_best_epoch(
    log_history: List[Dict[str, Any]],
    best_model_metric: str,
) -> int:
    """
    Extract the number of the best epoch from Trainer.state.log_history.

    Parameters
    ----------
    log_history : list of dict
        The Trainer's state.log_history attribute
    best_model_metric : str
        Metric to monitor for selecting the best-performing model checkpoint

    Returns
    -------
    int
        Number of the best epoch
    """
    eval_logs = [entry for entry in log_history if "eval_" + best_model_metric in entry]
    return int(max(eval_logs, key=lambda x: x["eval_" + best_model_metric])["epoch"])
