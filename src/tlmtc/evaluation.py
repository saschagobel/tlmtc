"""Evaluation metrics and diagnostics for multi-label text classification."""

from typing import Any

import numpy as np
import pandas as pd
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


def get_global_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute aggregate metrics for multi-label classification.

    Args:
        y_true: Ground-truth binary label matrix with shape `(n_samples, n_labels)`.
        y_pred: Predicted binary label matrix with the same shape as `y_true`.
        y_prob: Predicted probability matrix with the same shape as `y_true`.

    Returns:
        Aggregate evaluation metrics.
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


def get_label_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-label metrics for multi-label classification.

    Args:
        y_true: Ground-truth binary label matrix with shape `(n_samples, n_labels)`.
        y_pred: Predicted binary label matrix with the same shape as `y_true`.
        y_prob: Predicted probability matrix with the same shape as `y_true`.
        label_names: Label names in the same order as the label matrix columns.

    Returns:
        Per-label evaluation metrics keyed by label name.
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


def round_metric_dict(
    metrics: dict[str, float],
    ndigits: int = 2,
) -> dict[str, float]:
    """Round metric values for presentation.

    Args:
        metrics: Metric values keyed by metric name.
        ndigits: Number of decimal places.

    Returns:
        Metric values rounded to `ndigits` decimal places.
    """
    return {k: round(v, ndigits) for k, v in metrics.items()}


def get_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
) -> dict[str, dict[int | str, Any]]:
    """Compute per-label, micro, and macro ROC curve data.

    Args:
        y_true: Ground-truth binary label matrix with shape `(n_samples, n_labels)`.
        y_prob: Predicted probability matrix with the same shape as `y_true`.
        label_names: Label names aligned with matrix columns.

    Returns:
        False-positive rates, true-positive rates, and ROC-AUC values keyed by label index,
        `micro`, and `macro`.
    """
    num_labels = len(label_names)
    fpr: dict[int | str, np.ndarray] = dict()
    tpr: dict[int | str, np.ndarray] = dict()
    roc_auc: dict[int | str, float] = dict()
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


def get_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
) -> dict[str, dict[int | str, Any]]:
    """Compute per-label, micro, and macro precision-recall curve data.

    Args:
        y_true: Ground-truth binary label matrix with shape `(n_samples, n_labels)`.
        y_prob: Predicted probability matrix with the same shape as `y_true`.
        label_names: Label names aligned with matrix columns.

    Returns:
        Precision values, recall values, and average precision values keyed by label index,
        `micro`, and `macro`.
    """
    num_labels = len(label_names)
    precision: dict[int | str, np.ndarray] = dict()
    recall: dict[int | str, np.ndarray] = dict()
    avg_precision: dict[int | str, float] = dict()
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


def get_co_occurrence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute observed and predicted label co-occurrence matrices.

    Args:
        y_true: Ground-truth binary label matrix with shape `(n_samples, n_labels)`.
        y_pred: Predicted binary label matrix with the same shape as `y_true`.

    Returns:
        Absolute and normalized co-occurrence matrices for true and predicted labels.
    """
    results: dict[str, np.ndarray] = {}

    for name, labels in (("true", y_true), ("pred", y_pred)):
        c_abs = np.dot(labels.T, labels)
        diag = np.diag(c_abs)

        denom = np.sqrt(np.outer(diag, diag))
        c_rel = np.divide(
            c_abs,
            denom,
            out=np.zeros_like(c_abs, dtype=float),
            where=denom != 0,
        )
        np.fill_diagonal(c_rel, diag > 0)

        results[f"co_{name}_abs"] = c_abs
        results[f"co_{name}_rel"] = c_rel

    return results


def get_losses(
    log_history: list[dict[str, Any]],
) -> pd.DataFrame:
    """Extract aligned training and evaluation losses from Trainer logs.

    Args:
        log_history: Hugging Face Trainer log history.

    Returns:
        DataFrame with `epoch`, `train_loss`, and `eval_loss` columns.
    """
    train_losses = pd.DataFrame([{"epoch": d["epoch"], "train_loss": d["loss"]} for d in log_history if "loss" in d])
    eval_losses = pd.DataFrame(
        [{"epoch": d["epoch"], "eval_loss": d["eval_loss"]} for d in log_history if "eval_loss" in d]
    )
    return pd.merge(train_losses, eval_losses, on="epoch", how="inner")


def get_best_epoch(
    log_history: list[dict[str, Any]],
    best_model_metric: str,
) -> int:
    """Extract the number of the best epoch from Trainer.state.log_history.

    Args:
        log_history: The Trainer's state.log_history attribute.
        best_model_metric: Model-selection metric name as configured in training settings.

    Returns:
        Best epoch number.
    """
    eval_logs = [entry for entry in log_history if "eval_" + best_model_metric in entry]
    return int(max(eval_logs, key=lambda x: x["eval_" + best_model_metric])["epoch"])


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    best_threshold_metric: str,
    threshold_type: str,
) -> np.ndarray:
    """Search global or per-label decision thresholds for multi-label predictions.

    Args:
        y_true: Ground-truth binary label matrix of shape (n_samples, n_labels).
        y_prob: Predicted probabilities of the same shape as y_true.
        best_threshold_metric: Metric to monitor for selecting the best-performing global threshold.
        threshold_type: Type of threshold to compute, 'global' or 'label'.

    Returns:
        One-element array for a global threshold, or one threshold per label for label-specific mode.

    Raises:
        ValueError: If `best_threshold_metric` or `threshold_type` is unsupported.
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
