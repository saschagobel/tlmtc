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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EvalPrediction,
    PreTrainedModel,
    TrainingArguments,
)


def _df_preprocess(
    df_path: Union[str, Path],
) -> Tuple[pd.DataFrame, List[str], np.ndarray, np.ndarray]:
    """
    Import, preprocess and extract column labels from raw train/test data.

    Parameters
    ----------
    df_path : str or Path
        Path to the raw CSV data file, with required columns "text", "label_*"

    Returns
    -------
    df : pd.DataFrame
        Preprocessed data
    label_cols : list of str
        Label column names
    X : np.ndarray
        Texts
    y : np.ndarray
        Label matrix
    """
    df = pd.read_csv(df_path).dropna()
    if df.empty:
        raise ValueError("After dropping missing values, no valid samples remain.")

    if "text" not in df.columns:
        raise ValueError("Input data must contain a 'text' column.")

    label_cols = [col for col in df.columns if col.startswith("label_")]
    if len(label_cols) < 2:
        raise ValueError("Expected at least two 'label_*' columns for multi-label classification.")
    for col in label_cols:
        if not df[col].map(lambda x: isinstance(x, (int, float))).all():
            raise TypeError(f"Column '{col}' must contain only integer values.")
    allowed_values = {0, 1, 0.0, 1.0}
    if not set(df[label_cols].stack().unique()).issubset(allowed_values):
        raise ValueError("Label columns must contain only binary values {0, 1, 0.0, 1.0}.")

    X = df["text"].values
    y = df[label_cols].values
    return df, label_cols, X, y


def _df_split(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split preprocessed data into stratified train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data
    X : np.ndarray
        Texts
    y : np.ndarray
        Label matrix
    test_size : float
        Proportion of data set to be used for testing
    random_seed : int
        Random seed

    Returns
    -------
    train_data : pd.DataFrame
        Train set
    test_data : pd.DataFrame
        Test set
    """
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    for train_idx, test_idx in msss.split(X, y):
        train_data = df.iloc[train_idx].reset_index(drop=True)
        test_data = df.iloc[test_idx].reset_index(drop=True)
    return train_data, test_data


def _df_save(
    df: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """
    Save a DataFrame to disk as parquet files.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    path : str or Path
        Path where the DataFrame will be saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _get_training_args(
    logging_path: Union[str, Path],
    batch_size: int,
    epochs: int,
    weight_decay: float,
    learning_rate: float,
    lr_scheduler: str,
    best_model_metric: str,
    use_cpu: bool,
) -> TrainingArguments:
    """
    Initialize a TrainingArguments object with set hyperparameters.

    Parameters
    ----------
    logging_path : str or Path
        Path where intermediate checkpoints and logs will be saved
    batch_size : int
        Batch size for training and evaluation
    epochs : int
        Maximum number of training epochs
    weight_decay: float
        Strength of weight decay regularization applied to model parameters
    learning_rate: float
        Initial learning rate for optimizer
    lr_scheduler: str
        Type of learning rate scheduler to use
    best_model_metric : str
        Metric to monitor for selecting the best-performing model checkpoint
    use_cpu : bool
        Flag whether to force training on CPU instead of GPU

    Returns
    -------
    transformers.TrainingArguments
        Configured TrainingArguments instance for the Trainer class
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
    space: Dict[str, Any],
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


def _get_scaled_lr(
    learning_rate: float,
    checkpoint: str,
    proxy_checkpoint: str,
    peft: bool,
) -> float:
    """
    Scale the learning rate by hidden size.

    Parameters
    ----------
    learning_rate: float
        Learning rate for optimizer
    checkpoint : str
        Name of the pretrained model checkpoint on the Hugging Face Hub
    proxy_checkpoint : str
        Name of the proxy pretrained model checkpoint on the Hugging Face Hub
    peft : bool
        Flag whether model uses parameter-efficient fine-tuning

    Returns
    -------
    float
        Scaled learning rate
    """
    checkpoint_hidden_size = AutoConfig.from_pretrained(checkpoint).hidden_size
    proxy_checkpoint_hidden_size = AutoConfig.from_pretrained(proxy_checkpoint).hidden_size
    if peft:
        return learning_rate * (checkpoint_hidden_size / proxy_checkpoint_hidden_size) ** 0.5
    else:
        return learning_rate * (proxy_checkpoint_hidden_size / checkpoint_hidden_size)


def _get_class_weights(
    train_data_path: Union[str, Path],
    val_data_path: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """
    Compute label-specific weights for positive classes.

    Parameters
    ----------
    train_data_path : str
           Path to train split
    val_data_path : str
           Path to validation split

    Returns
    -------
    torch.Tensor
        Tensor with class weights for each label
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


def _multi_label_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute evaluation metrics for multi-label classification.

    Parameters
    ----------
    predictions : array-like or torch.Tensor
        Model outputs (logits) for each sample and label
    labels : array-like
        Binary labels

    Returns
    -------
    dict
        Dictionary containing the following metrics:
        - 'f1_micro': Micro-averaged F1 score
        - 'f1_macro': Macro-averaged F1 score
        - 'roc_auc_micro': Micro-averaged ROC-AUC score
        - 'roc_auc_macro': Macro-averaged ROC-AUC score
        - 'accuracy': Standard accuracy computed over all labels
    """
    probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    y_true = np.array(labels)
    threshold = _find_optimal_threshold(
        y_true=y_true, y_prob=probs, best_threshold_metric="f1_macro", threshold_type="global"
    )
    y_pred = (probs >= threshold).astype(int)
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
) -> Dict[str, Any]:
    """
    Wrap a Hugging Face `EvalPrediction` object to compute multi-label metrics.

    Parameters
    ----------
    p : EvalPrediction
        Evaluation prediction object from Hugging Face Trainer, with attributes:
        - `predictions`: Model output logits
        - `label_ids`: Ground-truth labels

    Returns
    -------
    dict
        Dictionary of evaluation metrics as returned by 'multi_label_metrics'
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = _multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result
