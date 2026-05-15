"""Prediction operations for Hugging Face multi-label text classification."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from tlmtc.data_contracts import DataContractError


def load_prediction_model(
    model_dir: Path,
    *,
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
) -> PreTrainedModel | PeftModel:
    """Load a trained prediction model from full-model or PEFT artifacts.

    Args:
        model_dir: Directory containing saved model or adapter artifacts.
        checkpoint: Base checkpoint used when loading PEFT adapters.
        num_labels: Number of labels in the trained classification head.
        wrap_peft: Whether `model_dir` contains PEFT adapter artifacts.

    Returns:
        Loaded sequence-classification model.
    """
    if wrap_peft:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=False,
        )
        return PeftModel.from_pretrained(
            base_model,
            model_dir,
            low_cpu_mem_usage=True,
        )

    return AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        trust_remote_code=False,
    )


def predict_probabilities(
    model: PreTrainedModel | PeftModel,
    dataset: Dataset,
    batch_size: int,
    use_cpu: bool,
) -> np.ndarray:
    """Predict multi-label probabilities for a tokenized prediction dataset.

    Args:
        model: Trained sequence-classification model.
        dataset: Tokenized Hugging Face prediction dataset.
        batch_size: Prediction batch size.
        use_cpu: Whether to force CPU inference.

    Returns:
        Probability matrix with shape `(n_rows, n_labels)`.
    """
    accelerator = Accelerator(cpu=use_cpu)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    prepared_model = accelerator.prepare_model(
        model,
        evaluation_mode=True,
    )
    prepared_dataloader = accelerator.prepare_data_loader(dataloader)

    prepared_model.eval()

    probabilities: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in prepared_dataloader:
            logits = prepared_model(**batch).logits
            probabilities.append(torch.sigmoid(logits))

    probabilities_tensor = accelerator.gather_for_metrics(torch.cat(probabilities))
    return probabilities_tensor.float().cpu().numpy()


def apply_thresholds(
    probabilities: np.ndarray,
    thresholds: list[float],
) -> np.ndarray:
    """Apply global or label-specific decision thresholds.

    Args:
        probabilities: Probability matrix with shape `(n_rows, n_labels)`.
        thresholds: One global threshold or one threshold per label.

    Returns:
        Binary prediction matrix with shape `(n_rows, n_labels)`.
    """
    return (probabilities >= np.asarray(thresholds, dtype=float)).astype(int)


def make_prediction_frame(
    input_df: pd.DataFrame,
    values: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """Create a prediction output dataframe.

    Args:
        input_df: Validated prediction input dataframe.
        values: Probability or binary prediction matrix with shape `(n_rows, n_labels)`.
        label_names: Ordered label names aligned with `values`.

    Returns:
        Dataframe containing original input columns and per-label prediction values.

    Raises:
        ValueError: If output row count does not match input row count.
        DataContractError: If label output columns conflict with existing input columns.
    """
    if values.shape[0] != len(input_df):
        raise ValueError(
            "Output row count does not match prediction input row count. "
            f"Got {values.shape[0]} output rows for {len(input_df)} input rows."
        )

    label_name_set = set(label_names)
    collisions = [col for col in input_df.columns if col in label_name_set]

    if collisions:
        raise DataContractError(
            "Prediction output label columns conflict with existing input columns. "
            f"Rename or remove these columns before prediction: {collisions}."
        )

    prediction_df = pd.DataFrame(
        values,
        columns=label_names,
    )
    return pd.concat(
        [input_df.reset_index(drop=True), prediction_df],
        axis=1,
    )
