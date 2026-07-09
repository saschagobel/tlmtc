"""Prediction operations for Hugging Face multi-label text classification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tlmtc.data_contracts import DataContractError

if TYPE_CHECKING:
    from datasets import Dataset
    from onnxruntime import InferenceSession
    from peft import PeftModel
    from transformers import PreTrainedModel

type PredictionModel = PreTrainedModel | PeftModel | InferenceSession


def load_prediction_model(
    model_dir: Path,
    *,
    inference_backend: str,
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
    trust_remote_code: bool,
) -> PredictionModel:
    """Load a trained prediction model from full-model or PEFT artifacts.

    Args:
        model_dir: Directory containing saved model or adapter artifacts.
        inference_backend: Runtime backend used for prediction.
        checkpoint: Base checkpoint used when loading PEFT adapters.
        num_labels: Number of labels in the trained classification head.
        wrap_peft: Whether `model_dir` contains PEFT adapter artifacts.
        trust_remote_code: Whether Hugging Face model loading may execute custom remote code.

    Returns:
        Loaded prediction model for the selected backend.
    """
    if inference_backend == "onnx":
        return _load_onnx_prediction_model(model_dir / "onnx")

    return _load_torch_prediction_model(
        model_dir=model_dir,
        checkpoint=checkpoint,
        num_labels=num_labels,
        wrap_peft=wrap_peft,
        trust_remote_code=trust_remote_code,
    )


def _load_torch_prediction_model(
    model_dir: Path,
    *,
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
    trust_remote_code: bool,
) -> PreTrainedModel | PeftModel:
    """Load a trained PyTorch prediction model from full-model or PEFT artifacts.

    Args:
        model_dir: Directory containing saved model or adapter artifacts.
        checkpoint: Base checkpoint used when loading PEFT adapters.
        num_labels: Number of labels in the trained classification head.
        wrap_peft: Whether `model_dir` contains PEFT adapter artifacts.
        trust_remote_code: Whether Hugging Face model loading may execute custom remote code.

    Returns:
        Loaded Transformers or PEFT sequence-classification model.
    """
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    if wrap_peft:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
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
        trust_remote_code=trust_remote_code,
    )


def _load_onnx_prediction_model(
    onnx_model_dir: Path,
) -> InferenceSession:
    """Load a trained ONNX Runtime inference session from exported artifacts.

    Args:
        onnx_model_dir: Directory containing exported ONNX model artifacts.

    Returns:
        ONNX Runtime inference session for the single exported model file.

    Raises:
        RuntimeError: If ONNX Runtime is not installed or the export directory does not contain exactly one ONNX file.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX prediction requires the optional ONNX runtime dependencies. Install them with `tlmtc[onnx-runtime]`."
        ) from exc

    onnx_files = list(onnx_model_dir.rglob("*.onnx"))
    if len(onnx_files) != 1:
        raise RuntimeError(f"Expected exactly one ONNX model under {onnx_model_dir}, found {len(onnx_files)}.")

    return ort.InferenceSession(
        str(onnx_files[0]),
        providers=["CPUExecutionProvider"],
    )


def predict_probabilities(
    model: PredictionModel,
    dataset: Dataset,
    batch_size: int,
    use_cpu: bool,
    inference_backend: str,
) -> np.ndarray:
    """Predict multi-label probabilities for a tokenized prediction dataset.

    Args:
        model: Loaded prediction model for the selected backend.
        dataset: Tokenized Hugging Face prediction dataset.
        batch_size: Prediction batch size.
        use_cpu: Whether to force CPU inference.
        inference_backend: Runtime backend used for prediction.

    Returns:
        Probability matrix with shape `(n_rows, n_labels)`.
    """
    if inference_backend == "onnx":
        return _predict_onnx_probabilities(model, dataset, batch_size)

    return _predict_torch_probabilities(model, dataset, batch_size, use_cpu)


def _predict_torch_probabilities(
    model: PreTrainedModel | PeftModel,
    dataset: Dataset,
    batch_size: int,
    use_cpu: bool,
) -> np.ndarray:
    """Predict multi-label probabilities with a PyTorch model.

    Args:
        model: Loaded Transformers or PEFT sequence-classification model.
        dataset: Tokenized Hugging Face prediction dataset in torch format.
        batch_size: Prediction batch size.
        use_cpu: Whether to force CPU inference.

    Returns:
        Probability matrix with shape `(n_rows, n_labels)`.
    """
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

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


def _predict_onnx_probabilities(
    model: InferenceSession,
    dataset: Dataset,
    batch_size: int,
) -> np.ndarray:
    """Predict multi-label probabilities with an ONNX Runtime inference session.

    Args:
        model: ONNX Runtime inference session returned by `_load_onnx_prediction_model`.
        dataset: Tokenized Hugging Face prediction dataset in numpy format.
        batch_size: Prediction batch size.

    Returns:
        Probability matrix with shape `(n_rows, n_labels)`.
    """
    input_names = {input_.name for input_ in model.get_inputs()}
    probabilities: list[np.ndarray] = []

    for batch in dataset.iter(batch_size=batch_size):
        inputs = {name: batch[name] for name in input_names if name in batch}
        logits = model.run(None, inputs)[0]
        probabilities.append(1.0 / (1.0 + np.exp(-logits)))

    return np.concatenate(probabilities, axis=0)


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
