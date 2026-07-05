"""Integration tests for ONNX export parity."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification, BertTokenizer

from tlmtc.api import train_tlmtc
from tlmtc.data_contracts import TEXT_PAIR_COL
from tlmtc.meta import read_run_meta
from tlmtc.prediction import load_prediction_model

pytestmark = pytest.mark.integration

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("olive.cli.api")


@pytest.fixture(autouse=True)
def offline_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Hugging Face components to use local test artifacts only."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture
def raw_paired_multilabel_csv(tmp_path: Path) -> Path:
    """Create a small valid paired-text multilabel CSV for end-to-end training."""
    label_patterns = [
        ("alpha policy query", "alpha evidence response", 1, 0),
        ("beta policy query", "beta evidence response", 0, 1),
        ("alpha beta policy query", "combined evidence response", 1, 1),
        ("neutral policy query", "neutral background response", 0, 0),
    ]

    rows = [
        {
            "text": f"{query} {i}",
            TEXT_PAIR_COL: f"{response} {i}",
            "label_a": label_a,
            "label_b": label_b,
        }
        for i in range(8)
        for query, response, label_a, label_b in label_patterns
    ]

    raw_csv = tmp_path / "raw_paired_multilabel.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)
    return raw_csv


@pytest.fixture
def tiny_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a tiny local Bert checkpoint with a compatible tokenizer."""
    checkpoint_dir = tmp_path / "tiny_bert_checkpoint"
    checkpoint_dir.mkdir()

    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "alpha",
        "beta",
        "policy",
        "text",
        "neutral",
        "query",
        "evidence",
        "response",
        "combined",
        "background",
        *[str(i) for i in range(8)],
    ]
    vocab_path = checkpoint_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")

    tokenizer = BertTokenizer(
        vocab_file=str(vocab_path),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(checkpoint_dir)

    config = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_labels=2,
        problem_type="multi_label_classification",
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(checkpoint_dir)

    return checkpoint_dir


def _load_onnx_session(onnx_model_dir: Path) -> ort.InferenceSession:
    """Load the single exported ONNX model from the export directory."""
    onnx_files = sorted(onnx_model_dir.rglob("*.onnx"))

    assert onnx_files, f"No ONNX model file found under {onnx_model_dir}"
    assert len(onnx_files) == 1

    return ort.InferenceSession(
        str(onnx_files[0]),
        providers=["CPUExecutionProvider"],
    )


def _torch_probabilities(
    *,
    model_dir: Path,
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
    encoded_inputs: dict[str, torch.Tensor],
) -> np.ndarray:
    """Run PyTorch inference from persisted tlmtc model artifacts."""
    model = load_prediction_model(
        model_dir=model_dir,
        checkpoint=checkpoint,
        num_labels=num_labels,
        wrap_peft=wrap_peft,
        trust_remote_code=False,
    )
    model.eval()

    with torch.inference_mode():
        logits = model(**encoded_inputs).logits

    return torch.sigmoid(logits).detach().cpu().numpy()


def _onnx_probabilities(
    *,
    onnx_model_dir: Path,
    encoded_inputs: dict[str, torch.Tensor],
) -> np.ndarray:
    """Run ONNX Runtime inference from exported artifacts."""
    session = _load_onnx_session(onnx_model_dir)
    input_names = {input_.name for input_ in session.get_inputs()}
    session_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in encoded_inputs.items()
        if name in input_names
    }

    logits = session.run(None, session_inputs)[0]
    return 1.0 / (1.0 + np.exp(-logits))


@pytest.mark.parametrize("wrap_peft", [False, True], ids=["full", "peft"])
def test_train_tlmtc_exports_onnx_with_pytorch_parity_for_paired_text(
    raw_paired_multilabel_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
    wrap_peft: bool,
) -> None:
    """Export ONNX from real training and compare PyTorch and ONNX probabilities."""
    result = train_tlmtc(
        labeled_data=raw_paired_multilabel_csv,
        work_dir=tmp_path,
        run_id=f"integration_onnx_{'peft' if wrap_peft else 'full'}",
        checkpoint=str(tiny_checkpoint_dir),
        proxy_checkpoint=str(tiny_checkpoint_dir),
        hyperparameter_tuning=False,
        transfer_learning=True,
        threshold_optimization=False,
        wrap_peft=wrap_peft,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_bias="none",
        train_epochs=1,
        batch_size=4,
        sequence_length=24,
        validation_size=0.25,
        test_size=0.25,
        early_stopping_patience=1,
        use_cpu=True,
        export_onnx=True,
    )

    meta = read_run_meta(result.paths.train_run_meta_path)
    assert meta.model_backends == ["torch", "onnx"]
    assert result.paths.onnx_model_dir.is_dir()

    tokenizer = AutoTokenizer.from_pretrained(result.paths.model_dir)
    encoded_inputs = tokenizer(
        ["alpha policy query", "beta policy query", "neutral policy query"],
        ["alpha evidence response", "beta evidence response", "neutral background response"],
        truncation="longest_first",
        padding="max_length",
        max_length=meta.sequence_length,
        return_tensors="pt",
    )

    torch_probs = _torch_probabilities(
        model_dir=result.paths.model_dir,
        checkpoint=meta.checkpoint,
        num_labels=len(meta.label_names or []),
        wrap_peft=meta.wrap_peft,
        encoded_inputs=dict(encoded_inputs),
    )
    onnx_probs = _onnx_probabilities(
        onnx_model_dir=result.paths.onnx_model_dir,
        encoded_inputs=dict(encoded_inputs),
    )

    np.testing.assert_allclose(onnx_probs, torch_probs, rtol=1e-3, atol=1e-3)
