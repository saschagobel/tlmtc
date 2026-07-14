"""Integration tests for ONNX Runtime prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlmtc.api import predict_tlmtc, train_tlmtc
from tlmtc.data_contracts import TEXT_PAIR_COL
from tlmtc.meta import read_run_meta

pytestmark = pytest.mark.integration

pytest.importorskip("onnxruntime")
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
def paired_prediction_csv(tmp_path: Path) -> Path:
    """Create a small unlabeled paired-text prediction CSV."""
    rows = [
        {
            "record_id": "pair-1",
            "text": "alpha policy query",
            TEXT_PAIR_COL: "alpha evidence response",
        },
        {
            "record_id": "pair-2",
            "text": "beta policy query",
            TEXT_PAIR_COL: "beta evidence response",
        },
        {
            "record_id": "pair-3",
            "text": "alpha beta policy query",
            TEXT_PAIR_COL: "combined evidence response",
        },
        {
            "record_id": "pair-4",
            "text": "neutral policy query",
            TEXT_PAIR_COL: "neutral background response",
        },
    ]

    prediction_path = tmp_path / "paired_prediction.csv"
    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    return prediction_path


@pytest.fixture
def tiny_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a tiny local Bert checkpoint with a compatible tokenizer."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

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


def test_predict_tlmtc_onnx_backend_uses_persisted_tokenizer_and_matches_torch_backend(
    raw_paired_multilabel_csv: Path,
    paired_prediction_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
) -> None:
    """Run offline torch and ONNX prediction with only persisted training-run artifacts."""
    train_result = train_tlmtc(
        labeled_data=raw_paired_multilabel_csv,
        work_dir=tmp_path,
        run_id="integration_onnx_prediction",
        checkpoint=str(tiny_checkpoint_dir),
        proxy_checkpoint=str(tiny_checkpoint_dir),
        hyperparameter_tuning=False,
        transfer_learning=True,
        threshold_optimization=False,
        wrap_peft=False,
        train_epochs=1,
        batch_size=4,
        sequence_length=24,
        validation_size=0.25,
        test_size=0.25,
        early_stopping_patience=1,
        use_cpu=True,
        export_onnx=True,
    )

    meta = read_run_meta(train_result.paths.train_run_meta_path)
    assert meta.model_backends == ["torch", "onnx"]
    assert train_result.paths.onnx_model_dir.is_dir()

    tiny_checkpoint_dir.rename(tmp_path / "unavailable_checkpoint")

    torch_result = predict_tlmtc(
        unlabeled_data=paired_prediction_csv,
        work_dir=tmp_path,
        run_id=train_result.paths.run_id,
        inference_backend="torch",
        batch_size=2,
        use_cpu=True,
    )
    torch_probabilities = pd.read_csv(torch_result.paths.probabilities_path)

    onnx_result = predict_tlmtc(
        unlabeled_data=paired_prediction_csv,
        work_dir=tmp_path,
        run_id=train_result.paths.run_id,
        inference_backend="onnx",
        batch_size=2,
        use_cpu=True,
    )
    onnx_probabilities = pd.read_csv(onnx_result.paths.probabilities_path)

    assert onnx_result.paths.run_id == torch_result.paths.run_id
    assert onnx_result.paths.predictions_path.exists()
    assert list(onnx_probabilities.columns) == list(torch_probabilities.columns)
    np.testing.assert_allclose(
        onnx_probabilities[meta.label_names].to_numpy(),
        torch_probabilities[meta.label_names].to_numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
