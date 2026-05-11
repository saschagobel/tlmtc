"""Integration tests for the public predict_tlmtc entrypoint."""

import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from typer.testing import CliRunner

from tlmtc.api import predict_tlmtc, train_tlmtc
from tlmtc.cli import app
from tlmtc.data_contracts import TEXT_PAIR_COL, InputMode
from tlmtc.meta import read_run_meta, write_run_meta
from tlmtc.paths import PredictionPaths, RunPaths, resolve_prediction_paths

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def offline_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Hugging Face components to use local test artifacts only."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture
def raw_multilabel_csv(tmp_path: Path) -> Path:
    """Create a small valid multilabel CSV for end-to-end training."""
    label_patterns = [
        ("alpha policy text", 1, 0),
        ("beta policy text", 0, 1),
        ("alpha beta policy text", 1, 1),
        ("neutral policy text", 0, 0),
    ]

    rows = [
        {
            "text": f"{text} {i}",
            "label_a": label_a,
            "label_b": label_b,
        }
        for i in range(8)
        for text, label_a, label_b in label_patterns
    ]

    raw_csv = tmp_path / "raw_multilabel.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)
    return raw_csv


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
def prediction_csv(tmp_path: Path) -> Path:
    """Create a small unlabeled single-text prediction CSV."""
    rows = [
        {"record_id": "doc-1", "text": "alpha policy text"},
        {"record_id": "doc-2", "text": "beta policy text"},
        {"record_id": "doc-3", "text": "alpha beta policy text"},
        {"record_id": "doc-4", "text": "neutral policy text"},
    ]

    prediction_path = tmp_path / "prediction.csv"
    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    return prediction_path


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


def _train_tiny_model(
    *,
    raw_csv: Path,
    tiny_checkpoint_dir: Path,
    work_dir: Path,
    run_id: str,
    sequence_length: int,
) -> RunPaths:
    """Train a tiny local model and return its training paths."""
    result = train_tlmtc(
        raw_csv=raw_csv,
        work_dir=work_dir,
        run_id=run_id,
        checkpoint=str(tiny_checkpoint_dir),
        proxy_checkpoint=str(tiny_checkpoint_dir),
        hyperparameter_tuning=False,
        transfer_learning=True,
        threshold_optimization=True,
        wrap_peft=False,
        train_epochs=1,
        batch_size=4,
        sequence_length=sequence_length,
        validation_size=0.25,
        test_size=0.25,
        early_stopping_patience=1,
        use_cpu=True,
    )

    assert result.paths.train_run_meta_path.exists()
    assert result.paths.model_dir.exists()
    assert (result.paths.model_dir / "config.json").exists()
    assert (result.paths.model_dir / "model.safetensors").exists() or (
        result.paths.model_dir / "pytorch_model.bin"
    ).exists()

    return result.paths


def assert_prediction_artifacts(
    paths: PredictionPaths,
    input_csv: Path,
    expected_input_mode: InputMode,
) -> None:
    """Assert that prediction probability and binary output artifacts are valid."""
    assert paths.prediction_run_dir.exists()
    assert paths.probabilities_path.exists()
    assert paths.predictions_path.exists()

    meta = read_run_meta(paths.train_run_meta_path)

    assert meta.run_id == paths.run_id
    assert meta.transfer_learning is True
    assert meta.input_mode is expected_input_mode
    assert meta.sequence_length > 0
    assert meta.checkpoint
    assert meta.label_names == ["a", "b"]
    assert len(meta.thresholds) in {1, len(meta.label_names)}

    input_df = pd.read_csv(input_csv)
    probabilities = pd.read_csv(paths.probabilities_path)
    predictions = pd.read_csv(paths.predictions_path)

    expected_columns = [*input_df.columns, *meta.label_names]
    assert list(probabilities.columns) == expected_columns
    assert list(predictions.columns) == expected_columns

    assert len(probabilities) == len(input_df)
    assert len(predictions) == len(input_df)

    pd.testing.assert_frame_equal(
        probabilities[list(input_df.columns)],
        input_df,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        predictions[list(input_df.columns)],
        input_df,
        check_dtype=False,
    )

    probability_values = probabilities[meta.label_names].to_numpy(dtype=float)
    prediction_values = predictions[meta.label_names].to_numpy()

    assert np.isfinite(probability_values).all()
    assert ((probability_values >= 0.0) & (probability_values <= 1.0)).all()
    assert set(np.unique(prediction_values)).issubset({0, 1})

    expected_predictions = (probability_values >= np.asarray(meta.thresholds, dtype=float)).astype(int)
    np.testing.assert_array_equal(prediction_values, expected_predictions)

    if expected_input_mode is InputMode.PAIRED_TEXT:
        assert TEXT_PAIR_COL in probabilities.columns
        assert TEXT_PAIR_COL in predictions.columns
        assert probabilities[TEXT_PAIR_COL].notna().all()
        assert predictions[TEXT_PAIR_COL].notna().all()


@pytest.mark.parametrize(
    (
        "raw_csv_fixture",
        "prediction_csv_fixture",
        "run_id",
        "sequence_length",
        "expected_input_mode",
    ),
    [
        (
            "raw_multilabel_csv",
            "prediction_csv",
            "integration_prediction_single",
            16,
            InputMode.SINGLE_TEXT,
        ),
        (
            "raw_paired_multilabel_csv",
            "paired_prediction_csv",
            "integration_prediction_paired",
            24,
            InputMode.PAIRED_TEXT,
        ),
    ],
)
def test_predict_tlmtc_runs_end_to_end_with_tiny_local_model(
    raw_csv_fixture: str,
    prediction_csv_fixture: str,
    run_id: str,
    sequence_length: int,
    expected_input_mode: InputMode,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Run predict_tlmtc end to end from tiny single-text and paired-text training runs."""
    raw_csv = request.getfixturevalue(raw_csv_fixture)
    prediction_csv = request.getfixturevalue(prediction_csv_fixture)

    train_paths = _train_tiny_model(
        raw_csv=raw_csv,
        tiny_checkpoint_dir=tiny_checkpoint_dir,
        work_dir=tmp_path,
        run_id=run_id,
        sequence_length=sequence_length,
    )

    train_meta = read_run_meta(train_paths.train_run_meta_path)
    assert train_meta.input_mode is expected_input_mode
    assert train_meta.label_names == ["a", "b"]

    result = predict_tlmtc(
        prediction_csv=prediction_csv,
        work_dir=tmp_path,
        run_id=train_paths.run_id,
        batch_size=2,
        use_cpu=True,
    )

    assert result.paths.run_id == run_id
    assert result.paths.train_run_dir == train_paths.run_dir
    assert result.paths.train_run_model_dir == train_paths.model_dir
    assert_prediction_artifacts(
        paths=result.paths,
        input_csv=prediction_csv,
        expected_input_mode=expected_input_mode,
    )


def test_predict_tlmtc_uses_latest_training_run_when_run_id_is_omitted(
    raw_multilabel_csv: Path,
    prediction_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
) -> None:
    """Run predict_tlmtc with implicit latest-run selection."""
    source_paths = _train_tiny_model(
        raw_csv=raw_multilabel_csv,
        tiny_checkpoint_dir=tiny_checkpoint_dir,
        work_dir=tmp_path,
        run_id="integration_prediction_older",
        sequence_length=16,
    )

    latest_run_id = "integration_prediction_latest"
    latest_run_dir = source_paths.run_dir.parent / latest_run_id
    shutil.copytree(source_paths.run_dir, latest_run_dir)

    source_meta = read_run_meta(source_paths.train_run_meta_path)
    latest_meta = source_meta.model_copy(
        update={
            "run_id": latest_run_id,
            "created_at": source_meta.created_at + timedelta(seconds=1),
        },
    )
    write_run_meta(
        meta=latest_meta,
        path=latest_run_dir / source_paths.train_run_meta_path.name,
    )

    result = predict_tlmtc(
        prediction_csv=prediction_csv,
        work_dir=tmp_path,
        batch_size=2,
        use_cpu=True,
    )

    assert result.paths.run_id == latest_run_id
    assert result.paths.train_run_dir == latest_run_dir
    assert_prediction_artifacts(
        paths=result.paths,
        input_csv=prediction_csv,
        expected_input_mode=InputMode.SINGLE_TEXT,
    )


def test_tlmtc_predict_cli_runs_end_to_end_with_tiny_local_model(
    raw_multilabel_csv: Path,
    prediction_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
) -> None:
    """Run the tlmtc predict CLI end to end with a tiny local model."""
    train_paths = _train_tiny_model(
        raw_csv=raw_multilabel_csv,
        tiny_checkpoint_dir=tiny_checkpoint_dir,
        work_dir=tmp_path,
        run_id="integration_prediction_cli",
        sequence_length=16,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "predict",
            "--prediction-csv",
            str(prediction_csv),
            "--work-dir",
            str(tmp_path),
            "--run-id",
            train_paths.run_id,
            "--batch-size",
            "2",
            "--use-cpu",
        ],
    )

    paths = resolve_prediction_paths(
        input_csv=prediction_csv,
        work_dir=tmp_path,
        run_id=train_paths.run_id,
    )

    assert result.exit_code == 0, result.output
    assert f"Prediction completed: {paths.prediction_run_dir}" in result.output
    assert f"Probabilities: {paths.probabilities_path}" in result.output
    assert f"Predictions: {paths.predictions_path}" in result.output

    assert_prediction_artifacts(
        paths=paths,
        input_csv=prediction_csv,
        expected_input_mode=InputMode.SINGLE_TEXT,
    )
