"""Integration tests for the public train_tlmtc entrypoint."""

import json
from pathlib import Path

import pandas as pd
import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from typer.testing import CliRunner

from tlmtc.api import train_tlmtc
from tlmtc.cli import app
from tlmtc.data_contracts import TEXT_PAIR_COL, InputMode
from tlmtc.meta import read_run_meta
from tlmtc.paths import RunPaths, resolve_paths

pytestmark = pytest.mark.integration


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


GLOBAL_METRIC_KEYS = {
    "f1_micro",
    "f1_macro",
    "roc_auc_micro",
    "roc_auc_macro",
    "pr_auc_micro",
    "pr_auc_macro",
    "true_cardinality",
    "pred_cardinality",
}


def has_full_model_artifacts(model_dir: Path) -> bool:
    """Return whether full Hugging Face model artifacts were saved."""
    return (model_dir / "config.json").exists() and (
        (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    )


def has_peft_adapter_artifacts(model_dir: Path) -> bool:
    """Return whether PEFT adapter artifacts were saved."""
    return (model_dir / "adapter_config.json").exists() and (
        (model_dir / "adapter_model.safetensors").exists() or (model_dir / "adapter_model.bin").exists()
    )


def assert_saved_model_exists(model_dir: Path) -> None:
    """Assert that full-model or PEFT adapter artifacts were saved."""
    assert has_full_model_artifacts(model_dir) or has_peft_adapter_artifacts(model_dir)


def assert_peft_adapter_exists(model_dir: Path) -> None:
    """Assert that PEFT adapter artifacts were saved."""
    assert has_peft_adapter_artifacts(model_dir)


def assert_common_training_artifacts(paths: RunPaths) -> None:
    """Assert that the core train_tlmtc artifacts were written."""
    assert paths.train_data_path.exists()
    assert paths.val_data_path.exists()
    assert paths.test_data_path.exists()

    assert paths.model_dir.exists()
    assert_saved_model_exists(paths.model_dir)

    assert paths.global_metrics_path.exists()
    assert paths.label_metrics_path.exists()

    assert paths.global_metrics_table_path.exists()
    assert paths.label_metrics_table_path.exists()
    assert paths.hyperparameters_table_path.exists()

    assert paths.roc_plot_path.exists()
    assert paths.co_occurrence_plot_path.exists()
    assert paths.loss_plot_path.exists()

    global_metrics = json.loads(paths.global_metrics_path.read_text(encoding="utf-8"))
    label_metrics = json.loads(paths.label_metrics_path.read_text(encoding="utf-8"))

    assert GLOBAL_METRIC_KEYS.issubset(global_metrics)
    assert set(label_metrics) == {"a", "b"}

    train_meta = read_run_meta(paths.train_run_meta_path)

    assert train_meta.run_id == paths.run_id
    assert train_meta.target_name == "Target"
    assert train_meta.checkpoint
    assert train_meta.proxy_checkpoint
    assert train_meta.label_names == ["a", "b"]
    assert train_meta.threshold_type == "label"
    assert len(train_meta.thresholds) == 2
    assert train_meta.transfer_learning is True
    assert train_meta.threshold_optimization is True


def assert_paired_text_artifacts(paths: RunPaths) -> None:
    """Assert that paired-text inputs were preserved and reported."""
    for split_path in (paths.train_data_path, paths.val_data_path, paths.test_data_path):
        split_df = pd.read_parquet(split_path)
        assert TEXT_PAIR_COL in split_df.columns
        assert split_df[TEXT_PAIR_COL].notna().all()

    for table_path in (
        paths.global_metrics_table_path,
        paths.label_metrics_table_path,
        paths.hyperparameters_table_path,
    ):
        assert "Paired text" in table_path.read_text(encoding="utf-8")

    train_meta = read_run_meta(paths.train_run_meta_path)
    assert train_meta.input_mode is InputMode.PAIRED_TEXT


def test_train_tlmtc_runs_end_to_end_with_tiny_local_model(
    raw_multilabel_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run train_tlmtc end to end with a tiny local model and real artifacts."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    result = train_tlmtc(
        raw_csv=raw_multilabel_csv,
        work_dir=tmp_path,
        run_id="integration_smoke",
        checkpoint=str(tiny_checkpoint_dir),
        proxy_checkpoint=str(tiny_checkpoint_dir),
        hyperparameter_tuning=False,
        transfer_learning=True,
        threshold_optimization=True,
        wrap_peft=False,
        train_epochs=1,
        batch_size=4,
        sequence_length=16,
        validation_size=0.25,
        test_size=0.25,
        early_stopping_patience=1,
        use_cpu=True,
    )

    assert result.paths.run_id == "integration_smoke"
    assert_common_training_artifacts(result.paths)

    train_meta = read_run_meta(result.paths.train_run_meta_path)
    assert train_meta.input_mode is InputMode.SINGLE_TEXT
    assert train_meta.sequence_length == 16
    assert train_meta.hyperparameter_tuning is False
    assert train_meta.wrap_peft is False


def test_train_tlmtc_runs_end_to_end_with_paired_text_input(
    raw_paired_multilabel_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run train_tlmtc end to end with paired-text sequence-classification input."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    result = train_tlmtc(
        raw_csv=raw_paired_multilabel_csv,
        work_dir=tmp_path,
        run_id="integration_paired_text",
        checkpoint=str(tiny_checkpoint_dir),
        proxy_checkpoint=str(tiny_checkpoint_dir),
        hyperparameter_tuning=False,
        transfer_learning=True,
        threshold_optimization=True,
        wrap_peft=False,
        train_epochs=1,
        batch_size=4,
        sequence_length=24,
        validation_size=0.25,
        test_size=0.25,
        early_stopping_patience=1,
        use_cpu=True,
    )

    assert result.paths.run_id == "integration_paired_text"
    assert_common_training_artifacts(result.paths)
    assert_paired_text_artifacts(result.paths)


def test_train_tlmtc_runs_hpo_with_tiny_local_model(
    raw_multilabel_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run train_tlmtc end to end with Optuna HPO enabled."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    result = train_tlmtc(
        raw_csv=raw_multilabel_csv,
        work_dir=tmp_path,
        run_id="integration_hpo",
        checkpoint=str(tiny_checkpoint_dir),
        proxy_checkpoint=str(tiny_checkpoint_dir),
        hyperparameter_tuning=True,
        tuning_trials=1,
        optuna_space={
            "lr_low": 1e-5,
            "lr_high": 2e-5,
            "batch_sizes": [4],
            "wd_low": 0.0,
            "wd_high": 0.0,
            "schedulers": ["linear"],
            "epoch_low": 1,
            "epoch_high": 1,
            "lr_reference_batch_size": 4,
        },
        transfer_learning=True,
        threshold_optimization=True,
        wrap_peft=False,
        train_epochs=1,
        batch_size=4,
        sequence_length=16,
        validation_size=0.25,
        test_size=0.25,
        early_stopping_patience=1,
        use_cpu=True,
    )

    assert result.paths.run_id == "integration_hpo"
    assert_common_training_artifacts(result.paths)

    assert result.paths.optuna_trials_path.exists()
    assert result.paths.objective_values_plot_path.exists()

    train_meta = read_run_meta(result.paths.train_run_meta_path)
    assert train_meta.hyperparameter_tuning is True
    assert train_meta.wrap_peft is False


def test_train_tlmtc_runs_peft_with_tiny_local_model(
    raw_multilabel_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run train_tlmtc end to end with PEFT/LoRA enabled."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    result = train_tlmtc(
        raw_csv=raw_multilabel_csv,
        work_dir=tmp_path,
        run_id="integration_peft",
        checkpoint=str(tiny_checkpoint_dir),
        proxy_checkpoint=str(tiny_checkpoint_dir),
        hyperparameter_tuning=False,
        transfer_learning=True,
        threshold_optimization=True,
        wrap_peft=True,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_bias="none",
        train_epochs=1,
        batch_size=4,
        sequence_length=16,
        validation_size=0.25,
        test_size=0.25,
        early_stopping_patience=1,
        use_cpu=True,
    )

    assert result.paths.run_id == "integration_peft"
    assert_common_training_artifacts(result.paths)
    assert_peft_adapter_exists(result.paths.model_dir)

    train_meta = read_run_meta(result.paths.train_run_meta_path)
    assert train_meta.wrap_peft is True


def test_tlmtc_train_cli_runs_end_to_end_with_tiny_local_model(
    raw_multilabel_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the tlmtc train CLI end to end with a tiny local model."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "--raw-csv",
            str(raw_multilabel_csv),
            "--work-dir",
            str(tmp_path),
            "--run-id",
            "integration_cli",
            "--checkpoint",
            str(tiny_checkpoint_dir),
            "--proxy-checkpoint",
            str(tiny_checkpoint_dir),
            "--no-hyperparameter-tuning",
            "--transfer-learning",
            "--threshold-optimization",
            "--no-wrap-peft",
            "--train-epochs",
            "1",
            "--batch-size",
            "4",
            "--sequence-length",
            "16",
            "--validation-size",
            "0.25",
            "--test-size",
            "0.25",
            "--early-stopping-patience",
            "1",
            "--use-cpu",
        ],
    )

    paths = resolve_paths(
        raw_csv=raw_multilabel_csv,
        raw_test_csv=None,
        work_dir=tmp_path,
        run_id="integration_cli",
    )

    assert result.exit_code == 0, result.output
    assert f"Run completed: {paths.run_dir}" in result.output
    assert_common_training_artifacts(paths)

    train_meta = read_run_meta(paths.train_run_meta_path)
    assert train_meta.run_id == "integration_cli"
    assert train_meta.sequence_length == 16
    assert train_meta.wrap_peft is False


def test_tlmtc_train_cli_quiet_runtime_mode_suppresses_progress_output(
    raw_multilabel_csv: Path,
    tiny_checkpoint_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the tlmtc train CLI with quiet runtime verbosity."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "--raw-csv",
            str(raw_multilabel_csv),
            "--work-dir",
            str(tmp_path),
            "--run-id",
            "integration_cli_quiet",
            "--checkpoint",
            str(tiny_checkpoint_dir),
            "--proxy-checkpoint",
            str(tiny_checkpoint_dir),
            "--no-hyperparameter-tuning",
            "--transfer-learning",
            "--threshold-optimization",
            "--no-wrap-peft",
            "--train-epochs",
            "1",
            "--batch-size",
            "4",
            "--sequence-length",
            "16",
            "--validation-size",
            "0.25",
            "--test-size",
            "0.25",
            "--early-stopping-patience",
            "1",
            "--use-cpu",
            "--verbosity",
            "quiet",
        ],
    )

    paths = resolve_paths(
        raw_csv=raw_multilabel_csv,
        raw_test_csv=None,
        work_dir=tmp_path,
        run_id="integration_cli_quiet",
    )

    assert result.exit_code == 0, result.output
    assert "tlmtc:" not in result.output
    assert f"Run completed: {paths.run_dir}" in result.output
    assert_common_training_artifacts(paths)

    train_meta = read_run_meta(paths.train_run_meta_path)
    assert train_meta.run_id == "integration_cli_quiet"
    assert train_meta.sequence_length == 16
    assert train_meta.wrap_peft is False
