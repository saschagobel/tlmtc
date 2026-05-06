"""Typer command-line interface for the tlmtc training pipeline."""

import json
from pathlib import Path
from typing import Any

import typer

from tlmtc import __version__
from tlmtc.settings import UNSET, Unset

app = typer.Typer(
    name="tlmtc",
    help="Transfer learning for multi-label text classification.",
)


def parse_optuna_space(
    value: str | None,
) -> dict[str, Any] | Unset:
    """Parse an optional Optuna search-space override from JSON or an @file path.

    Args:
        value: JSON object string, @file path, or None when the CLI option is omitted.

    Returns:
        Parsed Optuna search-space override, or UNSET for omitted CLI values.

    Raises:
        typer.BadParameter: If the value cannot be read, decoded, or parsed as a JSON object.
    """
    if value is None:
        return UNSET

    try:
        if value.startswith("@"):
            parsed = json.loads(Path(value[1:]).read_text(encoding="utf-8"))
        else:
            parsed = json.loads(value)
    except OSError as exc:
        raise typer.BadParameter(f"Could not read JSON file: {value}") from exc
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("Expected a JSON object or @file.json.") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter("Expected a JSON object like '{\"lr_low\": 1e-5}'.")

    return parsed


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the installed tlmtc version and exit.",
        is_eager=True,
    ),
) -> None:
    """Handle root CLI invocation, version output, and help display."""
    if version:
        typer.echo(__version__)
        raise typer.Exit(code=0)

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@app.command("train")
def train_command(
    raw_csv: str = typer.Option(
        ...,
        "--raw-csv",
        help="Path to the multilabel CSV.",
    ),
    raw_test_csv: str | None = typer.Option(
        None,
        "--raw-test-csv",
        help="Optional path to a test CSV. If omitted, a test split is created from --raw-csv.",
    ),
    work_dir: str | None = typer.Option(
        None,
        "--work-dir",
        help="Base directory for resolving relative inputs and creating the run directory.",
    ),
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        help="Optional path to a YAML configuration file.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run identifier used to name the run directory. If it exists, the run may resume.",
    ),
    target_name: str | None = typer.Option(
        None,
        "--target-name",
        help="Display name for the classification target/task.",
    ),
    validation_size: float | None = typer.Option(
        None,
        "--validation-size",
        help="Fraction of data used for validation split.",
    ),
    test_size: float | None = typer.Option(
        None,
        "--test-size",
        help="Fraction of data used for test split when --raw-test-csv is omitted.",
    ),
    random_seed: int | None = typer.Option(
        None,
        "--random-seed",
        help="Random seed used for splitting/shuffling.",
    ),
    transfer_learning: bool | None = typer.Option(
        None,
        "--transfer-learning/--no-transfer-learning",
        help="Whether to fine-tune a pretrained checkpoint.",
    ),
    hyperparameter_tuning: bool | None = typer.Option(
        None,
        "--hyperparameter-tuning/--no-hyperparameter-tuning",
        help="Whether to run Optuna hyperparameter tuning.",
    ),
    threshold_optimization: bool | None = typer.Option(
        None,
        "--threshold-optimization/--no-threshold-optimization",
        help="Whether to tune decision threshold(s) post-training.",
    ),
    threshold_type: str | None = typer.Option(
        None,
        "--threshold-type",
        help="Threshold mode.",
    ),
    scale_learning_rate: bool | None = typer.Option(
        None,
        "--scale-learning-rate/--no-scale-learning-rate",
        help="Whether to scale learning rate based on proxy/full checkpoint size.",
    ),
    wrap_peft: bool | None = typer.Option(
        None,
        "--wrap-peft/--no-wrap-peft",
        help="Whether to apply PEFT/LoRA wrapping.",
    ),
    proxy_checkpoint: str | None = typer.Option(
        None,
        "--proxy-checkpoint",
        help="Proxy pretrained model checkpoint used for HPO.",
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help="Base pretrained model checkpoint identifier.",
    ),
    sequence_length: int | None = typer.Option(
        None,
        "--sequence-length",
        help="Max sequence length for tokenization.",
    ),
    best_model_metric: str | None = typer.Option(
        None,
        "--best-model-metric",
        help="Metric used to select the best model checkpoint.",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="Training batch size.",
    ),
    train_epochs: int | None = typer.Option(
        None,
        "--train-epochs",
        help="Number of training epochs.",
    ),
    learning_rate: float | None = typer.Option(
        None,
        "--learning-rate",
        help="Initial learning rate.",
    ),
    weight_decay: float | None = typer.Option(
        None,
        "--weight-decay",
        help="Weight decay.",
    ),
    lr_scheduler: str | None = typer.Option(
        None,
        "--lr-scheduler",
        help="Learning rate scheduler identifier/name.",
    ),
    best_threshold_metric: str | None = typer.Option(
        None,
        "--best-threshold-metric",
        help="Metric used to select optimal decision threshold(s).",
    ),
    tuning_trials: int | None = typer.Option(
        None,
        "--tuning-trials",
        help="Number of Optuna trials.",
    ),
    optuna_space: str | None = typer.Option(
        None,
        "--optuna-space",
        help=(
            "Optional partial override for the Optuna search space as JSON or @file.json. "
            "Values are merged into the default space."
        ),
    ),
    lora_r: int | None = typer.Option(
        None,
        "--lora-r",
        help="LoRA rank.",
    ),
    lora_alpha: int | None = typer.Option(
        None,
        "--lora-alpha",
        help="LoRA alpha.",
    ),
    lora_dropout: float | None = typer.Option(
        None,
        "--lora-dropout",
        help="LoRA dropout.",
    ),
    lora_bias: str | None = typer.Option(
        None,
        "--lora-bias",
        help="LoRA bias handling.",
    ),
    early_stopping_patience: int | None = typer.Option(
        None,
        "--early-stopping-patience",
        help="Early stopping patience in epochs without improvement.",
    ),
    use_cpu: bool | None = typer.Option(
        None,
        "--use-cpu/--no-use-cpu",
        help="Force CPU execution.",
    ),
) -> None:
    """Run the tlmtc training pipeline from CLI options."""
    from tlmtc.api import train_tlmtc

    result = train_tlmtc(
        raw_csv=raw_csv,
        raw_test_csv=UNSET if raw_test_csv is None else raw_test_csv,
        work_dir=UNSET if work_dir is None else work_dir,
        config_path=UNSET if config_path is None else config_path,
        run_id=UNSET if run_id is None else run_id,
        target_name=UNSET if target_name is None else target_name,
        validation_size=UNSET if validation_size is None else validation_size,
        test_size=UNSET if test_size is None else test_size,
        random_seed=UNSET if random_seed is None else random_seed,
        transfer_learning=UNSET if transfer_learning is None else transfer_learning,
        hyperparameter_tuning=UNSET if hyperparameter_tuning is None else hyperparameter_tuning,
        threshold_optimization=UNSET if threshold_optimization is None else threshold_optimization,
        threshold_type=UNSET if threshold_type is None else threshold_type,
        scale_learning_rate=UNSET if scale_learning_rate is None else scale_learning_rate,
        wrap_peft=UNSET if wrap_peft is None else wrap_peft,
        proxy_checkpoint=UNSET if proxy_checkpoint is None else proxy_checkpoint,
        checkpoint=UNSET if checkpoint is None else checkpoint,
        sequence_length=UNSET if sequence_length is None else sequence_length,
        best_model_metric=UNSET if best_model_metric is None else best_model_metric,
        batch_size=UNSET if batch_size is None else batch_size,
        train_epochs=UNSET if train_epochs is None else train_epochs,
        learning_rate=UNSET if learning_rate is None else learning_rate,
        weight_decay=UNSET if weight_decay is None else weight_decay,
        lr_scheduler=UNSET if lr_scheduler is None else lr_scheduler,
        best_threshold_metric=UNSET if best_threshold_metric is None else best_threshold_metric,
        tuning_trials=UNSET if tuning_trials is None else tuning_trials,
        optuna_space=parse_optuna_space(optuna_space),
        lora_r=UNSET if lora_r is None else lora_r,
        lora_alpha=UNSET if lora_alpha is None else lora_alpha,
        lora_dropout=UNSET if lora_dropout is None else lora_dropout,
        lora_bias=UNSET if lora_bias is None else lora_bias,
        early_stopping_patience=UNSET if early_stopping_patience is None else early_stopping_patience,
        use_cpu=UNSET if use_cpu is None else use_cpu,
    )

    typer.echo(f"Run completed: {result.paths.run_dir}")
