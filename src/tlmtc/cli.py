"""Run the tlmtc pipeline via the command line.

Defines the argparse-based CLI entrypoint that maps command-line arguments to `run_tlmtc()`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

_THRESHOLD_CHOICES = ("global", "label")
_METRIC_CHOICES = ("f1_micro", "f1_macro", "roc_auc_micro", "roc_auc_macro")
_LORA_BIAS_CHOICES = ("none", "all", "lora_only")


def _json_or_file(
    value: str,
) -> dict[str, Any]:
    """Parse a JSON object from a string or an '@'-prefixed file path.

    Args:
        value: JSON string, or an '@'-prefixed path to a JSON file.

    Returns:
        parsed: Parsed JSON object as a dictionary.
    """
    try:
        if value.startswith("@"):
            with open(value[1:], "r", encoding="utf-8") as f:
                parsed = json.load(f)
        else:
            parsed = json.loads(value)
    except (OSError, json.JSONDecodeError) as e:
        raise argparse.ArgumentTypeError(
            "Invalid JSON for --optuna-space-user (expected JSON object or @file.json)."
        ) from e

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "Invalid JSON for --optuna-space-user (expected a JSON object like '{\"k\": 1}')."
        )

    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the tlmtc CLI argument parser.

    Returns:
        argparse.ArgumentParser: configured with flags corresponding to the `run_tlmtc()`
           keyword arguments.
    """
    parser = argparse.ArgumentParser(
        prog="tlmtc",
        description="Run the full tlmtc pipeline end-to-end.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--raw-csv",
        type=str,
        required=True,
        help="Path to the multilabel CSV.",
    )
    parser.add_argument(
        "--raw-test-csv",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            "Optional path to a test CSV. If omitted, a test split is created from --raw-csv according to --test-size."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Base directory for resolving relative inputs and creating the run directory.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=argparse.SUPPRESS,
        help="Optional path to a YAML configuration file.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=argparse.SUPPRESS,
        help="Optional run identifier used to name the run directory. If exists will resume.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=argparse.SUPPRESS,
        help="Display name for the classification target/task (used in logs/outputs).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of data used for validation split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of data used for test split (only used when `raw_test_csv` is None).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Random seed used for splitting/shuffling.",
    )
    parser.add_argument(
        "--transfer-learning",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Whether to fine-tune a pretrained checkpoint.",
    )
    parser.add_argument(
        "--hyperparameter-tuning",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Whether to run Optuna hyperparameter tuning.",
    )
    parser.add_argument(
        "--threshold-optimization",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Whether to tune decision threshold(s) post-training.",
    )
    parser.add_argument(
        "--threshold-type",
        type=str,
        choices=_THRESHOLD_CHOICES,
        default=argparse.SUPPRESS,
        help="Threshold mode (e.g., global vs per-label).",
    )
    parser.add_argument(
        "--scale-learning-rate",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Whether to scale learning rate based on batch size / device.",
    )
    parser.add_argument(
        "--wrap-peft",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Whether to apply PEFT (LoRA) wrapping.",
    )
    parser.add_argument(
        "--proxy-checkpoint",
        type=str,
        default=argparse.SUPPRESS,
        help="Optional proxy checkpoint. If unset assumes checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=argparse.SUPPRESS,
        help="Base pretrained model checkpoint identifier.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=argparse.SUPPRESS,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--best-model-metric",
        type=str,
        choices=_METRIC_CHOICES,
        default=argparse.SUPPRESS,
        help="Metric name used to select the best model checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Training batch size.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=argparse.SUPPRESS,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=argparse.SUPPRESS,
        help="Scheduler identifier/name.",
    )
    parser.add_argument(
        "--best-threshold-metric",
        type=str,
        choices=_METRIC_CHOICES,
        default=argparse.SUPPRESS,
        help="Metric name used to select optimal threshold(s).",
    )
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--optuna-space",
        type=_json_or_file,
        default=argparse.SUPPRESS,
        help=(
            "Optional partial override for the Optuna search space. Values are merged into the default space "
            "(base or PEFT, depending on `wrap_peft`)."
        ),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=argparse.SUPPRESS,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=argparse.SUPPRESS,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=argparse.SUPPRESS,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        choices=_LORA_BIAS_CHOICES,
        default=argparse.SUPPRESS,
        help="LoRA bias handling (validated downstream).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=argparse.SUPPRESS,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--use-cpu",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Force CPU execution.",
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    """Run the tlmtc CLI.

    Args:
        argv: Optional list of CLI arguments. If None, arguments are read from `sys.argv`.

    Returns:
        Process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    from tlmtc.run import run_tlmtc

    try:
        run_tlmtc(**vars(args))
    except Exception as e:
        parser.error(str(e))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
