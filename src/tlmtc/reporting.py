"""Reporting tables and figures for multi-label text classification evaluation."""

from collections.abc import Iterator
from contextlib import contextmanager
from html import escape
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from great_tables import GT, loc, md, style
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import MaxNLocator
from transformers import Trainer

REPORT_FONT_STACK = [
    "Ubuntu",
    "Segoe UI",
    "Roboto",
    "Oxygen",
    "Cantarell",
    "Helvetica Neue",
    "Arial",
    "DejaVu Sans",
]


@contextmanager
def report_font_context() -> Iterator[None]:
    """Temporarily apply the report font stack to Matplotlib figures."""
    with rc_context(
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": REPORT_FONT_STACK,
        }
    ):
        yield


def escape_report_text(
    value: object,
) -> str:
    """Escape text before Markdown/HTML report rendering."""
    return escape(str(value), quote=True)


def set_non_degenerate_xlim(ax: Axes, xmin: float, xmax: float) -> None:
    """Set robust x-axis limits for possibly single-point plots.

    Args:
        ax: Matplotlib axes to update.
        xmin: Requested lower x-axis limit.
        xmax: Requested upper x-axis limit.
    """
    if xmin == xmax:
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
    else:
        ax.set_xlim(xmin, xmax)


def make_global_metrics_table(
    eval_metrics: dict[str, float],
    target_name: str,
    label_names: list[str],
    checkpoint: str,
    train_data_path: Path,
    test_data_path: Path,
    input_mode: str,
) -> GT:
    """Create a formatted table of aggregate evaluation metrics.

    Args:
        eval_metrics: Aggregate evaluation metrics keyed by metric name.
        target_name: Display name for the classification target.
        label_names: Label names included in the evaluation.
        checkpoint: Model checkpoint identifier.
        train_data_path: Path to the prepared training split.
        test_data_path: Path to the prepared test split.
        input_mode: Human-readable text input mode.

    Returns:
        Renderable table object.
    """
    safe_target_name = escape_report_text(target_name)
    safe_input_mode = escape_report_text(input_mode)

    name_map = {
        "f1_micro": ("F1", "    micro"),
        "f1_macro": ("F1", "    macro"),
        "roc_auc_micro": ("ROC-AUC", "    micro"),
        "roc_auc_macro": ("ROC-AUC", "    macro"),
        "pr_auc_micro": ("PR-AUC", "    micro"),
        "pr_auc_macro": ("PR-AUC", "    macro"),
        "true_cardinality": ("Cardinality", "    true"),
        "pred_cardinality": ("Cardinality", "    pred"),
    }

    df = pd.DataFrame(
        [
            {
                "metric": metric,
                "metric_type": metric_type,
                "Value": eval_metrics[key],
            }
            for key, (metric, metric_type) in name_map.items()
        ]
    )

    model_name = checkpoint.rsplit("/", maxsplit=1)[-1]
    safe_model_name = escape_report_text(model_name)
    train_examples = len(pd.read_parquet(train_data_path))
    test_examples = len(pd.read_parquet(test_data_path))

    return (
        GT(df)
        .opt_table_font(font=REPORT_FONT_STACK)
        .tab_header(
            title=md(f"**Multi-label Classification<br>of {safe_target_name}**"),
            subtitle=md(f"*Performance metrics<br>for fine-tuned {safe_model_name}*"),
        )
        .tab_stub(rowname_col="metric_type", groupname_col="metric")
        .tab_stubhead(label="Metric")
        .fmt_number(columns="Value", decimals=2)
        .tab_source_note(
            source_note=md(
                f"*Input mode*: {safe_input_mode}<br>"
                f"*Labels*: {len(label_names)}<br>"
                f"*Train examples*: {train_examples}<br>"
                f"*Test examples*: {test_examples}"
            )
        )
        .tab_style(
            style=style.text(weight=250),
            locations=loc.row_groups(),
        )
        .tab_style(
            style=style.text(weight=500),
            locations=[loc.stubhead(), loc.column_header()],
        )
        .tab_style(
            style=style.text(whitespace="pre"),
            locations=loc.stub(),
        )
        .tab_options(stub_border_style="none")
    )


def make_label_metrics_table(
    eval_metrics: dict[str, dict[str, float]],
    target_name: str,
    checkpoint: str,
    train_data_path: Path,
    test_data_path: Path,
    input_mode: str,
) -> GT:
    """Create a formatted table of per-label evaluation metrics.

    Args:
        eval_metrics: Per-label evaluation metrics keyed by label name.
        target_name: Display name for the classification target.
        checkpoint: Model checkpoint identifier.
        train_data_path: Path to the prepared training split.
        test_data_path: Path to the prepared test split.
        input_mode: Human-readable text input mode.

    Returns:
        Renderable table object.
    """
    safe_target_name = escape_report_text(target_name)
    safe_input_mode = escape_report_text(input_mode)

    df = pd.DataFrame(eval_metrics).T.reset_index()
    df = df.rename(
        columns={
            "index": "Label",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
        }
    )

    model_name = checkpoint.rsplit("/", maxsplit=1)[-1]
    safe_model_name = escape_report_text(model_name)
    train_examples = len(pd.read_parquet(train_data_path))
    test_examples = len(pd.read_parquet(test_data_path))

    return (
        GT(df)
        .opt_table_font(font=REPORT_FONT_STACK)
        .tab_header(
            title=md(f"**Multi-label Classification of {safe_target_name}**"),
            subtitle=md(f"*Performance metrics for fine-tuned {safe_model_name}*"),
        )
        .cols_label(
            true_prevalence=md("True<br>prevalence"),
            pred_prevalence=md("Pred<br>prevalence"),
        )
        .fmt_number(
            columns=[
                "F1",
                "Precision",
                "Recall",
                "ROC-AUC",
                "PR-AUC",
                "true_prevalence",
                "pred_prevalence",
            ],
            decimals=2,
        )
        .tab_source_note(
            source_note=md(
                f"*Input mode*: {safe_input_mode}<br>"
                f"*Train examples*: {train_examples}<br>"
                f"*Test examples*: {test_examples}"
            )
        )
        .tab_style(
            style=style.text(weight=500),
            locations=loc.column_header(),
        )
    )


def make_hyperparameters_table(
    threshold: np.ndarray,
    trainer: Trainer,
    target_name: str,
    checkpoint: str,
    input_mode: str,
) -> GT:
    """Create a formatted table of selected hyperparameters and thresholds.

    Args:
        threshold: Global or per-label decision threshold values.
        trainer: Trainer instance containing the final training arguments.
        target_name: Display name for the classification target.
        checkpoint: Model checkpoint identifier.
        input_mode: Human-readable text input mode.

    Returns:
        Renderable table object.
    """
    safe_target_name = escape_report_text(target_name)
    safe_input_mode = escape_report_text(input_mode)

    if threshold.size == 1:
        threshold_name = "Global threshold"
        threshold_value = f"{threshold.item():.2f}"
    else:
        threshold_name = "Label-specific thresholds"
        threshold_value = ", ".join(f"{value:.2f}" for value in threshold)

    df = pd.DataFrame(
        {
            "Metric": [
                threshold_name,
                "Learning rate",
                "Batch size",
                "Weight decay",
                "Learning scheduler",
                "Epochs",
            ],
            "Value": [
                threshold_value,
                f"{trainer.args.learning_rate:.2e}",
                trainer.args.per_device_train_batch_size,
                f"{trainer.args.weight_decay:.3f}",
                trainer.args.lr_scheduler_type,
                trainer.args.num_train_epochs,
            ],
        }
    )

    model_name = checkpoint.rsplit("/", maxsplit=1)[-1]
    safe_model_name = escape_report_text(model_name)

    return (
        GT(df)
        .opt_table_font(font=REPORT_FONT_STACK)
        .tab_header(
            title=md(f"**Multi-label Classification<br>of {safe_target_name}**"),
            subtitle=md(f"*Hyperparameters<br>for fine-tuned {safe_model_name}*"),
        )
        .tab_stubhead(label="Metric")
        .tab_style(
            style=style.text(weight=250),
            locations=loc.row_groups(),
        )
        .tab_style(
            style=style.text(weight=500),
            locations=[loc.stubhead(), loc.column_header()],
        )
        .tab_style(style=style.text(whitespace="pre"), locations=loc.stub())
        .tab_source_note(source_note=md(f"*Input mode*: {safe_input_mode}"))
        .tab_options(stub_border_style="none")
    )


def make_roc_curves_plot(
    roc_curves: dict[str, dict[int | str, np.ndarray | float]],
    target_name: str,
    checkpoint: str,
    label_names: list[str],
) -> Figure:
    """Create a ROC-curve figure for aggregate and per-label performance.

    Args:
        roc_curves: ROC curve data returned by evaluation metrics.
        target_name: Display name for the classification target.
        checkpoint: Model checkpoint identifier.
        label_names: Label names in the same order as the label matrix columns.

    Returns:
        Matplotlib figure.
    """
    fpr = roc_curves["fpr"]
    tpr = roc_curves["tpr"]
    roc_auc = roc_curves["roc_auc"]

    num_labels = len(label_names)
    model_name = checkpoint.rsplit("/", maxsplit=1)[-1]

    with report_font_context():
        cmap = LinearSegmentedColormap.from_list(
            "roc_label_range",
            ["#3366CC", "#B39DDB"],
        )
        colors = [cmap(i / (num_labels - 1)) for i in range(num_labels)]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_aspect("equal", adjustable="box")

        ax.plot(fpr["micro"], tpr["micro"], label=f"Micro (AUC = {roc_auc['micro']:.2f})", color="#3366CC", linewidth=2)
        ax.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"Macro (AUC = {roc_auc['macro']:.2f})",
            color="#B39DDB",
            linewidth=2,
        )
        for i in range(num_labels):
            ax.plot(
                fpr[i],
                tpr[i],
                lw=1,
                alpha=0.25,
                color=colors[i],
                linestyle="--",
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.25)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(
            f"Multi-label Classification of {target_name}",
            fontsize=14,
            path_effects=[pe.withStroke(linewidth=0.25, foreground="black")],
            loc="center",
            pad=20,
        )
        ax.text(
            0.5,
            1.01,
            f"Discriminative performance for fine-tuned {model_name}",
            fontsize=11,
            style="italic",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.legend(loc="lower right", fontsize="small")
        fig.tight_layout()
        return fig


def make_cooccurrence_heatmaps_plot(
    co_occurrence: dict[str, np.ndarray], target_name: str, checkpoint: str, label_names: list[str]
) -> Figure:
    """Create heatmaps of observed and predicted label co-occurrence.

    Args:
        co_occurrence: Absolute and normalized co-occurrence matrices for true and predicted labels.
        target_name: Display name for the classification target.
        checkpoint: Model checkpoint identifier.
        label_names: Label names in the same order as the label matrix columns.

    Returns:
        Matplotlib figure.
    """
    model_name = checkpoint.rsplit("/", maxsplit=1)[-1]

    with report_font_context():
        cmap = LinearSegmentedColormap.from_list(
            "cooccurrence_range",
            ["#FFFFFF", "#3366CC"],
        )
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5),
            layout=ConstrainedLayoutEngine(h_pad=0.5),
        )

        sns.heatmap(
            co_occurrence["co_true_rel"],
            ax=axes[0],
            cmap=cmap,
            square=True,
            xticklabels=label_names,
            yticklabels=label_names,
            cbar=False,
            vmin=0,
            vmax=1,
            linewidths=0.3,
            annot=co_occurrence["co_true_abs"].astype(int),
            fmt="d",
        )
        axes[0].set_title("True label co-occurrence")

        sns.heatmap(
            co_occurrence["co_pred_rel"],
            ax=axes[1],
            cmap=cmap,
            square=True,
            xticklabels=label_names,
            yticklabels=label_names,
            cbar=True,
            vmin=0,
            vmax=1,
            linewidths=0.3,
            annot=co_occurrence["co_pred_abs"].astype(int),
            fmt="d",
        )
        axes[1].set_title("Predicted label co-occurrence")

        fig.suptitle(
            f"Multi-label Classification of {target_name}",
            fontsize=14,
            path_effects=[pe.withStroke(linewidth=0.25, foreground="black")],
            ha="center",
            y=0.99,
        )
        fig.text(
            0.5,
            0.93,
            f"Structural alignment for fine-tuned {model_name}",
            fontsize=11,
            style="italic",
            ha="center",
            va="center",
        )
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        return fig


def make_loss_curves_plot(
    losses: pd.DataFrame,
    target_name: str,
    checkpoint: str,
    best_epoch: int,
) -> Figure:
    """Create a loss-curve figure from training and evaluation logs.

    Args:
        losses: DataFrame with epoch-level training and evaluation losses.
        target_name: Display name for the classification target.
        checkpoint: Model checkpoint identifier.
        best_epoch: Epoch selected by the configured model-selection metric.

    Returns:
        Matplotlib figure.
    """
    model_name = checkpoint.rsplit("/", maxsplit=1)[-1]

    with report_font_context():
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            losses["epoch"],
            losses["train_loss"],
            label="Train Loss",
            linewidth=2,
            color="#3366CC",
        )
        ax.plot(
            losses["epoch"],
            losses["eval_loss"],
            label="Eval Loss",
            linewidth=2,
            color="#B39DDB",
        )
        ax.axvline(best_epoch, color="#E57383", linestyle="--", linewidth=1.5, alpha=0.6, label="Best Model")

        set_non_degenerate_xlim(
            ax=ax,
            xmin=float(losses["epoch"].min()),
            xmax=float(losses["epoch"].max()),
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ticks = [tick for tick in ax.get_xticks() if 1 < tick < losses["epoch"].max()]
        ax.set_xticks(ticks)

        ax.set_title(
            f"Multi-label Classification of {target_name}",
            fontsize=14,
            path_effects=[pe.withStroke(linewidth=0.25, foreground="black")],
            loc="center",
            pad=20,
        )
        ax.text(
            0.5,
            1.01,
            f"Training dynamics for fine-tuned {model_name}",
            fontsize=11,
            style="italic",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=11)
        fig.tight_layout()
        return fig


def make_objective_values_plot(
    objective_values: pd.DataFrame,
    target_name: str,
    checkpoint: str,
) -> Figure:
    """Create an Optuna objective-values figure across tuning trials.

    Args:
        objective_values: DataFrame with Optuna trial numbers and objective values.
        target_name: Display name for the classification target.
        checkpoint: Proxy model checkpoint identifier.

    Returns:
        Matplotlib figure.
    """
    values = (
        objective_values.sort_values("number")
        .assign(number=lambda df: df["number"] + 1)
        .assign(best_so_far=lambda df: df["value"].cummax())
    )
    model_name = checkpoint.rsplit("/", maxsplit=1)[-1]

    with report_font_context():
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            values["number"],
            values["value"],
            marker="o",
            markersize=6,
            linewidth=2,
            markerfacecolor="#3366CC",
            markeredgecolor="#3366CC",
            color="#B39DDB",
            label="Objective Value",
        )
        ax.plot(
            values["number"],
            values["best_so_far"],
            color="#E57383",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label="Best so far",
        )

        set_non_degenerate_xlim(
            ax=ax,
            xmin=float(values["number"].min()),
            xmax=float(values["number"].max()),
        )
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ticks = [tick for tick in ax.get_xticks() if 1 < tick < values["number"].max()]
        ax.set_xticks(ticks)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_title(
            f"Multi-label Classification of {target_name}",
            fontsize=14,
            path_effects=[pe.withStroke(linewidth=0.25, foreground="black")],
            loc="center",
            pad=20,
        )
        ax.text(
            0.5,
            1.01,
            f"Hyperparameter optimization for fine-tuned {model_name}",
            fontsize=11,
            style="italic",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

        ax.set_xlabel("Trial Number", fontsize=12)
        ax.set_ylabel("Objective Value", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        return fig
