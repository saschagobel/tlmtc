"""..."""

from pathlib import Path

import pandas as pd
from great_tables import GT, loc, md, style


def make_global_metrics_table(
    eval_metrics: dict[str, float],
    target_name: str,
    label_names: list[str],
    checkpoint: str,
    train_data_path: Path,
    test_data_path: Path,
) -> GT:
    """Create a formatted summary table of global evaluation metrics."""
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
    train_examples = len(pd.read_parquet(train_data_path))
    test_examples = len(pd.read_parquet(test_data_path))

    return (
        GT(df)
        .tab_header(
            title=md(f"**Multi-label Classification<br>of {target_name}**"),
            subtitle=md(f"*Performance metrics<br>for fine-tuned {model_name}*"),
        )
        .tab_stub(rowname_col="metric_type", groupname_col="metric")
        .tab_stubhead(label="Metric")
        .fmt_number(columns="Value", decimals=2)
        .tab_source_note(
            source_note=md(
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
) -> GT:
    """Create a formatted summary table of label-specific evaluation metrics."""
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
    train_examples = len(pd.read_parquet(train_data_path))
    test_examples = len(pd.read_parquet(test_data_path))

    return (
        GT(df)
        .tab_header(
            title=md(f"**Multi-label Classification of {target_name}**"),
            subtitle=md(f"*Performance metrics for fine-tuned {model_name}*"),
        )
        .cols_label(
            true_support=md("True<br>support"),
            pred_support=md("Pred<br>support"),
        )
        .fmt_number(
            columns=[
                "F1",
                "Precision",
                "Recall",
                "ROC-AUC",
                "PR-AUC",
                "true_support",
                "pred_support",
            ],
            decimals=2,
        )
        .tab_source_note(source_note=md(f"*Train examples*: {train_examples}<br>*Test examples*: {test_examples}"))
        .tab_style(
            style=style.text(weight=500),
            locations=loc.column_header(),
        )
    )
