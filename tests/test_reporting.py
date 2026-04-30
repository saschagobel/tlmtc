"""Tests for reporting helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from great_tables import GT
from matplotlib.figure import Figure
from transformers import TrainingArguments

from tlmtc.reporting import (
    make_cooccurrence_heatmaps_plot,
    make_global_metrics_table,
    make_hyperparameters_table,
    make_label_metrics_table,
    make_roc_curves_plot,
)


@pytest.fixture
def reporting_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal train and test parquet files for reporting tests."""
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"

    pd.DataFrame(
        {
            "text": ["a", "b"],
            "label_a": [1, 0],
            "label_b": [0, 1],
        }
    ).to_parquet(train_path, index=False)
    pd.DataFrame(
        {
            "text": ["c"],
            "label_a": [1],
            "label_b": [1],
        }
    ).to_parquet(test_path, index=False)

    return train_path, test_path


@pytest.fixture
def global_eval_metrics() -> dict[str, float]:
    """Provide global evaluation metrics matching the evaluation contract."""
    return {
        "f1_micro": 0.8,
        "f1_macro": 0.7,
        "roc_auc_micro": 0.9,
        "roc_auc_macro": 0.8,
        "pr_auc_micro": 0.7,
        "pr_auc_macro": 0.6,
        "true_cardinality": 1.0,
        "pred_cardinality": 1.1,
    }


@pytest.fixture
def label_eval_metrics() -> dict[str, dict[str, float]]:
    """Provide label-level evaluation metrics matching the evaluation contract."""
    return {
        "label_a": {
            "f1": 0.8,
            "precision": 0.75,
            "recall": 0.86,
            "roc_auc": 0.91,
            "pr_auc": 0.88,
            "true_support": 0.4,
            "pred_support": 0.46,
        },
        "label_b": {
            "f1": 0.7,
            "precision": 0.67,
            "recall": 0.74,
            "roc_auc": 0.84,
            "pr_auc": 0.79,
            "true_support": 0.6,
            "pred_support": 0.55,
        },
    }


@pytest.fixture
def roc_curves() -> dict[str, dict[int | str, np.ndarray | float]]:
    """Provide ROC curve payload matching the reporting contract."""
    return {
        "fpr": {
            0: np.array([0.0, 0.2, 1.0]),
            1: np.array([0.0, 0.1, 1.0]),
            "micro": np.array([0.0, 0.15, 1.0]),
            "macro": np.array([0.0, 0.18, 1.0]),
        },
        "tpr": {
            0: np.array([0.0, 0.8, 1.0]),
            1: np.array([0.0, 0.9, 1.0]),
            "micro": np.array([0.0, 0.85, 1.0]),
            "macro": np.array([0.0, 0.84, 1.0]),
        },
        "roc_auc": {
            0: 0.82,
            1: 0.91,
            "micro": 0.87,
            "macro": 0.86,
        },
    }


@pytest.fixture
def co_occurrence() -> dict[str, np.ndarray]:
    """Provide co-occurrence matrices matching the reporting contract."""
    return {
        "co_true_abs": np.array([[3, 1], [1, 2]]),
        "co_true_rel": np.array([[1.0, 0.41], [0.41, 1.0]]),
        "co_pred_abs": np.array([[2, 1], [1, 3]]),
        "co_pred_rel": np.array([[1.0, 0.33], [0.33, 1.0]]),
    }


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Automatically close all figures after every test."""
    yield
    plt.close("all")


@pytest.fixture
def trainer_with_args(tmp_path: Path) -> object:
    """Provide a minimal Trainer-like object with training arguments."""
    args = TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        num_train_epochs=5,
        report_to="none",
    )

    return type("TrainerStub", (), {"args": args})()


class TestMakeGlobalMetricsTable:
    """Tests for global metrics table rendering."""

    def test_returns_gt_table(
        self,
        reporting_paths: tuple[Path, Path],
        global_eval_metrics: dict[str, float],
    ) -> None:
        """Global metrics reporting returns a Great Tables object."""
        train_path, test_path = reporting_paths

        table = make_global_metrics_table(
            eval_metrics=global_eval_metrics,
            target_name="Policy",
            label_names=["label_a", "label_b"],
            checkpoint="user/model-v1",
            train_data_path=train_path,
            test_data_path=test_path,
        )

        assert isinstance(table, GT)

    def test_builds_expected_metric_rows(
        self,
        reporting_paths: tuple[Path, Path],
        global_eval_metrics: dict[str, float],
    ) -> None:
        """Global metrics table uses the expected metric labels, order, and values."""
        train_path, test_path = reporting_paths

        table = make_global_metrics_table(
            eval_metrics=global_eval_metrics,
            target_name="Policy",
            label_names=["label_a", "label_b"],
            checkpoint="user/model-v1",
            train_data_path=train_path,
            test_data_path=test_path,
        )

        result = table._tbl_data

        assert result.to_dict("records") == [
            {"metric": "F1", "metric_type": "    micro", "Value": 0.8},
            {"metric": "F1", "metric_type": "    macro", "Value": 0.7},
            {"metric": "ROC-AUC", "metric_type": "    micro", "Value": 0.9},
            {"metric": "ROC-AUC", "metric_type": "    macro", "Value": 0.8},
            {"metric": "PR-AUC", "metric_type": "    micro", "Value": 0.7},
            {"metric": "PR-AUC", "metric_type": "    macro", "Value": 0.6},
            {"metric": "Cardinality", "metric_type": "    true", "Value": 1.0},
            {"metric": "Cardinality", "metric_type": "    pred", "Value": 1.1},
        ]

    def test_renders_metadata(
        self,
        reporting_paths: tuple[Path, Path],
        global_eval_metrics: dict[str, float],
    ) -> None:
        """Global metrics table renders model and dataset metadata."""
        train_path, test_path = reporting_paths

        table = make_global_metrics_table(
            eval_metrics=global_eval_metrics,
            target_name="Policy",
            label_names=["label_a", "label_b"],
            checkpoint="user/model-v1",
            train_data_path=train_path,
            test_data_path=test_path,
        )

        html = table.as_raw_html()

        assert "Multi-label Classification" in html
        assert "Policy" in html
        assert "model-v1" in html
        assert "Labels" in html
        assert "Train examples" in html
        assert "Test examples" in html
        assert "2" in html
        assert "1" in html

    def test_missing_metric_raises_key_error(
        self,
        reporting_paths: tuple[Path, Path],
    ) -> None:
        """Global metrics table requires the complete global metrics contract."""
        train_path, test_path = reporting_paths

        with pytest.raises(KeyError):
            make_global_metrics_table(
                eval_metrics={},
                target_name="Policy",
                label_names=["label_a", "label_b"],
                checkpoint="user/model-v1",
                train_data_path=train_path,
                test_data_path=test_path,
            )


class TestMakeLabelMetricsTable:
    """Tests for label metrics table rendering."""

    def test_returns_gt_table(
        self,
        reporting_paths: tuple[Path, Path],
        label_eval_metrics: dict[str, dict[str, float]],
    ) -> None:
        """Label metrics reporting returns a Great Tables object."""
        train_path, test_path = reporting_paths

        table = make_label_metrics_table(
            eval_metrics=label_eval_metrics,
            target_name="Policy",
            checkpoint="user/model-v1",
            train_data_path=train_path,
            test_data_path=test_path,
        )

        assert isinstance(table, GT)

    def test_builds_expected_label_metric_rows(
        self,
        reporting_paths: tuple[Path, Path],
        label_eval_metrics: dict[str, dict[str, float]],
    ) -> None:
        """Label metrics table uses the expected labels, metric columns, and values."""
        train_path, test_path = reporting_paths

        table = make_label_metrics_table(
            eval_metrics=label_eval_metrics,
            target_name="Policy",
            checkpoint="user/model-v1",
            train_data_path=train_path,
            test_data_path=test_path,
        )

        result = table._tbl_data

        assert result.to_dict("records") == [
            {
                "Label": "label_a",
                "F1": 0.8,
                "Precision": 0.75,
                "Recall": 0.86,
                "ROC-AUC": 0.91,
                "PR-AUC": 0.88,
                "true_support": 0.4,
                "pred_support": 0.46,
            },
            {
                "Label": "label_b",
                "F1": 0.7,
                "Precision": 0.67,
                "Recall": 0.74,
                "ROC-AUC": 0.84,
                "PR-AUC": 0.79,
                "true_support": 0.6,
                "pred_support": 0.55,
            },
        ]

    def test_renders_metadata(
        self,
        reporting_paths: tuple[Path, Path],
        label_eval_metrics: dict[str, dict[str, float]],
    ) -> None:
        """Label metrics table renders model and dataset metadata."""
        train_path, test_path = reporting_paths

        table = make_label_metrics_table(
            eval_metrics=label_eval_metrics,
            target_name="Policy",
            checkpoint="user/model-v1",
            train_data_path=train_path,
            test_data_path=test_path,
        )

        html = table.as_raw_html()

        assert "Multi-label Classification" in html
        assert "Policy" in html
        assert "model-v1" in html
        assert "Train examples" in html
        assert "Test examples" in html
        assert "2" in html
        assert "1" in html


class TestMakeHyperparametersTable:
    """Tests for hyperparameter table rendering."""

    def test_returns_gt_table(
        self,
        trainer_with_args: object,
    ) -> None:
        """Hyperparameter reporting returns a Great Tables object."""
        table = make_hyperparameters_table(
            threshold=np.array([0.42]),
            trainer=trainer_with_args,
            target_name="Policy",
            checkpoint="user/model-v1",
        )

        assert isinstance(table, GT)

    def test_builds_expected_global_threshold_rows(
        self,
        trainer_with_args: object,
    ) -> None:
        """Hyperparameter table renders a single threshold as a global threshold."""
        table = make_hyperparameters_table(
            threshold=np.array([0.42]),
            trainer=trainer_with_args,
            target_name="Policy",
            checkpoint="user/model-v1",
        )

        result = table._tbl_data

        assert result.to_dict("records") == [
            {"Metric": "Global threshold", "Value": "0.42"},
            {"Metric": "Learning rate", "Value": "2.00e-05"},
            {"Metric": "Batch size", "Value": 16},
            {"Metric": "Weight decay", "Value": "0.010"},
            {"Metric": "Learning scheduler", "Value": "linear"},
            {"Metric": "Epochs", "Value": 5},
        ]

    def test_builds_expected_label_threshold_rows(
        self,
        trainer_with_args: object,
    ) -> None:
        """Hyperparameter table renders multiple thresholds as label-specific thresholds."""
        table = make_hyperparameters_table(
            threshold=np.array([0.31, 0.47, 0.62]),
            trainer=trainer_with_args,
            target_name="Policy",
            checkpoint="user/model-v1",
        )

        result = table._tbl_data

        assert result.to_dict("records")[0] == {
            "Metric": "Label-specific thresholds",
            "Value": "0.31, 0.47, 0.62",
        }

    def test_renders_header_metadata(
        self,
        trainer_with_args: object,
    ) -> None:
        """Hyperparameter table renders target and model metadata."""
        table = make_hyperparameters_table(
            threshold=np.array([0.42]),
            trainer=trainer_with_args,
            target_name="Policy",
            checkpoint="user/model-v1",
        )

        html = table.as_raw_html()

        assert "Multi-label Classification" in html
        assert "Policy" in html
        assert "model-v1" in html
        assert "Hyperparameters" in html


class TestMakeRocCurvesPlot:
    """Tests for ROC curve plot rendering."""

    def test_returns_figure(
        self,
        roc_curves: dict[str, dict[int | str, np.ndarray | float]],
    ) -> None:
        """ROC curve reporting returns a Matplotlib figure."""
        fig = make_roc_curves_plot(
            roc_curves=roc_curves,
            target_name="Policy",
            checkpoint="user/model-v1",
            label_names=["label_a", "label_b"],
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    @pytest.mark.parametrize(
        ("target_name", "checkpoint", "expected_model_name"),
        [
            ("Policy", "user/model-v1", "model-v1"),
            ("Risk", "org/nested-model", "nested-model"),
        ],
    )
    def test_renders_expected_axes_metadata_and_legend(
        self,
        roc_curves: dict[str, dict[int | str, np.ndarray | float]],
        target_name: str,
        checkpoint: str,
        expected_model_name: str,
    ) -> None:
        """ROC curve plot renders titles, labels, subtitle, and summary legend entries."""
        fig = make_roc_curves_plot(
            roc_curves=roc_curves,
            target_name=target_name,
            checkpoint=checkpoint,
            label_names=["label_a", "label_b"],
        )

        ax = fig.axes[0]
        legend = ax.get_legend()

        assert ax.get_title() == f"Multi-label Classification of {target_name}"
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() == "True Positive Rate"
        assert ax.texts[0].get_text() == f"Discriminative performance for fine-tuned {expected_model_name}"
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "Micro (AUC = 0.87)",
            "Macro (AUC = 0.86)",
        ]

    def test_plots_micro_macro_label_and_baseline_lines(
        self,
        roc_curves: dict[str, dict[int | str, np.ndarray | float]],
    ) -> None:
        """ROC curve plot contains one line per label plus micro, macro, and diagonal baseline."""
        label_names = ["label_a", "label_b"]

        fig = make_roc_curves_plot(
            roc_curves=roc_curves,
            target_name="Policy",
            checkpoint="user/model-v1",
            label_names=label_names,
        )

        ax = fig.axes[0]
        assert len(ax.lines) == len(label_names) + 3

    def test_missing_curve_key_raises_key_error(self) -> None:
        """ROC curve plot requires the complete ROC curve contract."""
        with pytest.raises(KeyError):
            make_roc_curves_plot(
                roc_curves={},
                target_name="Policy",
                checkpoint="user/model-v1",
                label_names=["label_a", "label_b"],
            )


class TestMakeCooccurrenceHeatmapsPlot:
    """Tests for co-occurrence heatmap rendering."""

    def test_returns_figure(
        self,
        co_occurrence: dict[str, np.ndarray],
    ) -> None:
        """Co-occurrence heatmap reporting returns a Matplotlib figure."""
        fig = make_cooccurrence_heatmaps_plot(
            co_occurrence=co_occurrence,
            target_name="Policy",
            checkpoint="user/model-v1",
            label_names=["label_a", "label_b"],
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3

    @pytest.mark.parametrize(
        ("target_name", "checkpoint", "expected_model_name"),
        [
            ("Policy", "user/model-v1", "model-v1"),
            ("Risk", "org/nested-model", "nested-model"),
        ],
    )
    def test_renders_expected_titles_and_metadata(
        self,
        co_occurrence: dict[str, np.ndarray],
        target_name: str,
        checkpoint: str,
        expected_model_name: str,
    ) -> None:
        """Co-occurrence heatmaps render panel titles and figure metadata."""
        fig = make_cooccurrence_heatmaps_plot(
            co_occurrence=co_occurrence,
            target_name=target_name,
            checkpoint=checkpoint,
            label_names=["label_a", "label_b"],
        )

        ax_true, ax_pred = fig.axes[:2]
        figure_texts = [text.get_text() for text in fig.texts]

        assert fig.get_suptitle() == f"Multi-label Classification of {target_name}"
        assert f"Structural alignment for fine-tuned {expected_model_name}" in figure_texts
        assert ax_true.get_title() == "True label co-occurrence"
        assert ax_pred.get_title() == "Predicted label co-occurrence"
        assert [tick.get_text() for tick in ax_true.get_xticklabels()] == ["label_a", "label_b"]
        assert [tick.get_text() for tick in ax_true.get_yticklabels()] == ["label_a", "label_b"]

    def test_annotates_all_cells(
        self,
        co_occurrence: dict[str, np.ndarray],
    ) -> None:
        """Co-occurrence heatmaps annotate every matrix cell with absolute counts."""
        fig = make_cooccurrence_heatmaps_plot(
            co_occurrence=co_occurrence,
            target_name="Policy",
            checkpoint="user/model-v1",
            label_names=["label_a", "label_b"],
        )

        ax_true, ax_pred = fig.axes[:2]

        assert [text.get_text() for text in ax_true.texts] == ["3", "1", "1", "2"]
        assert [text.get_text() for text in ax_pred.texts] == ["2", "1", "1", "3"]

    def test_missing_matrix_key_raises_key_error(self) -> None:
        """Co-occurrence heatmap plot requires the complete co-occurrence contract."""
        with pytest.raises(KeyError):
            make_cooccurrence_heatmaps_plot(
                co_occurrence={},
                target_name="Policy",
                checkpoint="user/model-v1",
                label_names=["label_a", "label_b"],
            )
