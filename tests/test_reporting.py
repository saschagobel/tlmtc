"""Tests for reporting helpers."""

from pathlib import Path

import pandas as pd
import pytest
from great_tables import GT

from tlmtc.reporting import make_global_metrics_table, make_label_metrics_table


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
