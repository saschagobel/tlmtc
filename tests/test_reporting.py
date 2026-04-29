"""Tests for reporting helpers."""

from pathlib import Path

import pandas as pd
import pytest
from great_tables import GT

from tlmtc.reporting import make_global_metrics_table


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
