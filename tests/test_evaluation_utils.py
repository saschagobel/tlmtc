"""Tests for fine-tuning utility functions."""

import numpy as np
import pytest

from tlmtc.utils import _get_global_eval_metrics, _get_label_eval_metrics, _round_metric_dict


@pytest.fixture
def perfect_multilabel_case():
    """Provide a perfectly predicted two-label multilabel case with well-separated probabilities."""
    y_true = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    y_pred = y_true.copy()
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.7],
        ]
    )
    return y_true, y_pred, y_prob


@pytest.fixture
def imperfect_multilabel_case():
    """Provide an imperfectly predicted two-label multilabel case with non-trivial probability scores."""
    y_true = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
        ]
    )
    y_pred = np.array(
        [
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 0],
        ]
    )
    y_prob = np.array(
        [
            [0.9, 0.2],
            [0.6, 0.4],
            [0.8, 0.7],
            [0.2, 0.1],
        ]
    )
    return y_true, y_pred, y_prob


@pytest.fixture
def label_names_two():
    """Return label names for two-label test cases."""
    return ["label_0", "label_1"]


@pytest.fixture
def raw_metric_dict():
    """Provide an unrounded metrics dictionary matching the evaluation output schema."""
    return {
        "f1_micro": 0.1234,
        "f1_macro": 0.2345,
        "roc_auc_micro": 0.9876,
        "roc_auc_macro": 0.8765,
        "pr_auc_micro": 0.4567,
        "pr_auc_macro": 0.5678,
        "true_cardinality": 1.3333,
        "pred_cardinality": 1.0000,
    }


class TestMetricsExtractionUtils:
    """Test suite for metric extraction utility functions."""

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_global_eval_metrics_returns_complete_metric_dictionary(self, case_fixture, request):
        """Ensure _get_global_eval_metrics returns the full documented metric set."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)

        metrics = _get_global_eval_metrics(y_true, y_pred, y_prob)

        assert set(metrics) == {
            "f1_micro",
            "f1_macro",
            "roc_auc_micro",
            "roc_auc_macro",
            "pr_auc_micro",
            "pr_auc_macro",
            "true_cardinality",
            "pred_cardinality",
        }

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_global_eval_metrics_returns_plain_floats(self, case_fixture, request):
        """Ensure _get_global_eval_metrics returns JSON-serializable float values."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)

        metrics = _get_global_eval_metrics(y_true, y_pred, y_prob)

        assert all(isinstance(v, float) for v in metrics.values())

    def test_get_global_eval_metrics_returns_expected_values_for_perfect_predictions(self, perfect_multilabel_case):
        """Ensure _get_global_eval_metrics yields perfect scores for perfect predictions."""
        y_true, y_pred, y_prob = perfect_multilabel_case

        metrics = _get_global_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

        for k in [
            "f1_micro",
            "f1_macro",
            "roc_auc_micro",
            "roc_auc_macro",
            "pr_auc_micro",
            "pr_auc_macro",
        ]:
            assert metrics[k] == pytest.approx(1.0)

    def test_get_global_eval_metrics_returns_expected_cardinalities_for_perfect_predictions(
        self, perfect_multilabel_case
    ):
        """Ensure _get_global_eval_metrics returns correct average label cardinalities for perfect predictions."""
        y_true, y_pred, y_prob = perfect_multilabel_case

        metrics = _get_global_eval_metrics(y_true, y_pred, y_prob)

        assert metrics["true_cardinality"] == pytest.approx(4 / 3)
        assert metrics["pred_cardinality"] == pytest.approx(4 / 3)

    def test_get_global_eval_metrics_returns_non_degenerate_auc_scores_for_imperfect_predictions(
        self, imperfect_multilabel_case
    ):
        """Ensure _get_global_eval_metrics computes probability-driven AUC values for imperfect predictions."""
        y_true, y_pred, y_prob = imperfect_multilabel_case

        metrics = _get_global_eval_metrics(y_true, y_pred, y_prob)

        assert 0.0 < metrics["roc_auc_micro"] < 1.0
        assert 0.0 < metrics["pr_auc_micro"] < 1.0

    def test_get_global_eval_metrics_returns_non_trivial_f1_scores_for_imperfect_predictions(
        self, imperfect_multilabel_case
    ):
        """Ensure _get_global_eval_metrics returns non-trivial F1 values for imperfect predictions."""
        y_true, y_pred, y_prob = imperfect_multilabel_case

        metrics = _get_global_eval_metrics(y_true, y_pred, y_prob)

        assert 0.0 < metrics["f1_micro"] < 1.0
        assert 0.0 < metrics["f1_macro"] < 1.0

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_global_eval_metrics_does_not_modify_inputs_in_place(self, case_fixture, request):
        """Ensure _get_global_eval_metrics does not modify input arrays in place."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)
        y_true0, y_pred0, y_prob0 = y_true.copy(), y_pred.copy(), y_prob.copy()

        _get_global_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

        assert np.array_equal(y_true, y_true0)
        assert np.array_equal(y_pred, y_pred0)
        assert np.array_equal(y_prob, y_prob0)

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_label_eval_metrics_returns_complete_metric_dictionary(self, case_fixture, request, label_names_two):
        """Ensure _get_label_eval_metrics returns one metrics dict per label with the full documented key set."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)

        metrics = _get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        assert list(metrics.keys()) == label_names_two
        for name in label_names_two:
            assert set(metrics[name]) == {
                "f1",
                "precision",
                "recall",
                "roc_auc",
                "pr_auc",
                "true_support",
                "pred_support",
            }

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_label_eval_metrics_returns_plain_floats(self, case_fixture, request, label_names_two):
        """Ensure _get_label_eval_metrics returns JSON-serializable float values for all labels and metrics."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)

        metrics = _get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        assert all(isinstance(v, float) for per_label in metrics.values() for v in per_label.values())

    def test_get_label_eval_metrics_returns_expected_values_for_perfect_predictions(
        self, perfect_multilabel_case, label_names_two
    ):
        """Ensure _get_label_eval_metrics returns perfect per-label scores for perfect predictions."""
        y_true, y_pred, y_prob = perfect_multilabel_case

        metrics = _get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        for name in label_names_two:
            for k in ["f1", "precision", "recall", "roc_auc", "pr_auc"]:
                assert metrics[name][k] == pytest.approx(1.0)

    def test_get_label_eval_metrics_returns_expected_supports_for_perfect_predictions(
        self, perfect_multilabel_case, label_names_two
    ):
        """Ensure _get_label_eval_metrics returns correct per-label true and predicted supports."""
        y_true, y_pred, y_prob = perfect_multilabel_case

        metrics = _get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        for i, name in enumerate(label_names_two):
            assert metrics[name]["true_support"] == pytest.approx(float(y_true[:, i].mean()))
            assert metrics[name]["pred_support"] == pytest.approx(float(y_pred[:, i].mean()))

    def test_get_label_eval_metrics_is_label_specific_for_imperfect_predictions(
        self, imperfect_multilabel_case, label_names_two
    ):
        """Ensure _get_label_eval_metrics reflects per-label errors rather than collapsing to global behavior."""
        y_true, y_pred, y_prob = imperfect_multilabel_case

        metrics = _get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        assert metrics["label_0"]["f1"] != metrics["label_1"]["f1"]
        assert metrics["label_0"]["f1"] < 1.0 or metrics["label_1"]["f1"] < 1.0

    def test_get_label_eval_metrics_auc_metrics_depend_on_probabilities_not_predictions(
        self, imperfect_multilabel_case, label_names_two
    ):
        """Ensure _get_label_eval_metrics AUC metrics change when probabilities change while labels stay fixed."""
        y_true, y_pred, y_prob = imperfect_multilabel_case

        metrics_good = _get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        y_prob_bad = y_prob.copy()
        y_prob_bad[:, 0] = 1.0 - y_prob_bad[:, 0]

        metrics_bad = _get_label_eval_metrics(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob_bad, label_names=label_names_two
        )

        assert metrics_good["label_0"]["roc_auc"] != metrics_bad["label_0"]["roc_auc"]
        assert metrics_good["label_0"]["pr_auc"] != metrics_bad["label_0"]["pr_auc"]

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_label_eval_metrics_does_not_modify_inputs_in_place(self, case_fixture, request, label_names_two):
        """Ensure _get_label_eval_metrics does not modify input arrays in place."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)
        y_true0, y_pred0, y_prob0 = y_true.copy(), y_pred.copy(), y_prob.copy()

        _get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        assert np.array_equal(y_true, y_true0)
        assert np.array_equal(y_pred, y_pred0)
        assert np.array_equal(y_prob, y_prob0)


class TestMetricRoundingUtil:
    """Test suite for metric rounding utility function."""

    def test_round_metric_dict_returns_new_dictionary_with_same_keys(self, raw_metric_dict):
        """Ensure _round_metric_dict returns a new dict preserving the original keys."""
        metrics_in = raw_metric_dict
        metrics_out = _round_metric_dict(metrics_in)

        assert metrics_out is not metrics_in
        assert set(metrics_out.keys()) == set(metrics_in.keys())

    def test_round_metric_dict_does_not_modify_input_dictionary_in_place(self, raw_metric_dict):
        """Ensure _round_metric_dict does not modify the input dictionary in place."""
        metrics_in = raw_metric_dict
        snapshot = dict(metrics_in)

        _round_metric_dict(metrics_in)

        assert metrics_in == snapshot

    @pytest.mark.parametrize("ndigits", [3, 1, 0])
    def test_round_metric_dict_respects_ndigits_parameter(self, raw_metric_dict, ndigits):
        """Ensure _round_metric_dict rounds all values using the provided ndigits value."""
        out = _round_metric_dict(raw_metric_dict, ndigits=ndigits)
        assert out == {k: round(v, ndigits) for k, v in raw_metric_dict.items()}
