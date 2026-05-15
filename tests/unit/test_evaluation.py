"""Tests for evaluation utility functions."""

import numpy as np
import pandas as pd
import pytest

from tlmtc.evaluation import (
    find_optimal_threshold,
    get_best_epoch,
    get_co_occurrence,
    get_global_eval_metrics,
    get_label_eval_metrics,
    get_losses,
    get_pr_curves,
    get_roc_curves,
    round_metric_dict,
)


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


@pytest.fixture
def log_history_with_losses():
    """Provide a minimal Trainer.log_history with train/eval loss entries plus a monitored eval metric."""
    return [
        {"epoch": 1.0, "loss": 0.9},
        {"epoch": 1.0, "eval_loss": 0.8},
        {"epoch": 1.0, "eval_f1": 0.50},
        {"epoch": 2.0, "loss": 0.7},
        {"epoch": 2.0, "learning_rate": 1e-5},
        {"epoch": 2.0, "eval_f1": 0.60},
        {"epoch": 3.0, "loss": 0.6},
        {"epoch": 3.0, "eval_loss": 0.5},
        {"epoch": 3.0, "eval_f1": 0.55},
        {"epoch": 4.0, "eval_loss": 0.4},
    ]


def _is_non_decreasing(x: np.ndarray) -> bool:
    """Return True if a 1D array is non-decreasing."""
    return bool(np.all(np.diff(x) >= -1e-12))


def _is_non_increasing(x: np.ndarray) -> bool:
    """Return True if a 1D array is non-increasing."""
    return bool(np.all(np.diff(x) <= 1e-12))


class TestMetricsExtractionUtils:
    """Test suite for metric extraction utility functions."""

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_global_eval_metrics_returns_complete_metric_dictionary(self, case_fixture, request):
        """Ensure _get_global_eval_metrics returns the full documented metric set."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)

        metrics = get_global_eval_metrics(y_true, y_pred, y_prob)

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

        metrics = get_global_eval_metrics(y_true, y_pred, y_prob)

        assert all(isinstance(v, float) for v in metrics.values())

    def test_get_global_eval_metrics_returns_expected_values_for_perfect_predictions(self, perfect_multilabel_case):
        """Ensure _get_global_eval_metrics yields perfect scores for perfect predictions."""
        y_true, y_pred, y_prob = perfect_multilabel_case

        metrics = get_global_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

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

        metrics = get_global_eval_metrics(y_true, y_pred, y_prob)

        assert metrics["true_cardinality"] == pytest.approx(4 / 3)
        assert metrics["pred_cardinality"] == pytest.approx(4 / 3)

    def test_get_global_eval_metrics_returns_non_degenerate_auc_scores_for_imperfect_predictions(
        self, imperfect_multilabel_case
    ):
        """Ensure _get_global_eval_metrics computes probability-driven AUC values for imperfect predictions."""
        y_true, y_pred, y_prob = imperfect_multilabel_case

        metrics = get_global_eval_metrics(y_true, y_pred, y_prob)

        assert 0.0 < metrics["roc_auc_micro"] < 1.0
        assert 0.0 < metrics["pr_auc_micro"] < 1.0

    def test_get_global_eval_metrics_returns_non_trivial_f1_scores_for_imperfect_predictions(
        self, imperfect_multilabel_case
    ):
        """Ensure _get_global_eval_metrics returns non-trivial F1 values for imperfect predictions."""
        y_true, y_pred, y_prob = imperfect_multilabel_case

        metrics = get_global_eval_metrics(y_true, y_pred, y_prob)

        assert 0.0 < metrics["f1_micro"] < 1.0
        assert 0.0 < metrics["f1_macro"] < 1.0

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_global_eval_metrics_does_not_modify_inputs_in_place(self, case_fixture, request):
        """Ensure _get_global_eval_metrics does not modify input arrays in place."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)
        y_true0, y_pred0, y_prob0 = y_true.copy(), y_pred.copy(), y_prob.copy()

        get_global_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

        assert np.array_equal(y_true, y_true0)
        assert np.array_equal(y_pred, y_pred0)
        assert np.array_equal(y_prob, y_prob0)

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_label_eval_metrics_returns_complete_metric_dictionary(self, case_fixture, request, label_names_two):
        """Ensure _get_label_eval_metrics returns one metrics dict per label with the full documented key set."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)

        metrics = get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        assert list(metrics.keys()) == label_names_two
        for name in label_names_two:
            assert set(metrics[name]) == {
                "f1",
                "precision",
                "recall",
                "roc_auc",
                "pr_auc",
                "true_prevalence",
                "pred_prevalence",
            }

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_label_eval_metrics_returns_plain_floats(self, case_fixture, request, label_names_two):
        """Ensure _get_label_eval_metrics returns JSON-serializable float values for all labels and metrics."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)

        metrics = get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        assert all(isinstance(v, float) for per_label in metrics.values() for v in per_label.values())

    def test_get_label_eval_metrics_returns_expected_values_for_perfect_predictions(
        self, perfect_multilabel_case, label_names_two
    ):
        """Ensure _get_label_eval_metrics returns perfect per-label scores for perfect predictions."""
        y_true, y_pred, y_prob = perfect_multilabel_case

        metrics = get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        for name in label_names_two:
            for k in ["f1", "precision", "recall", "roc_auc", "pr_auc"]:
                assert metrics[name][k] == pytest.approx(1.0)

    def test_get_label_eval_metrics_returns_expected_prevalences_for_perfect_predictions(
        self, perfect_multilabel_case, label_names_two
    ):
        """Ensure _get_label_eval_metrics returns correct per-label true and predicted prevalences."""
        y_true, y_pred, y_prob = perfect_multilabel_case

        metrics = get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        for i, name in enumerate(label_names_two):
            assert metrics[name]["true_prevalence"] == pytest.approx(float(y_true[:, i].mean()))
            assert metrics[name]["pred_prevalence"] == pytest.approx(float(y_pred[:, i].mean()))

    def test_get_label_eval_metrics_is_label_specific_for_imperfect_predictions(
        self, imperfect_multilabel_case, label_names_two
    ):
        """Ensure _get_label_eval_metrics reflects per-label errors rather than collapsing to global behavior."""
        y_true, y_pred, y_prob = imperfect_multilabel_case

        metrics = get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        assert metrics["label_0"]["f1"] != metrics["label_1"]["f1"]
        assert metrics["label_0"]["f1"] < 1.0 or metrics["label_1"]["f1"] < 1.0

    def test_get_label_eval_metrics_auc_metrics_depend_on_probabilities_not_predictions(
        self, imperfect_multilabel_case, label_names_two
    ):
        """Ensure _get_label_eval_metrics AUC metrics change when probabilities change while labels stay fixed."""
        y_true, y_pred, y_prob = imperfect_multilabel_case

        metrics_good = get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        y_prob_bad = y_prob.copy()
        y_prob_bad[:, 0] = 1.0 - y_prob_bad[:, 0]

        metrics_bad = get_label_eval_metrics(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob_bad, label_names=label_names_two
        )

        assert metrics_good["label_0"]["roc_auc"] != metrics_bad["label_0"]["roc_auc"]
        assert metrics_good["label_0"]["pr_auc"] != metrics_bad["label_0"]["pr_auc"]

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_label_eval_metrics_does_not_modify_inputs_in_place(self, case_fixture, request, label_names_two):
        """Ensure _get_label_eval_metrics does not modify input arrays in place."""
        y_true, y_pred, y_prob = request.getfixturevalue(case_fixture)
        y_true0, y_pred0, y_prob0 = y_true.copy(), y_pred.copy(), y_prob.copy()

        get_label_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names_two)

        assert np.array_equal(y_true, y_true0)
        assert np.array_equal(y_pred, y_pred0)
        assert np.array_equal(y_prob, y_prob0)


class TestMetricRoundingUtil:
    """Test suite for metric rounding utility function."""

    def test_round_metric_dict_returns_new_dictionary_with_same_keys(self, raw_metric_dict):
        """Ensure _round_metric_dict returns a new dict preserving the original keys."""
        metrics_in = raw_metric_dict
        metrics_out = round_metric_dict(metrics_in)

        assert metrics_out is not metrics_in
        assert set(metrics_out.keys()) == set(metrics_in.keys())

    def test_round_metric_dict_does_not_modify_input_dictionary_in_place(self, raw_metric_dict):
        """Ensure _round_metric_dict does not modify the input dictionary in place."""
        metrics_in = raw_metric_dict
        snapshot = dict(metrics_in)

        round_metric_dict(metrics_in)

        assert metrics_in == snapshot

    @pytest.mark.parametrize("ndigits", [3, 1, 0])
    def test_round_metric_dict_respects_ndigits_parameter(self, raw_metric_dict, ndigits):
        """Ensure _round_metric_dict rounds all values using the provided ndigits value."""
        out = round_metric_dict(raw_metric_dict, ndigits=ndigits)
        assert out == {k: round(v, ndigits) for k, v in raw_metric_dict.items()}


class TestEvaluationArtifactsUtils:
    """Test suite for utilities that extract structured evaluation artifacts."""

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_roc_curves_returns_expected_structure_and_invariants(self, case_fixture, request, label_names_two):
        """Ensure _get_roc_curves returns the expected schema and basic ROC invariants."""
        y_true, _, y_prob = request.getfixturevalue(case_fixture)

        out = get_roc_curves(y_true=y_true, y_prob=y_prob, label_names=label_names_two)

        assert set(out) == {"fpr", "tpr", "roc_auc"}

        for k in [0, 1, "micro", "macro"]:
            assert k in out["fpr"]
            assert k in out["tpr"]
            assert k in out["roc_auc"]

        for i in [0, 1, "micro", "macro"]:
            fpr = out["fpr"][i]
            tpr = out["tpr"][i]

            assert isinstance(fpr, np.ndarray)
            assert isinstance(tpr, np.ndarray)
            assert fpr.ndim == 1
            assert tpr.ndim == 1
            assert fpr.shape == tpr.shape
            assert np.all((0.0 <= fpr) & (fpr <= 1.0))
            assert np.all((0.0 <= tpr) & (tpr <= 1.0))
            assert _is_non_decreasing(fpr)
            assert _is_non_decreasing(tpr)

            auc_val = out["roc_auc"][i]
            assert isinstance(auc_val, (float, np.floating))
            assert 0.0 <= float(auc_val) <= 1.0

    def test_get_roc_curves_auc_is_one_for_perfectly_separable_case(self, perfect_multilabel_case, label_names_two):
        """Ensure _get_roc_curves yields AUC=1.0 for all aggregates in a perfectly separable case."""
        y_true, _, y_prob = perfect_multilabel_case

        out = get_roc_curves(y_true=y_true, y_prob=y_prob, label_names=label_names_two)

        for k in [0, 1, "micro", "macro"]:
            assert float(out["roc_auc"][k]) == pytest.approx(1.0)

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_roc_curves_does_not_modify_inputs_in_place(self, case_fixture, request, label_names_two):
        """Ensure _get_roc_curves does not modify the input arrays in place."""
        y_true, _, y_prob = request.getfixturevalue(case_fixture)
        y_true0, y_prob0 = y_true.copy(), y_prob.copy()

        get_roc_curves(y_true=y_true, y_prob=y_prob, label_names=label_names_two)

        assert np.array_equal(y_true, y_true0)
        assert np.array_equal(y_prob, y_prob0)

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_pr_curves_returns_expected_structure_and_invariants(self, case_fixture, request, label_names_two):
        """Ensure _get_pr_curves returns the expected schema and basic PR invariants."""
        y_true, _, y_prob = request.getfixturevalue(case_fixture)

        out = get_pr_curves(y_true=y_true, y_prob=y_prob, label_names=label_names_two)

        assert set(out) == {"precision", "recall", "avg_precision"}

        for k in [0, 1, "micro", "macro"]:
            assert k in out["precision"]
            assert k in out["recall"]
            assert k in out["avg_precision"]

        for i in [0, 1, "micro", "macro"]:
            p = out["precision"][i]
            r = out["recall"][i]

            assert isinstance(p, np.ndarray)
            assert isinstance(r, np.ndarray)
            assert p.ndim == 1
            assert r.ndim == 1
            assert p.shape == r.shape
            assert np.all((0.0 <= p) & (p <= 1.0))
            assert np.all((0.0 <= r) & (r <= 1.0))

            if i == "macro":
                assert _is_non_decreasing(r)
            else:
                assert _is_non_increasing(r)

            ap = out["avg_precision"][i]
            assert isinstance(ap, (float, np.floating))
            assert 0.0 <= float(ap) <= 1.0

    def test_get_pr_curves_ap_is_one_for_perfectly_separable_case(self, perfect_multilabel_case, label_names_two):
        """Ensure _get_pr_curves yields AP=1.0 for all aggregates in a perfectly separable case."""
        y_true, _, y_prob = perfect_multilabel_case

        out = get_pr_curves(y_true=y_true, y_prob=y_prob, label_names=label_names_two)

        for k in [0, 1, "micro", "macro"]:
            assert float(out["avg_precision"][k]) == pytest.approx(1.0)

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_pr_curves_does_not_modify_inputs_in_place(self, case_fixture, request, label_names_two):
        """Ensure _get_pr_curves does not modify the input arrays in place."""
        y_true, _, y_prob = request.getfixturevalue(case_fixture)
        y_true0, y_prob0 = y_true.copy(), y_prob.copy()

        get_pr_curves(y_true=y_true, y_prob=y_prob, label_names=label_names_two)

        assert np.array_equal(y_true, y_true0)
        assert np.array_equal(y_prob, y_prob0)

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_co_occurrence_returns_expected_structure_and_invariants(self, case_fixture, request):
        """Ensure _get_co_occurrence returns expected schema and invariants."""
        y_true, y_pred, _ = request.getfixturevalue(case_fixture)

        out = get_co_occurrence(y_true=y_true, y_pred=y_pred)

        assert set(out) == {"co_true_abs", "co_pred_abs", "co_true_rel", "co_pred_rel"}

        for y, co_abs, co_rel in [
            (y_true, out["co_true_abs"], out["co_true_rel"]),
            (y_pred, out["co_pred_abs"], out["co_pred_rel"]),
        ]:
            n_labels = y.shape[1]

            assert co_abs.shape == (n_labels, n_labels)
            assert co_rel.shape == (n_labels, n_labels)

            assert np.array_equal(co_abs, y.T @ y)
            assert np.allclose(co_abs, co_abs.T)

            assert np.array_equal(np.diag(co_abs), y.sum(axis=0))

            assert np.allclose(co_rel, co_rel.T)
            assert np.array_equal(np.diag(co_rel), (np.diag(co_abs) > 0).astype(float))

            diag = np.diag(co_abs).astype(float)
            assert np.all(diag > 0)
            denom = np.sqrt(np.outer(diag, diag))
            expected_rel = co_abs.astype(float) / denom

            assert np.allclose(co_rel, expected_rel, rtol=1e-12, atol=1e-12)
            assert np.all((co_rel >= -1e-12) & (co_rel <= 1.0 + 1e-12))

    def test_get_co_occurrence_true_and_pred_matrices_match_for_perfect_case(self, perfect_multilabel_case):
        """Ensure true/pred co-occurrence outputs match when y_true == y_pred."""
        y_true, y_pred, _ = perfect_multilabel_case
        assert np.array_equal(y_true, y_pred)

        out = get_co_occurrence(y_true=y_true, y_pred=y_pred)

        assert np.array_equal(out["co_true_abs"], out["co_pred_abs"])
        assert np.allclose(out["co_true_rel"], out["co_pred_rel"], rtol=0.0, atol=0.0)

    def test_get_co_occurrence_reflects_prediction_differences(self, imperfect_multilabel_case):
        """Ensure predicted co-occurrence differs from true co-occurrence when y_pred differs."""
        y_true, y_pred, _ = imperfect_multilabel_case
        assert not np.array_equal(y_true, y_pred)

        out = get_co_occurrence(y_true=y_true, y_pred=y_pred)

        assert not np.array_equal(out["co_true_abs"], out["co_pred_abs"])
        assert not np.allclose(out["co_true_rel"], out["co_pred_rel"])

    @pytest.mark.parametrize("case_fixture", ["perfect_multilabel_case", "imperfect_multilabel_case"])
    def test_get_co_occurrence_does_not_modify_inputs_in_place(self, case_fixture, request):
        """Ensure _get_co_occurrence does not modify input arrays in place."""
        y_true, y_pred, _ = request.getfixturevalue(case_fixture)
        y_true0, y_pred0 = y_true.copy(), y_pred.copy()

        get_co_occurrence(y_true=y_true, y_pred=y_pred)

        assert np.array_equal(y_true, y_true0)
        assert np.array_equal(y_pred, y_pred0)

    def test_get_co_occurrence_handles_zero_predicted_support_without_undefined_values(self):
        """Ensure predicted zero-support labels produce finite relative co-occurrence values."""
        y_true = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )
        y_pred = np.array(
            [
                [1, 0],
                [1, 0],
                [0, 0],
            ]
        )

        out = get_co_occurrence(y_true=y_true, y_pred=y_pred)

        assert np.isfinite(out["co_true_rel"]).all()
        assert np.isfinite(out["co_pred_rel"]).all()

        assert np.array_equal(out["co_pred_abs"], y_pred.T @ y_pred)
        assert np.array_equal(np.diag(out["co_pred_abs"]), np.array([2, 0]))

        assert out["co_pred_rel"][0, 0] == pytest.approx(1.0)
        assert out["co_pred_rel"][1, 1] == pytest.approx(0.0)
        assert out["co_pred_rel"][0, 1] == pytest.approx(0.0)
        assert out["co_pred_rel"][1, 0] == pytest.approx(0.0)

    def test_get_losses_returns_expected_dataframe_and_inner_join_behavior(self, log_history_with_losses):
        """Ensure _get_losses returns a well-formed per-epoch loss table and keeps only shared epochs."""
        df = get_losses(log_history=log_history_with_losses)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["epoch", "train_loss", "eval_loss"]

        epochs = set(df["epoch"].tolist())
        assert epochs == {1.0, 3.0}

        df_idx = df.set_index("epoch")
        assert df_idx.loc[1.0, "train_loss"] == pytest.approx(0.9)
        assert df_idx.loc[1.0, "eval_loss"] == pytest.approx(0.8)
        assert df_idx.loc[3.0, "train_loss"] == pytest.approx(0.6)
        assert df_idx.loc[3.0, "eval_loss"] == pytest.approx(0.5)

    def test_get_losses_does_not_modify_log_history_in_place(self, log_history_with_losses):
        """Ensure _get_losses does not mutate the input log history."""
        snapshot = [dict(d) for d in log_history_with_losses]

        get_losses(log_history=log_history_with_losses)

        assert log_history_with_losses == snapshot

    def test_get_best_epoch_returns_epoch_of_max_metric(self, log_history_with_losses):
        """Ensure _get_best_epoch selects the epoch with the maximum monitored eval metric."""
        best = get_best_epoch(log_history=log_history_with_losses, best_model_metric="f1")

        assert isinstance(best, int)
        assert best == 2


class TestThresholdOptimizationUtils:
    """Test suite for optimal threshold selection utilities."""

    @pytest.mark.parametrize("metric", ["f1_micro", "f1_macro"])
    def test_find_optimal_threshold_selects_global_threshold_for_metric(self, metric):
        """Ensure `_find_optimal_threshold` finds a global threshold that maximizes the chosen F1 metric."""
        y_true = np.array(
            [
                [0, 1],
                [1, 0],
                [1, 1],
            ]
        )
        y_prob = np.array(
            [
                [0.2, 0.8],
                [0.7, 0.3],
                [0.9, 0.9],
            ]
        )

        threshold = find_optimal_threshold(
            y_true=y_true,
            y_prob=y_prob,
            best_threshold_metric=metric,
            threshold_type="global",
        )

        assert isinstance(threshold, np.ndarray)
        assert threshold.shape == (1,)
        assert 0.3 <= threshold[0] <= 0.32

    def test_find_optimal_threshold_returns_one_threshold_per_label(self):
        """Ensure `_find_optimal_threshold` returns separate thresholds for each label in label-specific mode."""
        y_true = np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ]
        )
        y_prob = np.array(
            [
                [0.4, 0.2],
                [0.8, 0.9],
                [0.1, 0.8],
            ]
        )

        thresholds = find_optimal_threshold(
            y_true=y_true,
            y_prob=y_prob,
            best_threshold_metric="f1_macro",
            threshold_type="label",
        )

        assert thresholds.shape == (2,)
        assert 0.10 <= thresholds[0] <= 0.12
        assert 0.19 <= thresholds[1] <= 0.21

    def test_find_optimal_threshold_raises_for_unknown_metric(self):
        """Ensure `_find_optimal_threshold` raises ValueError for unsupported best_threshold_metric values."""
        y_true = np.array([[1], [0]])
        y_prob = np.array([[0.8], [0.2]])

        with pytest.raises(ValueError):
            find_optimal_threshold(
                y_true=y_true,
                y_prob=y_prob,
                best_threshold_metric="not_a_metric",
                threshold_type="global",
            )

    def test_find_optimal_threshold_raises_for_unknown_threshold_type(self):
        """Ensure `_find_optimal_threshold` raises ValueError for unsupported threshold_type values."""
        y_true = np.array([[1], [0]])
        y_prob = np.array([[0.8], [0.2]])

        with pytest.raises(ValueError):
            find_optimal_threshold(
                y_true=y_true,
                y_prob=y_prob,
                best_threshold_metric="f1_micro",
                threshold_type="wrong",
            )
