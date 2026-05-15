"""Tests for tlmtc.paths."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tlmtc.data_contracts import InputMode
from tlmtc.meta import TrainRunMeta, write_run_meta
from tlmtc.paths import (
    DEFAULT_DATA_DIRNAME,
    DEFAULT_EVAL_DIRNAME,
    DEFAULT_LOGS_DIRNAME,
    DEFAULT_MODEL_DIRNAME,
    DEFAULT_PREDICTION_OUTPUTS_DIRNAME,
    DEFAULT_TRAIN_OUTPUTS_DIRNAME,
    TRAIN_RUN_META_FILENAME,
    find_latest_train_run_id,
    resolve_paths,
    resolve_prediction_paths,
)


def _write_train_run_meta(
    run_dir: Path,
    *,
    run_id: str,
    created_at: datetime,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(
        meta=TrainRunMeta(
            run_id=run_id,
            created_at=created_at,
            target_name="Target",
            checkpoint="checkpoint",
            proxy_checkpoint="proxy-checkpoint",
            sequence_length=128,
            input_mode=InputMode.SINGLE_TEXT,
            label_names=["a", "b"],
            threshold_type="label",
            thresholds=[0.5, 0.5],
            transfer_learning=True,
            hyperparameter_tuning=False,
            threshold_optimization=False,
            scale_learning_rate=False,
            wrap_peft=False,
        ),
        path=run_dir / TRAIN_RUN_META_FILENAME,
    )


def _make_prediction_source_run(
    work_dir: Path,
    *,
    run_id: str,
    created_at: datetime,
) -> Path:
    run_dir = work_dir / DEFAULT_TRAIN_OUTPUTS_DIRNAME / run_id
    _write_train_run_meta(run_dir, run_id=run_id, created_at=created_at)

    model_dir = run_dir / DEFAULT_MODEL_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "artifact.txt").write_text("model artifact", encoding="utf-8")

    return run_dir


class TestResolvePaths:
    """Test suite for resolve_paths."""

    def test_resolves_run_layout_under_work_dir(self, tmp_path: Path) -> None:
        """Ensure the run artifact layout is constructed under work_dir."""
        raw_csv = tmp_path / "input" / "raw.csv"
        raw_test_csv = tmp_path / "input" / "raw_test.csv"
        work_dir = tmp_path / "workspace"

        paths = resolve_paths(
            raw_csv=raw_csv,
            raw_test_csv=raw_test_csv,
            work_dir=work_dir,
            run_id="run123",
        )

        expected_work_dir = work_dir.resolve()
        expected_run_dir = expected_work_dir / DEFAULT_TRAIN_OUTPUTS_DIRNAME / "run123"

        assert paths.work_dir == expected_work_dir
        assert paths.run_id == "run123"
        assert paths.run_dir == expected_run_dir
        assert paths.train_run_meta_path == expected_run_dir / TRAIN_RUN_META_FILENAME
        assert paths.data_dir == expected_run_dir / DEFAULT_DATA_DIRNAME
        assert paths.eval_dir == expected_run_dir / DEFAULT_EVAL_DIRNAME
        assert paths.logs_dir == expected_run_dir / DEFAULT_LOGS_DIRNAME
        assert paths.model_dir == expected_run_dir / DEFAULT_MODEL_DIRNAME

        assert paths.train_data_path == paths.data_dir / "train.parquet"
        assert paths.val_data_path == paths.data_dir / "val.parquet"
        assert paths.test_data_path == paths.data_dir / "test.parquet"

        assert paths.global_metrics_path == paths.eval_dir / "global_metrics.json"
        assert paths.label_metrics_path == paths.eval_dir / "label_metrics.json"
        assert paths.global_metrics_table_path == paths.eval_dir / "global_metrics_table.html"
        assert paths.label_metrics_table_path == paths.eval_dir / "label_metrics_table.html"
        assert paths.hyperparameters_table_path == paths.eval_dir / "hyperparameters_table.html"
        assert paths.roc_plot_path == paths.eval_dir / "roc_plot.pdf"
        assert paths.co_occurrence_plot_path == paths.eval_dir / "co_occurrence.pdf"
        assert paths.loss_plot_path == paths.eval_dir / "loss_plot.pdf"
        assert paths.objective_values_plot_path == paths.eval_dir / "objective_values_plot.pdf"
        assert paths.optuna_trials_path == paths.logs_dir / "optuna_trials.db"

    def test_resolves_raw_inputs_independently_from_work_dir(self, tmp_path: Path, monkeypatch) -> None:
        """Ensure relative raw input paths are resolved from cwd, not from work_dir."""
        project_dir = tmp_path / "project"
        work_dir = tmp_path / "workspace"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        paths = resolve_paths(
            raw_csv=Path("data/raw.csv"),
            raw_test_csv=Path("data/raw_test.csv"),
            work_dir=work_dir,
            run_id="run123",
        )

        assert paths.work_dir == work_dir.resolve()
        assert paths.raw_data_path == (project_dir / "data" / "raw.csv").resolve()
        assert paths.raw_test_data_path == (project_dir / "data" / "raw_test.csv").resolve()

        assert paths.raw_data_path != paths.work_dir / "data" / "raw.csv"
        assert paths.raw_test_data_path != paths.work_dir / "data" / "raw_test.csv"

    def test_preserves_missing_raw_test_csv_as_none(self, tmp_path: Path) -> None:
        """Ensure omitted raw_test_csv stays absent instead of probing a sibling file."""
        raw_csv = tmp_path / "raw.csv"

        paths = resolve_paths(
            raw_csv=raw_csv,
            raw_test_csv=None,
            work_dir=tmp_path / "workspace",
            run_id="run123",
        )

        assert paths.raw_data_path == raw_csv.resolve()
        assert paths.raw_test_data_path is None

    def test_resolves_relative_work_dir_from_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """Ensure relative work_dir paths are resolved from the current working directory."""
        monkeypatch.chdir(tmp_path)

        paths = resolve_paths(
            raw_csv=Path("raw.csv"),
            raw_test_csv=None,
            work_dir=Path("workspace"),
            run_id="run123",
        )

        assert paths.work_dir == (tmp_path / "workspace").resolve()
        assert paths.run_dir == (tmp_path / "workspace" / DEFAULT_TRAIN_OUTPUTS_DIRNAME / "run123").resolve()
        assert paths.raw_data_path == (tmp_path / "raw.csv").resolve()

    def test_accepts_absolute_raw_inputs_outside_work_dir(self, tmp_path: Path) -> None:
        """Ensure raw inputs may live outside the artifact workspace."""
        raw_dir = tmp_path / "datasets"
        work_dir = tmp_path / "workspace"
        raw_csv = raw_dir / "train.csv"
        raw_test_csv = raw_dir / "test.csv"

        paths = resolve_paths(
            raw_csv=raw_csv,
            raw_test_csv=raw_test_csv,
            work_dir=work_dir,
            run_id="run123",
        )

        assert paths.raw_data_path == raw_csv.resolve()
        assert paths.raw_test_data_path == raw_test_csv.resolve()
        assert paths.run_dir == work_dir.resolve() / DEFAULT_TRAIN_OUTPUTS_DIRNAME / "run123"


class TestRunPaths:
    """Test suite for RunPaths."""

    def test_ensure_dirs_creates_artifact_structure_and_returns_self(self, tmp_path: Path) -> None:
        """Ensure ensure_dirs creates the artifact directory structure and returns self."""
        paths = resolve_paths(
            raw_csv=tmp_path / "raw.csv",
            raw_test_csv=None,
            work_dir=tmp_path / "workspace",
            run_id="run123",
        )

        result = paths.ensure_dirs()

        assert result is paths
        assert paths.run_dir.is_dir()
        assert paths.data_dir.is_dir()
        assert paths.eval_dir.is_dir()
        assert paths.logs_dir.is_dir()
        assert paths.model_dir.is_dir()

    def test_ensure_dirs_does_not_create_raw_input_parent_dirs(self, tmp_path: Path) -> None:
        """Ensure ensure_dirs only creates artifact directories, not raw input locations."""
        raw_csv = tmp_path / "missing-inputs" / "raw.csv"
        paths = resolve_paths(
            raw_csv=raw_csv,
            raw_test_csv=None,
            work_dir=tmp_path / "workspace",
            run_id="run123",
        )

        paths.ensure_dirs()

        assert paths.data_dir.is_dir()
        assert paths.eval_dir.is_dir()
        assert not raw_csv.parent.exists()


class TestFindLatestTrainRunId:
    """Test suite for find_latest_train_run_id."""

    def test_returns_run_id_with_latest_metadata_timestamp(self, tmp_path: Path) -> None:
        """Ensure latest-run selection uses persisted metadata timestamps."""
        train_outputs_dir = tmp_path / DEFAULT_TRAIN_OUTPUTS_DIRNAME
        now = datetime.now(UTC)

        _write_train_run_meta(
            train_outputs_dir / "older",
            run_id="older",
            created_at=now - timedelta(days=1),
        )
        _write_train_run_meta(
            train_outputs_dir / "newer",
            run_id="newer",
            created_at=now,
        )

        assert find_latest_train_run_id(train_outputs_dir) == "newer"

    def test_ignores_directories_without_training_metadata(self, tmp_path: Path) -> None:
        """Ensure directories without train_run_meta.json are not treated as completed runs."""
        train_outputs_dir = tmp_path / DEFAULT_TRAIN_OUTPUTS_DIRNAME
        train_outputs_dir.mkdir()
        (train_outputs_dir / "incomplete").mkdir()

        with pytest.raises(FileNotFoundError, match="No completed tlmtc training runs found"):
            find_latest_train_run_id(train_outputs_dir)


class TestResolvePredictionPaths:
    """Test suite for resolve_prediction_paths."""

    def test_resolves_prediction_layout_for_explicit_run_id(self, tmp_path: Path) -> None:
        """Ensure prediction paths are resolved from an existing training run."""
        work_dir = tmp_path / "workspace"
        input_csv = tmp_path / "inputs" / "predict.csv"
        input_csv.parent.mkdir()
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        run_dir = _make_prediction_source_run(
            work_dir,
            run_id="run123",
            created_at=datetime.now(UTC),
        )

        paths = resolve_prediction_paths(
            input_csv=input_csv,
            work_dir=work_dir,
            run_id="run123",
        )

        expected_work_dir = work_dir.resolve()
        expected_prediction_run_dir = expected_work_dir / DEFAULT_PREDICTION_OUTPUTS_DIRNAME / "run123"

        assert paths.work_dir == expected_work_dir
        assert paths.run_id == "run123"
        assert paths.input_data_path == input_csv.resolve()
        assert paths.train_outputs_dir == expected_work_dir / DEFAULT_TRAIN_OUTPUTS_DIRNAME
        assert paths.train_run_dir == run_dir.resolve()
        assert paths.train_run_meta_path == run_dir.resolve() / TRAIN_RUN_META_FILENAME
        assert paths.train_run_model_dir == run_dir.resolve() / DEFAULT_MODEL_DIRNAME
        assert paths.prediction_outputs_dir == expected_work_dir / DEFAULT_PREDICTION_OUTPUTS_DIRNAME
        assert paths.prediction_run_dir == expected_prediction_run_dir
        assert paths.probabilities_path == expected_prediction_run_dir / "probabilities.csv"
        assert paths.predictions_path == expected_prediction_run_dir / "predictions.csv"

    def test_resolves_latest_run_id_when_run_id_is_none(self, tmp_path: Path) -> None:
        """Ensure omitted run_id selects the latest completed training run."""
        work_dir = tmp_path / "workspace"
        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")
        now = datetime.now(UTC)

        _make_prediction_source_run(
            work_dir,
            run_id="older",
            created_at=now - timedelta(days=1),
        )
        _make_prediction_source_run(
            work_dir,
            run_id="newer",
            created_at=now,
        )

        paths = resolve_prediction_paths(
            input_csv=input_csv,
            work_dir=work_dir,
            run_id=None,
        )

        assert paths.run_id == "newer"
        assert paths.prediction_run_dir == work_dir.resolve() / DEFAULT_PREDICTION_OUTPUTS_DIRNAME / "newer"

    def test_does_not_create_prediction_output_dirs_during_resolution(self, tmp_path: Path) -> None:
        """Ensure path resolution does not create prediction artifact directories."""
        work_dir = tmp_path / "workspace"
        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        _make_prediction_source_run(
            work_dir,
            run_id="run123",
            created_at=datetime.now(UTC),
        )

        paths = resolve_prediction_paths(
            input_csv=input_csv,
            work_dir=work_dir,
            run_id="run123",
        )

        assert not paths.prediction_run_dir.exists()

    def test_fails_when_work_dir_does_not_exist(self, tmp_path: Path) -> None:
        """Ensure prediction never creates a missing work_dir."""
        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="`work_dir` does not exist"):
            resolve_prediction_paths(
                input_csv=input_csv,
                work_dir=tmp_path / "missing-workspace",
                run_id="run123",
            )

    def test_fails_when_prediction_input_csv_does_not_exist(self, tmp_path: Path) -> None:
        """Ensure missing prediction input fails before path construction succeeds."""
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Unlabeled prediction input CSV does not exist"):
            resolve_prediction_paths(
                input_csv=tmp_path / "missing.csv",
                work_dir=work_dir,
                run_id="run123",
            )

    def test_fails_when_train_outputs_dir_does_not_exist(self, tmp_path: Path) -> None:
        """Ensure prediction requires existing training outputs."""
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="No tlmtc training outputs found"):
            resolve_prediction_paths(
                input_csv=input_csv,
                work_dir=work_dir,
                run_id="run123",
            )

    def test_fails_when_explicit_run_id_does_not_exist(self, tmp_path: Path) -> None:
        """Ensure explicit run_id must point to an existing training run."""
        work_dir = tmp_path / "workspace"
        (work_dir / DEFAULT_TRAIN_OUTPUTS_DIRNAME).mkdir(parents=True)
        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="Requested tlmtc training run not found"):
            resolve_prediction_paths(
                input_csv=input_csv,
                work_dir=work_dir,
                run_id="missing-run",
            )

    def test_fails_when_training_metadata_is_missing(self, tmp_path: Path) -> None:
        """Ensure explicit training runs must contain train_run_meta.json."""
        work_dir = tmp_path / "workspace"
        run_dir = work_dir / DEFAULT_TRAIN_OUTPUTS_DIRNAME / "run123"
        model_dir = run_dir / DEFAULT_MODEL_DIRNAME
        model_dir.mkdir(parents=True)
        (model_dir / "artifact.txt").write_text("model artifact", encoding="utf-8")

        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="Training run metadata not found"):
            resolve_prediction_paths(
                input_csv=input_csv,
                work_dir=work_dir,
                run_id="run123",
            )

    def test_fails_when_training_model_dir_is_missing(self, tmp_path: Path) -> None:
        """Ensure selected training runs must contain a model directory."""
        work_dir = tmp_path / "workspace"
        run_dir = work_dir / DEFAULT_TRAIN_OUTPUTS_DIRNAME / "run123"
        _write_train_run_meta(
            run_dir,
            run_id="run123",
            created_at=datetime.now(UTC),
        )

        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="Training model directory not found"):
            resolve_prediction_paths(
                input_csv=input_csv,
                work_dir=work_dir,
                run_id="run123",
            )

    def test_fails_when_training_model_dir_is_empty(self, tmp_path: Path) -> None:
        """Ensure selected training runs must contain non-empty model artifacts."""
        work_dir = tmp_path / "workspace"
        run_dir = work_dir / DEFAULT_TRAIN_OUTPUTS_DIRNAME / "run123"
        _write_train_run_meta(
            run_dir,
            run_id="run123",
            created_at=datetime.now(UTC),
        )
        (run_dir / DEFAULT_MODEL_DIRNAME).mkdir()

        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="Training model directory is empty"):
            resolve_prediction_paths(
                input_csv=input_csv,
                work_dir=work_dir,
                run_id="run123",
            )


class TestPredictionPaths:
    """Test suite for PredictionPaths."""

    def test_ensure_dirs_creates_prediction_run_dir_and_returns_self(self, tmp_path: Path) -> None:
        """Ensure ensure_dirs creates the prediction artifact directory and returns self."""
        work_dir = tmp_path / "workspace"
        input_csv = tmp_path / "predict.csv"
        input_csv.write_text("text\nexample\n", encoding="utf-8")

        _make_prediction_source_run(
            work_dir,
            run_id="run123",
            created_at=datetime.now(UTC),
        )

        paths = resolve_prediction_paths(
            input_csv=input_csv,
            work_dir=work_dir,
            run_id="run123",
        )

        result = paths.ensure_dirs()

        assert result is paths
        assert paths.prediction_outputs_dir.is_dir()
        assert paths.prediction_run_dir.is_dir()
        assert not paths.probabilities_path.exists()
        assert not paths.predictions_path.exists()
