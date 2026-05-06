"""Tests for tlmtc.paths."""

from pathlib import Path

from tlmtc.paths import (
    DEFAULT_DATA_DIRNAME,
    DEFAULT_EVAL_DIRNAME,
    DEFAULT_LOGS_DIRNAME,
    DEFAULT_MODEL_DIRNAME,
    DEFAULT_OUTPUTS_DIRNAME,
    resolve_paths,
)


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
        expected_run_dir = expected_work_dir / DEFAULT_OUTPUTS_DIRNAME / "run123"

        assert paths.work_dir == expected_work_dir
        assert paths.run_id == "run123"
        assert paths.run_dir == expected_run_dir
        assert paths.run_meta_path == expected_run_dir / "run_meta.json"
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
        assert paths.run_dir == (tmp_path / "workspace" / DEFAULT_OUTPUTS_DIRNAME / "run123").resolve()
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
        assert paths.run_dir == work_dir.resolve() / DEFAULT_OUTPUTS_DIRNAME / "run123"


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
