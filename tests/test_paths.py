"""Tests for tlmtc.paths."""

from pathlib import Path

import pytest

from tlmtc.paths import (
    DEFAULT_DATA_DIRNAME,
    DEFAULT_LOGS_DIRNAME,
    DEFAULT_MODEL_DIRNAME,
    DEFAULT_OUTPUTS_DIRNAME,
    resolve_paths,
)


class TestResolvePaths:
    """Test suite for resolve_paths."""

    def test_resolves_relative_inputs_under_work_dir(self, tmp_path: Path):
        """Ensure relative input paths are anchored under work_dir and run layout is constructed."""
        paths = resolve_paths(
            raw_csv="data/raw.csv",
            raw_test_csv="data/raw_test.csv",
            work_dir=tmp_path,
            run_id="run123",
        )

        assert paths.work_dir == tmp_path.resolve()
        assert paths.raw_data_path == (tmp_path / "data" / "raw.csv").resolve()
        assert paths.raw_test_data_path == (tmp_path / "data" / "raw_test.csv").resolve()

        expected_run_dir = (tmp_path / DEFAULT_OUTPUTS_DIRNAME / "run123").resolve()
        assert paths.run_dir == expected_run_dir
        assert paths.data_dir == expected_run_dir / DEFAULT_DATA_DIRNAME
        assert paths.logs_dir == expected_run_dir / DEFAULT_LOGS_DIRNAME
        assert paths.model_dir == expected_run_dir / DEFAULT_MODEL_DIRNAME

        assert paths.train_data_path == paths.data_dir / "train.parquet"
        assert paths.val_data_path == paths.data_dir / "val.parquet"
        assert paths.test_data_path == paths.data_dir / "test.parquet"

    def test_defaults_raw_test_csv_to_sibling_raw_test(self, tmp_path: Path):
        """Ensure raw_test_csv defaults to a sibling raw_test.csv next to raw_csv."""
        raw_csv = tmp_path / "raw.csv"
        paths = resolve_paths(raw_csv=raw_csv, work_dir=tmp_path, run_id="run123")

        assert paths.raw_test_data_path == raw_csv.with_name("raw_test.csv").resolve()

    def test_uses_cwd_when_work_dir_is_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Ensure work_dir defaults to the current working directory when not provided."""
        monkeypatch.chdir(tmp_path)

        paths = resolve_paths(raw_csv="raw.csv", run_id="run123")

        assert paths.work_dir == tmp_path.resolve()
        assert paths.raw_data_path == (tmp_path / "raw.csv").resolve()

    def test_generates_run_id_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Ensure a default run_id is generated when run_id is not provided."""
        monkeypatch.setattr("tlmtc.paths._default_run_id", lambda: "fixed-run-id", raising=True)

        paths = resolve_paths(raw_csv="raw.csv", work_dir=tmp_path)

        assert paths.run_id == "fixed-run-id"
        assert paths.run_dir == (tmp_path / DEFAULT_OUTPUTS_DIRNAME / "fixed-run-id").resolve()


class TestRunPaths:
    """Test suite for RunPaths."""

    def test_ensure_dirs_creates_structure_and_returns_self(self, tmp_path: Path):
        """Ensure ensure_dirs creates the run directory structure and returns self."""
        paths = resolve_paths(raw_csv="raw.csv", work_dir=tmp_path, run_id="run123")

        result = paths.ensure_dirs()

        assert result is paths
        assert paths.run_dir.is_dir()
        assert paths.data_dir.is_dir()
        assert paths.logs_dir.is_dir()
        assert paths.model_dir.is_dir()
