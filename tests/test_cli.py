"""Tests for tlmtc.cli."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from tlmtc.cli import app, parse_optuna_space
from tlmtc.settings import UNSET


@pytest.fixture
def runner() -> CliRunner:
    """Provide an isolated Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def stub_train_tlmtc(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Provide a stub tlmtc.api.train_tlmtc and capture kwargs passed by the CLI.

    The CLI imports train_tlmtc lazily from tlmtc.api inside the train command, so
    tests stub the module import target rather than patching a top-level cli symbol.
    """
    calls: dict[str, Any] = {}

    def _train_tlmtc(**kwargs: Any) -> SimpleNamespace:
        calls["kwargs"] = kwargs
        return SimpleNamespace(paths=SimpleNamespace(run_dir=Path("tlmtc_outputs/test-run")))

    stub = types.ModuleType("tlmtc.api")
    stub.train_tlmtc = _train_tlmtc  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tlmtc.api", stub)
    return calls


class TestParseOptunaSpace:
    """Test suite for parse_optuna_space()."""

    def test_returns_unset_for_omitted_value(self) -> None:
        """Ensure omitted CLI values map to UNSET."""
        assert parse_optuna_space(None) is UNSET

    def test_parses_json_object_from_string(self) -> None:
        """Ensure a JSON object string is parsed into a dict."""
        parsed = parse_optuna_space('{"a": 1, "b": "x"}')

        assert parsed == {"a": 1, "b": "x"}

    def test_parses_json_object_from_at_file(self, tmp_path: Path) -> None:
        """Ensure an '@'-prefixed path is read and parsed as a JSON object."""
        fp = tmp_path / "space.json"
        fp.write_text('{"lr_low": 1e-5}', encoding="utf-8")

        parsed = parse_optuna_space(f"@{fp}")

        assert parsed == {"lr_low": 1e-5}

    @pytest.mark.parametrize(
        "value, match",
        [
            ("not-json", r"Expected a JSON object or @file\.json"),
            ("[1, 2]", r"Expected a JSON object"),
        ],
    )
    def test_rejects_invalid_or_non_object_json(self, value: str, match: str) -> None:
        """Ensure invalid JSON and non-object JSON are rejected."""
        with pytest.raises(typer.BadParameter, match=match):
            parse_optuna_space(value)

    def test_rejects_missing_at_file(self, tmp_path: Path) -> None:
        """Ensure a missing '@file' path is rejected."""
        missing = tmp_path / "does_not_exist.json"

        with pytest.raises(typer.BadParameter, match=r"Could not read JSON file"):
            parse_optuna_space(f"@{missing}")


class TestCliApp:
    """Test suite for the Typer CLI app."""

    def test_root_help_is_shown_without_command(self, runner: CliRunner) -> None:
        """Ensure invoking the root command without a subcommand prints help."""
        result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert "Transfer learning for multi-label text classification." in result.output
        assert "train" in result.output

    def test_version_flag_prints_version(self, runner: CliRunner) -> None:
        """Ensure --version prints the package version."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "0.0.1" in result.output

    def test_train_help_is_available(self, runner: CliRunner) -> None:
        """Ensure train command help is available."""
        result = runner.invoke(app, ["train", "--help"])

        assert result.exit_code == 0
        assert "--raw-csv" in result.output
        assert "--optuna-space" in result.output
        assert "--use-cpu" in result.output

    @pytest.mark.parametrize(
        "flag, expected",
        [
            ("--transfer-learning", True),
            ("--no-transfer-learning", False),
        ],
    )
    def test_boolean_flag_pairs_parse_explicit_values(
        self,
        runner: CliRunner,
        flag: str,
        expected: bool,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure Typer boolean flag pairs set the expected boolean value."""
        result = runner.invoke(app, ["train", "--raw-csv", "raw.csv", flag])

        assert result.exit_code == 0
        assert stub_train_tlmtc["kwargs"]["transfer_learning"] is expected

    def test_train_invokes_train_tlmtc(
        self,
        runner: CliRunner,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure train maps CLI options to train_tlmtc kwargs."""
        result = runner.invoke(
            app,
            [
                "train",
                "--raw-csv",
                "raw.csv",
                "--optuna-space",
                '{"lr_low": 1e-5}',
                "--no-hyperparameter-tuning",
            ],
        )

        assert result.exit_code == 0
        assert "Run completed: tlmtc_outputs/test-run" in result.output

        kwargs = stub_train_tlmtc["kwargs"]
        assert kwargs["raw_csv"] == "raw.csv"
        assert kwargs["optuna_space"] == {"lr_low": 1e-5}
        assert kwargs["hyperparameter_tuning"] is False
        assert kwargs["batch_size"] is UNSET
        assert kwargs["threshold_type"] is UNSET
        assert kwargs["use_cpu"] is UNSET

    def test_train_forwards_config_path(
        self,
        runner: CliRunner,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure --config-path is forwarded to train_tlmtc."""
        result = runner.invoke(app, ["train", "--raw-csv", "raw.csv", "--config-path", "config.yaml"])

        assert result.exit_code == 0
        kwargs = stub_train_tlmtc["kwargs"]
        assert kwargs["raw_csv"] == "raw.csv"
        assert kwargs["config_path"] == "config.yaml"

    def test_train_omitted_optional_args_are_forwarded_as_unset(
        self,
        runner: CliRunner,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure omitted optional flags preserve layered settings semantics via UNSET."""
        result = runner.invoke(app, ["train", "--raw-csv", "raw.csv"])

        assert result.exit_code == 0

        kwargs = stub_train_tlmtc["kwargs"]
        assert kwargs["raw_csv"] == "raw.csv"
        assert kwargs["raw_test_csv"] is UNSET
        assert kwargs["work_dir"] is UNSET
        assert kwargs["batch_size"] is UNSET
        assert kwargs["hyperparameter_tuning"] is UNSET
        assert kwargs["threshold_type"] is UNSET
        assert kwargs["optuna_space"] is UNSET
        assert kwargs["use_cpu"] is UNSET

    def test_train_accepts_optuna_space_from_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure --optuna-space accepts an @file JSON object."""
        fp = tmp_path / "space.json"
        fp.write_text('{"batch_sizes": [8, 16]}', encoding="utf-8")

        result = runner.invoke(app, ["train", "--raw-csv", "raw.csv", "--optuna-space", f"@{fp}"])

        assert result.exit_code == 0
        assert stub_train_tlmtc["kwargs"]["optuna_space"] == {"batch_sizes": [8, 16]}

    def test_train_rejects_invalid_optuna_space_json(self, runner: CliRunner) -> None:
        """Ensure --optuna-space rejects invalid JSON values."""
        result = runner.invoke(app, ["train", "--raw-csv", "raw.csv", "--optuna-space", "not-json"])

        assert result.exit_code != 0
        assert "Expected a JSON object or @file.json" in result.output

    def test_train_rejects_non_object_optuna_space_json(self, runner: CliRunner) -> None:
        """Ensure --optuna-space rejects JSON values that are not objects."""
        result = runner.invoke(app, ["train", "--raw-csv", "raw.csv", "--optuna-space", "[1, 2]"])

        assert result.exit_code != 0
        assert "Expected a JSON object" in result.output

    def test_train_requires_raw_csv(self, runner: CliRunner) -> None:
        """Ensure train exits with a usage error when required args are missing."""
        result = runner.invoke(app, ["train"])

        assert result.exit_code != 0
        assert "Missing option" in result.output
        assert "--raw-csv" in result.output

    def test_train_propagates_downstream_exception(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure downstream exceptions are not converted into fake parser errors."""

        def _boom(**_kwargs: Any) -> None:
            raise RuntimeError("boom")

        stub = types.ModuleType("tlmtc.api")
        stub.train_tlmtc = _boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tlmtc.api", stub)

        result = runner.invoke(app, ["train", "--raw-csv", "raw.csv"])

        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)
        assert str(result.exception) == "boom"
