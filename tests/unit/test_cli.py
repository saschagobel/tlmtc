"""Tests for tlmtc.cli."""

import re
import sys
import types
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from tlmtc.cli import app, parse_json_object
from tlmtc.settings import UNSET

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
OPTUNA_JSON_OPTION = {"option_name": "--optuna-space", "example": '{"lr_low": 1e-5}'}


def clean_cli_output(
    output: str,
) -> str:
    """Remove ANSI styling from Typer/Rich CLI output."""
    return _ANSI_ESCAPE_RE.sub("", output)


@pytest.fixture
def runner() -> CliRunner:
    """Provide an isolated Typer CLI runner."""
    return CliRunner()


def invoke_cli(
    runner: CliRunner,
    args: list[str],
) -> Any:
    """Invoke the Typer app with deterministic help/error rendering."""
    return runner.invoke(app, args, color=False, terminal_width=120)


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


@pytest.fixture
def stub_predict_tlmtc(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Provide a stub tlmtc.api.predict_tlmtc and capture kwargs passed by the CLI.

    The CLI imports predict_tlmtc lazily from tlmtc.api inside the predict command, so
    tests stub the module import target rather than patching a top-level cli symbol.
    """
    calls: dict[str, Any] = {}

    def _predict_tlmtc(**kwargs: Any) -> SimpleNamespace:
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            paths=SimpleNamespace(
                prediction_run_dir=Path("prediction_outputs/test-run"),
                probabilities_path=Path("prediction_outputs/test-run/probabilities.csv"),
                predictions_path=Path("prediction_outputs/test-run/predictions.csv"),
            )
        )

    stub = types.ModuleType("tlmtc.api")
    stub.predict_tlmtc = _predict_tlmtc  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tlmtc.api", stub)
    return calls


class TestParseJsonObject:
    """Test suite for parse_json_object()."""

    def test_returns_unset_for_omitted_value(self) -> None:
        """Ensure omitted CLI values map to UNSET."""
        assert parse_json_object(None, **OPTUNA_JSON_OPTION) is UNSET

    def test_parses_json_object_from_string(self) -> None:
        """Ensure a JSON object string is parsed into a dict."""
        parsed = parse_json_object('{"a": 1, "b": "x"}', **OPTUNA_JSON_OPTION)

        assert parsed == {"a": 1, "b": "x"}

    def test_parses_json_object_from_at_file(self, tmp_path: Path) -> None:
        """Ensure an '@'-prefixed path is read and parsed as a JSON object."""
        fp = tmp_path / "space.json"
        fp.write_text('{"lr_low": 1e-5}', encoding="utf-8")

        parsed = parse_json_object(f"@{fp}", **OPTUNA_JSON_OPTION)

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
            parse_json_object(value, **OPTUNA_JSON_OPTION)

    def test_rejects_missing_at_file(self, tmp_path: Path) -> None:
        """Ensure a missing '@file' path is rejected."""
        missing = tmp_path / "does_not_exist.json"

        with pytest.raises(typer.BadParameter, match=r"Could not read JSON file"):
            parse_json_object(f"@{missing}", **OPTUNA_JSON_OPTION)


class TestCliApp:
    """Test suite for shared Typer CLI app behavior."""

    def test_root_help_is_shown_without_command(self, runner: CliRunner) -> None:
        """Ensure invoking the root command without a subcommand prints help."""
        result = invoke_cli(runner, [])
        output = clean_cli_output(result.output)

        assert result.exit_code == 0
        assert "Production Workflows for Transformer-based Multi-Label Text Classification." in output
        assert "train" in output
        assert "predict" in output

    def test_version_flag_prints_version(self, runner: CliRunner) -> None:
        """Ensure --version prints the package version."""
        result = invoke_cli(runner, ["--version"])
        output = clean_cli_output(result.output)

        assert result.exit_code == 0
        assert output.strip() == version("tlmtc")


class TestTrainCliApp:
    """Test suite for the train CLI command."""

    def test_train_help_is_available(self, runner: CliRunner) -> None:
        """Ensure train command help is available."""
        result = invoke_cli(runner, ["train", "--help"])
        output = clean_cli_output(result.output)

        assert result.exit_code == 0
        assert "--labeled-data" in output
        assert "--optuna-space" in output
        assert "--trainer-args" in output
        assert "--use-cpu" in output
        assert "--verbosity" in output
        assert "--trust-remote-co" in output

    @pytest.mark.parametrize(
        "flag, expected",
        [
            ("--transfer-learning", True),
            ("--no-transfer-learning", False),
        ],
    )
    def test_train_boolean_flag_pairs_parse_explicit_values(
        self,
        runner: CliRunner,
        flag: str,
        expected: bool,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure train boolean flag pairs set the expected boolean value."""
        result = invoke_cli(runner, ["train", "--labeled-data", "raw.csv", flag])

        assert result.exit_code == 0
        assert stub_train_tlmtc["kwargs"]["transfer_learning"] is expected

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("--trust-remote-code", True),
            ("--no-trust-remote-code", False),
        ],
    )
    def test_train_trust_remote_code_flag_pair_parse_explicit_values(
        self,
        runner: CliRunner,
        flag: str,
        expected: bool,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        result = invoke_cli(runner, ["train", "--labeled-data", "raw.csv", flag])

        assert result.exit_code == 0
        assert stub_train_tlmtc["kwargs"]["trust_remote_code"] is expected

    @pytest.mark.parametrize("verbosity", ["progress", "quiet"])
    def test_train_forwards_verbosity(
        self,
        runner: CliRunner,
        verbosity: str,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure --verbosity is forwarded to train_tlmtc."""
        result = invoke_cli(
            runner,
            [
                "train",
                "--labeled-data",
                "raw.csv",
                "--verbosity",
                verbosity,
            ],
        )

        assert result.exit_code == 0
        assert stub_train_tlmtc["kwargs"]["verbosity"] == verbosity

    def test_train_invokes_train_tlmtc(
        self,
        runner: CliRunner,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure train maps CLI options to train_tlmtc kwargs."""
        result = invoke_cli(
            runner,
            [
                "train",
                "--labeled-data",
                "raw.csv",
                "--optuna-space",
                '{"lr_low": 1e-5}',
                "--trainer-args",
                '{"gradient_accumulation_steps": 4}',
                "--no-hyperparameter-tuning",
                "--trust-remote-code",
            ],
        )
        output = clean_cli_output(result.output)

        assert result.exit_code == 0
        assert "Run completed: tlmtc_outputs/test-run" in output

        kwargs = stub_train_tlmtc["kwargs"]
        assert kwargs["labeled_data"] == "raw.csv"
        assert kwargs["optuna_space"] == {"lr_low": 1e-5}
        assert kwargs["trainer_args"] == {"gradient_accumulation_steps": 4}
        assert kwargs["hyperparameter_tuning"] is False
        assert kwargs["batch_size"] is UNSET
        assert kwargs["trust_remote_code"] is True
        assert kwargs["threshold_type"] is UNSET
        assert kwargs["use_cpu"] is UNSET

    def test_train_forwards_config_path(
        self,
        runner: CliRunner,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure --config-path is forwarded to train_tlmtc."""
        result = invoke_cli(runner, ["train", "--labeled-data", "raw.csv", "--config-path", "config.yaml"])

        assert result.exit_code == 0
        kwargs = stub_train_tlmtc["kwargs"]
        assert kwargs["labeled_data"] == "raw.csv"
        assert kwargs["config_path"] == "config.yaml"

    def test_train_omitted_optional_args_are_forwarded_as_unset(
        self,
        runner: CliRunner,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure omitted optional flags preserve layered settings semantics via UNSET."""
        result = invoke_cli(runner, ["train", "--labeled-data", "raw.csv"])

        assert result.exit_code == 0

        kwargs = stub_train_tlmtc["kwargs"]
        assert kwargs["labeled_data"] == "raw.csv"
        assert kwargs["raw_test_csv"] is UNSET
        assert kwargs["work_dir"] is UNSET
        assert kwargs["batch_size"] is UNSET
        assert kwargs["trust_remote_code"] is UNSET
        assert kwargs["hyperparameter_tuning"] is UNSET
        assert kwargs["threshold_type"] is UNSET
        assert kwargs["optuna_space"] is UNSET
        assert kwargs["trainer_args"] is UNSET
        assert kwargs["use_cpu"] is UNSET
        assert kwargs["verbosity"] is UNSET

    def test_train_accepts_optuna_space_from_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_train_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure --optuna-space accepts an @file JSON object."""
        fp = tmp_path / "space.json"
        fp.write_text('{"batch_sizes": [8, 16]}', encoding="utf-8")

        result = invoke_cli(runner, ["train", "--labeled-data", "raw.csv", "--optuna-space", f"@{fp}"])

        assert result.exit_code == 0
        assert stub_train_tlmtc["kwargs"]["optuna_space"] == {"batch_sizes": [8, 16]}

    def test_train_rejects_invalid_optuna_space_json(self, runner: CliRunner) -> None:
        """Ensure --optuna-space rejects invalid JSON values."""
        result = invoke_cli(runner, ["train", "--labeled-data", "raw.csv", "--optuna-space", "not-json"])
        output = clean_cli_output(result.output)

        assert result.exit_code != 0
        assert "Expected a JSON object or @file.json" in output

    def test_train_rejects_non_object_optuna_space_json(self, runner: CliRunner) -> None:
        """Ensure --optuna-space rejects JSON values that are not objects."""
        result = invoke_cli(runner, ["train", "--labeled-data", "raw.csv", "--optuna-space", "[1, 2]"])
        output = clean_cli_output(result.output)

        assert result.exit_code != 0
        assert "Expected a JSON object" in output

    def test_train_requires_labeled_data(self, runner: CliRunner) -> None:
        """Ensure train exits with a usage error when required args are missing."""
        result = invoke_cli(runner, ["train"])
        output = clean_cli_output(result.output)

        assert result.exit_code != 0
        assert "Missing option" in output
        assert "--labeled-data" in output

    def test_train_propagates_downstream_exception(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure downstream training exceptions are not converted into fake parser errors."""

        def _boom(**_kwargs: Any) -> None:
            raise RuntimeError("boom")

        stub = types.ModuleType("tlmtc.api")
        stub.train_tlmtc = _boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tlmtc.api", stub)

        result = invoke_cli(runner, ["train", "--labeled-data", "raw.csv"])

        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)
        assert str(result.exception) == "boom"


class TestPredictCliApp:
    """Test suite for the predict CLI command."""

    def test_predict_help_is_available(self, runner: CliRunner) -> None:
        """Ensure predict command help is available."""
        result = invoke_cli(runner, ["predict", "--help"])
        output = clean_cli_output(result.output)

        assert result.exit_code == 0
        assert "--prediction-csv" in output
        assert "--run-id" in output
        assert "--batch-size" in output
        assert "--trust-remote-co" in output
        assert "--use-cpu" in output
        assert "--verbosity" in output

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("--trust-remote-code", True),
            ("--no-trust-remote-code", False),
        ],
    )
    def test_predict_trust_remote_code_flag_pair_parse_explicit_values(
        self,
        runner: CliRunner,
        flag: str,
        expected: bool,
        stub_predict_tlmtc: dict[str, Any],
    ) -> None:
        result = invoke_cli(runner, ["predict", "--prediction-csv", "prediction.csv", flag])

        assert result.exit_code == 0
        assert stub_predict_tlmtc["kwargs"]["trust_remote_code"] is expected

    @pytest.mark.parametrize(
        "flag, expected",
        [
            ("--use-cpu", True),
            ("--no-use-cpu", False),
        ],
    )
    def test_predict_boolean_flag_pairs_parse_explicit_values(
        self,
        runner: CliRunner,
        flag: str,
        expected: bool,
        stub_predict_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure predict boolean flag pairs set the expected boolean value."""
        result = invoke_cli(runner, ["predict", "--prediction-csv", "prediction.csv", flag])

        assert result.exit_code == 0
        assert stub_predict_tlmtc["kwargs"]["use_cpu"] is expected

    @pytest.mark.parametrize("verbosity", ["progress", "quiet"])
    def test_predict_forwards_verbosity(
        self,
        runner: CliRunner,
        verbosity: str,
        stub_predict_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure --verbosity is forwarded to predict_tlmtc."""
        result = invoke_cli(
            runner,
            [
                "predict",
                "--prediction-csv",
                "prediction.csv",
                "--verbosity",
                verbosity,
            ],
        )

        assert result.exit_code == 0
        assert stub_predict_tlmtc["kwargs"]["verbosity"] == verbosity

    def test_predict_invokes_predict_tlmtc(
        self,
        runner: CliRunner,
        stub_predict_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure predict maps CLI options to predict_tlmtc kwargs."""
        result = invoke_cli(
            runner,
            [
                "predict",
                "--prediction-csv",
                "prediction.csv",
                "--work-dir",
                "work",
                "--run-id",
                "test-run",
                "--batch-size",
                "8",
                "--trust-remote-code",
                "--use-cpu",
            ],
        )
        output = clean_cli_output(result.output)

        assert result.exit_code == 0
        assert "Prediction completed: prediction_outputs/test-run" in output
        assert "Probabilities: prediction_outputs/test-run/probabilities.csv" in output
        assert "Predictions: prediction_outputs/test-run/predictions.csv" in output

        kwargs = stub_predict_tlmtc["kwargs"]
        assert kwargs["prediction_csv"] == "prediction.csv"
        assert kwargs["work_dir"] == "work"
        assert kwargs["run_id"] == "test-run"
        assert kwargs["batch_size"] == 8
        assert kwargs["trust_remote_code"] is True
        assert kwargs["use_cpu"] is True

    def test_predict_forwards_config_path(
        self,
        runner: CliRunner,
        stub_predict_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure --config-path is forwarded to predict_tlmtc."""
        result = invoke_cli(
            runner,
            [
                "predict",
                "--prediction-csv",
                "prediction.csv",
                "--config-path",
                "config.yaml",
            ],
        )

        assert result.exit_code == 0
        kwargs = stub_predict_tlmtc["kwargs"]
        assert kwargs["prediction_csv"] == "prediction.csv"
        assert kwargs["config_path"] == "config.yaml"

    def test_predict_omitted_optional_args_are_forwarded_as_unset(
        self,
        runner: CliRunner,
        stub_predict_tlmtc: dict[str, Any],
    ) -> None:
        """Ensure omitted predict flags preserve layered settings semantics via UNSET."""
        result = invoke_cli(runner, ["predict", "--prediction-csv", "prediction.csv"])

        assert result.exit_code == 0

        kwargs = stub_predict_tlmtc["kwargs"]
        assert kwargs["prediction_csv"] == "prediction.csv"
        assert kwargs["work_dir"] is UNSET
        assert kwargs["config_path"] is UNSET
        assert kwargs["run_id"] is UNSET
        assert kwargs["batch_size"] is UNSET
        assert kwargs["trust_remote_code"] is UNSET
        assert kwargs["use_cpu"] is UNSET
        assert kwargs["verbosity"] is UNSET

    def test_predict_requires_prediction_csv(self, runner: CliRunner) -> None:
        """Ensure predict exits with a usage error when required args are missing."""
        result = invoke_cli(runner, ["predict"])
        output = clean_cli_output(result.output)

        assert result.exit_code != 0
        assert "Missing option" in output
        assert "--prediction-csv" in output

    def test_predict_propagates_downstream_exception(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure downstream prediction exceptions are not converted into fake parser errors."""

        def _boom(**_kwargs: Any) -> None:
            raise RuntimeError("boom")

        stub = types.ModuleType("tlmtc.api")
        stub.predict_tlmtc = _boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tlmtc.api", stub)

        result = invoke_cli(runner, ["predict", "--prediction-csv", "prediction.csv"])

        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)
        assert str(result.exception) == "boom"
