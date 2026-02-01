"""Tests for tlmtc.cli."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tlmtc.cli import _json_or_file, build_parser, main


class TestJsonOrFile:
    """Test suite for _json_or_file()."""

    def test_parses_json_object_from_string(self):
        """Ensure a JSON object string is parsed into a dict."""
        parsed = _json_or_file('{"a": 1, "b": "x"}')
        assert parsed == {"a": 1, "b": "x"}

    def test_parses_json_object_from_at_file(self, tmp_path: Path):
        """Ensure an '@'-prefixed path is read and parsed as a JSON object."""
        fp = tmp_path / "space.json"
        fp.write_text('{"lr_low": 1e-5}', encoding="utf-8")

        parsed = _json_or_file(f"@{fp}")
        assert parsed == {"lr_low": 1e-5}

    @pytest.mark.parametrize(
        "value, match",
        [
            ("not-json", r"Invalid JSON for --optuna-space-user"),
            ("[1, 2]", r"expected a JSON object"),
        ],
    )
    def test_rejects_invalid_or_non_object_json(self, value: str, match: str):
        """Ensure invalid JSON and non-object JSON are rejected with ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError, match=match):
            _json_or_file(value)

    def test_rejects_missing_at_file(self, tmp_path: Path):
        """Ensure a missing '@file' path is rejected with ArgumentTypeError."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(argparse.ArgumentTypeError, match=r"Invalid JSON for --optuna-space-user"):
            _json_or_file(f"@{missing}")


class TestBuildParser:
    """Test suite for build_parser()."""

    @pytest.mark.parametrize(
        "flag, expected",
        [
            ("--transfer-learning", True),
            ("--no-transfer-learning", False),
        ],
    )
    def test_boolean_optional_action_parses_explicit_flags(self, flag: str, expected: bool):
        """Ensure BooleanOptionalAction flags set the expected boolean value."""
        parser = build_parser()
        args = parser.parse_args(["--raw-csv", "raw.csv", flag])
        assert args.transfer_learning is expected

    def test_choice_validation_rejects_invalid_threshold_type(self):
        """Ensure argparse rejects invalid values for choice-constrained flags."""
        parser = build_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--raw-csv", "raw.csv", "--threshold-type", "nope"])
        assert excinfo.value.code == 2

    def test_optuna_space_user_accepts_json_object_string(self):
        """Ensure --optuna-space-user accepts an inline JSON object string."""
        parser = build_parser()
        args = parser.parse_args(["--raw-csv", "raw.csv", "--optuna-space-user", '{"batch_sizes": [8, 16]}'])
        assert args.optuna_space_user == {"batch_sizes": [8, 16]}

    def test_optuna_space_user_rejects_invalid_json(self):
        """Ensure --optuna-space-user rejects invalid JSON values."""
        parser = build_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--raw-csv", "raw.csv", "--optuna-space-user", "not-json"])
        assert excinfo.value.code == 2


class TestMain:
    """Test suite for cli.main()."""

    @pytest.fixture
    def stub_run_tlmtc(self, monkeypatch: pytest.MonkeyPatch):
        """Provide a stub tlmtc.run.run_tlmtc and capture kwargs passed by main()."""
        calls: dict[str, Any] = {}

        def _run_tlmtc(**kwargs: Any) -> None:
            calls["kwargs"] = kwargs

        stub = types.ModuleType("tlmtc.run")
        stub.run_tlmtc = _run_tlmtc  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "tlmtc.run", stub)
        return calls

    def test_main_invokes_run_tlmtc_and_returns_zero(self, stub_run_tlmtc):
        """Ensure main() maps argv to run_tlmtc kwargs and returns 0 on success."""
        exit_code = main(
            [
                "--raw-csv",
                "raw.csv",
                "--optuna-space-user",
                '{"lr_low": 1e-5}',
                "--no-hyperparameter-tuning",
            ]
        )

        assert exit_code == 0
        kwargs = stub_run_tlmtc["kwargs"]
        assert kwargs["raw_csv"] == "raw.csv"
        assert kwargs["optuna_space_user"] == {"lr_low": 1e-5}
        assert kwargs["hyperparameter_tuning"] is False

    def test_main_propagates_parser_error_when_run_tlmtc_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Ensure main() converts downstream exceptions into an argparse usage error (exit code 2)."""

        def _boom(**_kwargs: Any) -> None:
            raise RuntimeError("boom")

        stub = types.ModuleType("tlmtc.run")
        stub.run_tlmtc = _boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tlmtc.run", stub)

        with pytest.raises(SystemExit) as excinfo:
            main(["--raw-csv", "raw.csv"])

        # argparse uses exit code 2 for CLI usage errors.
        assert excinfo.value.code == 2

    def test_main_requires_raw_csv(self):
        """Ensure main() exits with a usage error when required args are missing."""
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2
