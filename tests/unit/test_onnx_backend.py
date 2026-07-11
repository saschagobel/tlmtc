"""Tests for ONNX backend operations."""

import builtins
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tlmtc.onnx_backend import _stage_merged_peft_model, export_onnx_model


@pytest.fixture
def olive_optimize(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide a fake Olive optimize API."""
    optimize = MagicMock()

    def fake_optimize(**kwargs: object) -> None:
        output_path = Path(str(kwargs["output_path"]))
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "model.onnx").write_bytes(b"placeholder")

    optimize.side_effect = fake_optimize

    olive_module = ModuleType("olive")
    olive_cli_module = ModuleType("olive.cli")
    olive_cli_api_module = ModuleType("olive.cli.api")
    olive_cli_api_module.optimize = optimize
    olive_cli_module.api = olive_cli_api_module
    olive_module.cli = olive_cli_module

    monkeypatch.setitem(sys.modules, "olive", olive_module)
    monkeypatch.setitem(sys.modules, "olive.cli", olive_cli_module)
    monkeypatch.setitem(sys.modules, "olive.cli.api", olive_cli_api_module)

    return optimize


def test_export_onnx_model_exports_full_model_from_model_dir(
    tmp_path: Path,
    olive_optimize: MagicMock,
) -> None:
    """Ensure full-model exports use the persisted model directory directly."""
    model_dir = tmp_path / "model"
    onnx_model_dir = model_dir / "onnx"

    export_onnx_model(
        model_dir=model_dir,
        onnx_model_dir=onnx_model_dir,
        checkpoint="base-checkpoint",
        num_labels=2,
        wrap_peft=False,
        trust_remote_code=False,
    )

    assert onnx_model_dir.is_dir()
    olive_optimize.assert_called_once_with(
        model_name_or_path=str(model_dir),
        task="text-classification",
        output_path=str(onnx_model_dir),
        device="cpu",
        exporter="dynamo_exporter",
    )


def test_export_onnx_model_exports_peft_model_from_staging_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    olive_optimize: MagicMock,
) -> None:
    """Ensure PEFT exports stage a merged full model before calling Olive."""
    model_dir = tmp_path / "model"
    onnx_model_dir = model_dir / "onnx"
    stage_model = MagicMock()

    monkeypatch.setattr("tlmtc.onnx_backend._stage_merged_peft_model", stage_model)

    export_onnx_model(
        model_dir=model_dir,
        onnx_model_dir=onnx_model_dir,
        checkpoint="base-checkpoint",
        num_labels=3,
        wrap_peft=True,
        trust_remote_code=True,
    )

    assert onnx_model_dir.is_dir()
    stage_model.assert_called_once()

    _, stage_kwargs = stage_model.call_args
    staging_model_dir = stage_kwargs["staging_model_dir"]

    assert stage_kwargs == {
        "model_dir": model_dir,
        "staging_model_dir": staging_model_dir,
        "checkpoint": "base-checkpoint",
        "num_labels": 3,
        "trust_remote_code": True,
    }
    assert isinstance(staging_model_dir, Path)
    assert not staging_model_dir.exists()
    olive_optimize.assert_called_once_with(
        model_name_or_path=str(staging_model_dir),
        task="text-classification",
        output_path=str(onnx_model_dir),
        device="cpu",
        exporter="dynamo_exporter",
    )


def test_export_onnx_model_raises_when_olive_writes_no_onnx_file(
    tmp_path: Path,
    olive_optimize: MagicMock,
) -> None:
    """Ensure silent Olive export failures do not get recorded as successful artifacts."""
    olive_optimize.side_effect = None

    with pytest.raises(RuntimeError, match="did not produce an ONNX model artifact"):
        export_onnx_model(
            model_dir=tmp_path / "model",
            onnx_model_dir=tmp_path / "model" / "onnx",
            checkpoint="base-checkpoint",
            num_labels=2,
            wrap_peft=False,
            trust_remote_code=False,
        )


@pytest.mark.parametrize(
    ("trust_remote_code", "expected_default_available"),
    [(False, False), (True, True)],
)
def test_export_onnx_model_makes_default_rope_available_only_for_trusted_remote_code(
    tmp_path: Path,
    olive_optimize: MagicMock,
    trust_remote_code: bool,
    expected_default_available: bool,
) -> None:
    """Ensure default RoPE is available at trusted remote-code ONNX export."""
    original_rope_init_functions = dict(ROPE_INIT_FUNCTIONS)
    ROPE_INIT_FUNCTIONS.pop("default", None)

    def fake_optimize(**kwargs: object) -> None:
        assert ("default" in ROPE_INIT_FUNCTIONS) is expected_default_available
        output_path = Path(str(kwargs["output_path"]))
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "model.onnx").write_bytes(b"placeholder")

    olive_optimize.side_effect = fake_optimize

    try:
        export_onnx_model(
            model_dir=tmp_path / "model",
            onnx_model_dir=tmp_path / "model" / "onnx",
            checkpoint="base-checkpoint",
            num_labels=2,
            wrap_peft=False,
            trust_remote_code=trust_remote_code,
        )
    finally:
        ROPE_INIT_FUNCTIONS.clear()
        ROPE_INIT_FUNCTIONS.update(original_rope_init_functions)


def test_stage_merged_peft_model_merges_adapter_and_reuses_persisted_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure PEFT staging creates a full-model directory for Olive."""
    model_dir = tmp_path / "model"
    staging_model_dir = tmp_path / "staging"

    peft_model = MagicMock()
    merged_model = MagicMock()
    tokenizer = MagicMock()
    peft_model.merge_and_unload.return_value = merged_model
    load_prediction_model = MagicMock(return_value=peft_model)
    from_pretrained = MagicMock(return_value=tokenizer)

    monkeypatch.setattr("tlmtc.onnx_backend.load_prediction_model", load_prediction_model)
    monkeypatch.setattr("tlmtc.onnx_backend.AutoTokenizer.from_pretrained", from_pretrained)

    _stage_merged_peft_model(
        model_dir=model_dir,
        staging_model_dir=staging_model_dir,
        checkpoint="base-checkpoint",
        num_labels=4,
        trust_remote_code=True,
    )

    load_prediction_model.assert_called_once_with(
        model_dir=model_dir,
        inference_backend="torch",
        checkpoint="base-checkpoint",
        num_labels=4,
        wrap_peft=True,
        trust_remote_code=True,
    )
    peft_model.merge_and_unload.assert_called_once_with()
    merged_model.save_pretrained.assert_called_once_with(staging_model_dir)
    from_pretrained.assert_called_once_with(model_dir, trust_remote_code=True)
    tokenizer.save_pretrained.assert_called_once_with(staging_model_dir)


def test_export_onnx_model_requires_onnx_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure missing Olive dependencies produce actionable install guidance."""
    original_import = builtins.__import__

    def import_without_olive(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "olive.cli.api":
            raise ImportError("No module named 'olive'")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_olive)

    with pytest.raises(RuntimeError, match=r"tlmtc\[train,onnx-export\]"):
        export_onnx_model(
            model_dir=tmp_path / "model",
            onnx_model_dir=tmp_path / "model" / "onnx",
            checkpoint="base-checkpoint",
            num_labels=2,
            wrap_peft=False,
            trust_remote_code=False,
        )
