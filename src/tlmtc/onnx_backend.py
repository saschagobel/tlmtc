"""ONNX backend operations for trained tlmtc model artifacts."""

from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

from tlmtc.prediction import load_prediction_model


def _stage_merged_peft_model(
    *,
    model_dir: Path,
    staging_model_dir: Path,
    checkpoint: str,
    num_labels: int,
    trust_remote_code: bool,
) -> None:
    """Stage a merged full model from PEFT adapter artifacts.

    Args:
        model_dir: Directory containing persisted PEFT adapter and tokenizer artifacts.
        staging_model_dir: Temporary destination for the merged full-model export source.
        checkpoint: Base checkpoint used to load the PEFT adapter model.
        num_labels: Number of labels in the trained classification head.
        trust_remote_code: Whether Hugging Face loading may execute custom remote code.
    """
    peft_model = load_prediction_model(
        model_dir=model_dir,
        checkpoint=checkpoint,
        num_labels=num_labels,
        wrap_peft=True,
        trust_remote_code=trust_remote_code,
    )
    merged_model = getattr(peft_model, "merge_and_unload")()
    merged_model.save_pretrained(staging_model_dir)
    AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code).save_pretrained(staging_model_dir)


def export_onnx_model(
    *,
    model_dir: Path,
    onnx_model_dir: Path,
    checkpoint: str,
    num_labels: int,
    wrap_peft: bool,
    trust_remote_code: bool,
) -> None:
    """Export trained model artifacts to an ONNX inference model.

    Args:
        model_dir: Directory containing the persisted full model or PEFT adapter artifacts.
        onnx_model_dir: Destination directory for ONNX artifacts under the model artifact tree.
        checkpoint: Base checkpoint used for tokenizer loading and PEFT adapter merging.
        num_labels: Number of labels in the trained classification head.
        wrap_peft: Whether `model_dir` contains PEFT adapter artifacts.
        trust_remote_code: Whether Hugging Face loading may execute custom remote code.
    """
    try:
        from olive.cli.api import optimize
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the optional ONNX dependencies. "
            "Install the training and ONNX extras with `tlmtc[full,onnx]`."
        ) from exc

    onnx_model_dir.mkdir(parents=True, exist_ok=True)

    if not wrap_peft:
        optimize(
            model_name_or_path=str(model_dir),
            task="text-classification",
            output_path=str(onnx_model_dir),
            device="cpu",
            exporter="dynamo_exporter",
        )
    else:
        with TemporaryDirectory(prefix="tlmtc-onnx-export-") as staging_dir:
            staging_model_dir = Path(staging_dir)
            _stage_merged_peft_model(
                model_dir=model_dir,
                staging_model_dir=staging_model_dir,
                checkpoint=checkpoint,
                num_labels=num_labels,
                trust_remote_code=trust_remote_code,
            )
            optimize(
                model_name_or_path=str(staging_model_dir),
                task="text-classification",
                output_path=str(onnx_model_dir),
                device="cpu",
                exporter="dynamo_exporter",
            )

    if not any(onnx_model_dir.rglob("*.onnx")):
        raise RuntimeError(f"ONNX export did not produce an ONNX model artifact under {onnx_model_dir}.")
