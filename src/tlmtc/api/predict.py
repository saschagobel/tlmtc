"""Public Python API for running tlmtc prediction workflows."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tlmtc.data_preparation import create_prediction_dataset, read_prediction_data, tokenize_prediction_dataset
from tlmtc.distributed import create_prediction_context
from tlmtc.meta import read_run_meta
from tlmtc.paths import PredictionPaths, resolve_prediction_paths
from tlmtc.prediction import (
    apply_thresholds,
    load_prediction_model,
    make_prediction_frame,
    predict_probabilities,
)
from tlmtc.runtime_output import configure_runtime_output, emit_progress
from tlmtc.settings import UNSET, PredictionSettings, Unset, load_config_file


@dataclass(frozen=True, slots=True)
class PredictResult:
    """Result metadata for a completed tlmtc prediction run.

    Attributes:
        paths: Resolved filesystem layout containing prediction inputs and generated artifacts.
    """

    paths: PredictionPaths


def predict_tlmtc(
    unlabeled_data: str | Path | pd.DataFrame,
    *,
    work_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
    run_id: str | None | Unset = UNSET,
    inference_backend: str | Unset = UNSET,
    batch_size: int | Unset = UNSET,
    trust_remote_code: bool | Unset = UNSET,
    use_cpu: bool | Unset = UNSET,
    verbosity: str | Unset = UNSET,
) -> PredictResult:
    """Run the multi-label text classification prediction workflow.

    Prediction consumes persisted metadata and model artifacts from a completed
    training run, applies the persisted decision thresholds, and writes probability
    and binary prediction artifacts.

    Args:
        unlabeled_data: Path to unlabeled prediction data, or an in-memory DataFrame. The data must contain
            a `text` column and, for models trained with paired-text inputs, a `text_pair` column.
            Prediction artifacts preserve input text columns unchanged.
        work_dir: Base directory for resolving inputs, reading training artifacts, and writing
            prediction artifacts. Defaults to the current working directory.
        config_path: Path to a YAML configuration file. Defaults to no configuration file.
        run_id: Run identifier used to select the completed training run. If omitted, the latest
            completed training run is selected from persisted training metadata. Prediction reloads
            the trained model or adapter artifacts using the resolved `trust_remote_code` setting.
            Only use saved model artifacts and adapters you trust.
        inference_backend: Runtime backend used for prediction. Defaults to `"torch"`.
        batch_size: Prediction batch size used for batched inference. Defaults to `32`.
        trust_remote_code: Whether Hugging Face tokenizer and model loading may execute custom
            remote code during prediction. Defaults to `False`. Required for runs trained with
            `trust_remote_code=True`; only enable it for artifacts and checkpoints you trust.
        use_cpu: Whether to force CPU execution. Defaults to `False`.
        verbosity: Runtime output mode. Supported values are `"progress"` and `"quiet"`. Defaults to
            `"progress"`.

    Returns:
        Result metadata containing the resolved input and artifact paths.
    """
    settings = PredictionSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env=None,
        overrides={
            "unlabeled_data": unlabeled_data,
            "work_dir": work_dir,
            "run_id": run_id,
            "inference_backend": inference_backend,
            "batch_size": batch_size,
            "trust_remote_code": trust_remote_code,
            "hardware": {
                "use_cpu": use_cpu,
            },
            "runtime": {
                "verbosity": verbosity,
            },
        },
    )

    distributed = create_prediction_context(
        inference_backend=settings.inference_backend,
        use_cpu=settings.hardware.use_cpu,
    )
    configure_runtime_output(
        settings.runtime.verbosity,
        is_main_process=distributed.is_main_process,
    )

    emit_progress("Starting prediction run")

    paths = resolve_prediction_paths(
        unlabeled_data=settings.unlabeled_data if isinstance(settings.unlabeled_data, Path) else None,
        work_dir=settings.work_dir,
        run_id=settings.run_id,
    )

    distributed.run_on_main(paths.ensure_dirs, sync=True)

    emit_progress("Reading training metadata")

    meta = read_run_meta(paths.train_run_meta_path)

    if not meta.transfer_learning:
        raise RuntimeError(
            "Prediction requires a training run with transfer_learning=True. "
            f"Run '{meta.run_id}' did not persist a fine-tuned prediction model."
        )

    if meta.trust_remote_code and not settings.trust_remote_code:
        raise RuntimeError(
            f"Prediction run '{meta.run_id}' was trained with trust_remote_code=True, "
            "but prediction was started with trust_remote_code=False. "
            "Re-run prediction with trust_remote_code=True or --trust-remote-code only if you trust "
            f"the Hugging Face model repository or local model artifacts for '{meta.checkpoint}'."
        )

    input_mode = meta.input_mode
    label_names = meta.label_names

    assert input_mode is not None
    assert label_names is not None

    if settings.inference_backend == "onnx" and "onnx" not in meta.model_backends:
        raise RuntimeError(f"Training run '{meta.run_id}' does not provide an ONNX model backend.")

    emit_progress("Reading prediction inputs")

    assert isinstance(settings.unlabeled_data, pd.DataFrame) or paths.unlabeled_data_path is not None
    input_df = read_prediction_data(
        data=(
            settings.unlabeled_data if isinstance(settings.unlabeled_data, pd.DataFrame) else paths.unlabeled_data_path
        ),
        expected_input_mode=input_mode,
    )
    prediction_dataset = create_prediction_dataset(
        df=input_df,
        input_mode=input_mode,
    )
    emit_progress("Tokenizing prediction inputs")
    tokenized_dataset = tokenize_prediction_dataset(
        dataset=prediction_dataset,
        tokenizer_dir=paths.train_run_model_dir,
        input_mode=input_mode,
        sequence_length=meta.sequence_length,
        trust_remote_code=settings.trust_remote_code,
        inference_backend=settings.inference_backend,
    )
    emit_progress("Loading fine-tuned prediction model")
    model = load_prediction_model(
        model_dir=paths.train_run_model_dir,
        inference_backend=settings.inference_backend,
        checkpoint=meta.checkpoint,
        num_labels=len(label_names),
        wrap_peft=meta.wrap_peft,
        trust_remote_code=settings.trust_remote_code,
    )
    emit_progress("Running prediction")
    probabilities = predict_probabilities(
        model=model,
        dataset=tokenized_dataset,
        batch_size=settings.batch_size,
        use_cpu=settings.hardware.use_cpu,
        inference_backend=settings.inference_backend,
    )
    probability_df = make_prediction_frame(
        input_df=input_df,
        values=probabilities,
        label_names=label_names,
    )
    predictions = apply_thresholds(
        probabilities=probabilities,
        thresholds=meta.thresholds,
    )
    prediction_df = make_prediction_frame(
        input_df=input_df,
        values=predictions,
        label_names=label_names,
    )

    emit_progress("Writing prediction artifacts")
    distributed.run_on_main(probability_df.to_csv, paths.probabilities_path, index=False)
    distributed.run_on_main(prediction_df.to_csv, paths.predictions_path, index=False)

    emit_progress("Prediction run complete")
    return PredictResult(paths=paths)
