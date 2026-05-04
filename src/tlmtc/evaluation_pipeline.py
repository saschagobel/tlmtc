"""Evaluation pipeline for trained multi-label classifiers."""

import json
from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
import torch
from datasets import DatasetDict
from matplotlib import pyplot as plt
from transformers import Trainer

from tlmtc.data_contracts import InputMode
from tlmtc.evaluation import (
    get_best_epoch,
    get_co_occurrence,
    get_global_eval_metrics,
    get_label_eval_metrics,
    get_losses,
    get_roc_curves,
)
from tlmtc.paths import RunPaths
from tlmtc.reporting import (
    make_cooccurrence_heatmaps_plot,
    make_global_metrics_table,
    make_hyperparameters_table,
    make_label_metrics_table,
    make_loss_curves_plot,
    make_objective_values_plot,
    make_roc_curves_plot,
)
from tlmtc.settings import ModelSettings, TrainingSettings, WorkflowSettings


class EvaluationPipeline:
    """Evaluate a trained multi-label classifier and render persisted reports.

    Attributes:
        tokenized_dataset: Tokenized Hugging Face dataset ready for PyTorch
        updated_trainer: The instantiated Trainer after fine-tuning.
        paths: Resolved filesystem locations for persisted train and validation splits, logs, and model outputs.
        model: Model configuration (proxy-checkpoint and checkpoint).
        workflow: High-level workflow toggles (HPO, learning rate scaling, transfer learning, threshold optimization).
        training: Resolved training input settings.
        tuned_threshold: Tuned global or label-specific thresholds for multi-label classification
        input_mode: Input mode inferred from the validated data contract.
        label_names: Human-readable label names without the ``label_`` prefix.
        probabilities: Test-set predicted probabilities.
        true_labels: Test-set ground-truth labels.
        pred_labels: Test-set thresholded predictions.
        global_eval_metrics: Aggregated multi-label evaluation metrics.
        label_eval_metrics: Label-specific evaluation metrics.
        roc_curves: ROC curve data for global and label-specific plots.
        co_occurrence: True/predicted label co-occurrence matrices.
        losses: Per-epoch train/evaluation losses.
        best_epoch: Best epoch according to ``training.best_model_metric``.
        hp_objective_values: Optuna trial numbers and objective values.
    """

    def __init__(
        self,
        tokenized_dataset: DatasetDict,
        updated_trainer: Trainer | None,
        paths: RunPaths,
        model: ModelSettings,
        workflow: WorkflowSettings,
        training: TrainingSettings,
        tuned_threshold: np.ndarray,
        input_mode: InputMode | None,
    ) -> None:
        """Initialize the evaluation pipeline.

        Args:
            tokenized_dataset: Tokenized Hugging Face dataset ready for PyTorch
            updated_trainer: The instantiated Trainer after fine-tuning.
            paths: Resolved filesystem locations for persisted train and validation splits, logs, and model outputs.
            model: Model configuration (proxy-checkpoint and checkpoint).
            workflow: High-level workflow toggles (HPO, learning rate scaling, transfer learning,
                threshold optimization).
            training: Resolved training input settings.
            tuned_threshold: Tuned global or label-specific thresholds for multi-label classification
            input_mode: Input mode inferred from the validated data contract.
        """
        self.tokenized_dataset = tokenized_dataset
        self.updated_trainer = updated_trainer
        self.paths = paths
        self.model = model
        self.workflow = workflow
        self.training = training
        self.tuned_threshold = tuned_threshold
        self.input_mode = input_mode
        self.label_names: list[str] | None = None
        self.probabilities: np.ndarray | None = None
        self.true_labels: np.ndarray | None = None
        self.pred_labels: np.ndarray | None = None
        self.global_eval_metrics: dict[str, float] | None = None
        self.label_eval_metrics: dict[str, dict[str, float]] | None = None
        self.roc_curves: dict[str, dict[int | str, Any]] | None = None
        self.co_occurrence: dict[str, np.ndarray] | None = None
        self.losses: pd.DataFrame | None = None
        self.best_epoch: int | None = None
        self.hp_objective_values: pd.DataFrame | None = None

    def run_evaluation(self) -> Self:
        """Compute test metrics and collect HPO diagnostics.

        Returns:
            EvaluationPipeline: The updated pipeline instance.
        """
        if self.workflow.hyperparameter_tuning:
            if not self.paths.optuna_trials_path.exists():
                raise RuntimeError(f"Optuna study database not found at {self.paths.optuna_trials_path}.")

            optuna_study = optuna.load_study(
                study_name=f"{self.model.target_name.replace(' ', '_')}_optuna_study",
                storage=f"sqlite:///{self.paths.optuna_trials_path.as_posix()}",
            )
            self.hp_objective_values = optuna_study.trials_dataframe(attrs=("number", "value"))[["number", "value"]]

            if self.hp_objective_values.empty:
                raise RuntimeError("Optuna study does not contain any completed trials with objective values.")

        if not self.workflow.transfer_learning:
            return self

        if self.updated_trainer is None:
            raise RuntimeError("Trained model not found. Run FinetunePipeline.fine_tune_pretrained() first.")

        self.label_names = [
            label.removeprefix("label_")
            for label in pd.read_parquet(self.paths.train_data_path).columns
            if label.startswith("label_")
        ]

        preds = self.updated_trainer.predict(self.tokenized_dataset["test"])
        logits = preds.predictions

        self.probabilities = torch.sigmoid(torch.as_tensor(logits)).numpy()
        self.true_labels = np.asarray(self.tokenized_dataset["test"]["labels"])
        self.pred_labels = (self.probabilities >= self.tuned_threshold).astype(int)

        self.global_eval_metrics = get_global_eval_metrics(
            y_true=self.true_labels,
            y_pred=self.pred_labels,
            y_prob=self.probabilities,
        )
        self.label_eval_metrics = get_label_eval_metrics(
            y_true=self.true_labels,
            y_pred=self.pred_labels,
            y_prob=self.probabilities,
            label_names=self.label_names,
        )
        self.roc_curves = get_roc_curves(
            y_true=self.true_labels,
            y_prob=self.probabilities,
            label_names=self.label_names,
        )
        self.co_occurrence = get_co_occurrence(
            y_true=self.true_labels,
            y_pred=self.pred_labels,
        )
        self.losses = get_losses(log_history=self.updated_trainer.state.log_history)
        self.best_epoch = get_best_epoch(
            log_history=self.updated_trainer.state.log_history,
            best_model_metric=self.training.best_model_metric,
        )
        return self

    def save_metrics(self) -> Self:
        """Save global and label-specific evaluation metrics as JSON.

        Returns:
            EvaluationPipeline: The updated pipeline instance.
        """
        if not self.workflow.transfer_learning:
            return self

        if self.global_eval_metrics is None or self.label_eval_metrics is None:
            raise RuntimeError("Evaluation metrics not found. Run run_evaluation() first.")

        self.paths.global_metrics_path.write_text(
            json.dumps(self.global_eval_metrics, indent=4),
            encoding="utf-8",
        )
        self.paths.label_metrics_path.write_text(
            json.dumps(self.label_eval_metrics, indent=4),
            encoding="utf-8",
        )
        return self

    def render_tables(self) -> Self:
        """Render evaluation and hyperparameter summary tables as HTML.

        Returns:
            EvaluationPipeline: The updated pipeline instance.
        """
        if not self.workflow.transfer_learning:
            return self

        if self.global_eval_metrics is None or self.label_eval_metrics is None or self.label_names is None:
            raise RuntimeError("Evaluation metrics not found. Run run_evaluation() first.")

        if self.updated_trainer is None:
            raise RuntimeError("Trained model not found. Run fine-tuning before rendering hyperparameter tables.")

        if self.input_mode is None:
            raise RuntimeError("Input mode not found. Run DataPipeline.split_data() before EvaluationPipeline.")

        input_mode_label = "Paired text" if self.input_mode is InputMode.PAIRED_TEXT else "Single text"

        global_metrics_table = make_global_metrics_table(
            eval_metrics=self.global_eval_metrics,
            target_name=self.model.target_name,
            label_names=self.label_names,
            checkpoint=self.model.checkpoint,
            train_data_path=self.paths.train_data_path,
            test_data_path=self.paths.test_data_path,
            input_mode=input_mode_label,
        )
        label_metrics_table = make_label_metrics_table(
            eval_metrics=self.label_eval_metrics,
            target_name=self.model.target_name,
            checkpoint=self.model.checkpoint,
            train_data_path=self.paths.train_data_path,
            test_data_path=self.paths.test_data_path,
            input_mode=input_mode_label,
        )
        hyperparameters_table = make_hyperparameters_table(
            threshold=self.tuned_threshold,
            trainer=self.updated_trainer,
            target_name=self.model.target_name,
            checkpoint=self.model.checkpoint,
            input_mode=input_mode_label,
        )
        global_metrics_table.write_raw_html(self.paths.global_metrics_table_path)
        label_metrics_table.write_raw_html(self.paths.label_metrics_table_path)
        hyperparameters_table.write_raw_html(self.paths.hyperparameters_table_path)
        return self

    def render_figures(self) -> Self:
        """Render HPO and test-set diagnostic figures as PDFs.

        Returns:
            EvaluationPipeline: The updated pipeline instance.
        """
        if self.workflow.hyperparameter_tuning:
            if self.hp_objective_values is None:
                raise RuntimeError("Hyperparameter tuning results not found. Run run_evaluation() first.")

            objective_values_plot = make_objective_values_plot(
                objective_values=self.hp_objective_values,
                target_name=self.model.target_name,
                checkpoint=self.model.proxy_checkpoint,
            )
            objective_values_plot.savefig(self.paths.objective_values_plot_path, bbox_inches="tight")
            plt.close(objective_values_plot)

        if not self.workflow.transfer_learning:
            return self

        if self.roc_curves is None or self.co_occurrence is None or self.losses is None or self.label_names is None:
            raise RuntimeError("Evaluation figure data not found. Run run_evaluation() first.")

        if self.best_epoch is None:
            raise RuntimeError("Best epoch not found. Run run_evaluation() first.")

        roc_plot = make_roc_curves_plot(
            roc_curves=self.roc_curves,
            target_name=self.model.target_name,
            checkpoint=self.model.checkpoint,
            label_names=self.label_names,
        )
        co_occurrence_plot = make_cooccurrence_heatmaps_plot(
            co_occurrence=self.co_occurrence,
            target_name=self.model.target_name,
            checkpoint=self.model.checkpoint,
            label_names=self.label_names,
        )
        loss_plot = make_loss_curves_plot(
            losses=self.losses,
            target_name=self.model.target_name,
            checkpoint=self.model.checkpoint,
            best_epoch=self.best_epoch,
        )

        roc_plot.savefig(self.paths.roc_plot_path, bbox_inches="tight")
        co_occurrence_plot.savefig(self.paths.co_occurrence_plot_path, bbox_inches="tight")
        loss_plot.savefig(self.paths.loss_plot_path, bbox_inches="tight")

        plt.close(roc_plot)
        plt.close(co_occurrence_plot)
        plt.close(loss_plot)

        return self
