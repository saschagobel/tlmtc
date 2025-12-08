"""Tests for the WeightedTrainer and FinetunePipeline class."""

import torch
from torch import nn
from transformers import TrainingArguments

from tlmtc.finetune_pipeline import WeightedTrainer


class DummyModel(nn.Module):
    """Minimal feed-forward classifier used for testing the `WeightedTrainer`."""

    def __init__(self, num_labels=3):
        """Initialize the dummy model."""
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(4, num_labels)

    def forward(self, input_ids=None):
        """Compute logits for a batch of inputs."""
        logits = self.linear(input_ids.float())
        return type("Output", (), {"logits": logits})


def test_weighted_trainer_loss_changes_with_class_weights():
    """Ensure that applying class weights increases the computed BCE loss."""
    model = DummyModel(num_labels=3)

    inputs = {
        "labels": torch.tensor([[1, 0, 1], [0, 1, 0]]).float(),
        "input_ids": torch.zeros((2, 4)),
    }

    args = TrainingArguments(
        output_dir="test",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        report_to="none",
    )

    unweighted = WeightedTrainer(model=model, args=args, class_weights=None)
    loss_unweighted = unweighted.compute_loss(model, inputs.copy())

    cw = torch.tensor([5.0, 5.0, 5.0])
    weighted = WeightedTrainer(model=model, args=args, class_weights=cw)
    loss_weighted = weighted.compute_loss(model, inputs.copy())

    assert loss_weighted > loss_unweighted


def test_weighted_trainer_returns_outputs_when_requested():
    """Ensure that compute_loss returns (loss, outputs) when requested."""
    model = DummyModel(num_labels=3)
    args = TrainingArguments(output_dir="test", report_to="none")

    trainer = WeightedTrainer(model=model, args=args)
    inputs = {"labels": torch.zeros((1, 3)), "input_ids": torch.zeros((1, 4))}

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    assert isinstance(loss, torch.Tensor)
    assert hasattr(outputs, "logits")


def test_weighted_trainer_handles_model_module_attribute():
    """Ensure compute_loss reads num_labels correctly from model.module."""
    model = torch.nn.DataParallel(DummyModel(num_labels=4))
    args = TrainingArguments(output_dir="test", report_to="none")

    trainer = WeightedTrainer(model=model, args=args)
    inputs = {"labels": torch.zeros((1, 4)), "input_ids": torch.zeros((1, 4))}

    loss = trainer.compute_loss(model, inputs)
    assert isinstance(loss, torch.Tensor)
