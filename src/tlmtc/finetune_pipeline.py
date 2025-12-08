"""
Transfer Learning for Multi-Label Text Classification.

Hyperparameter tuning and model fine-tuning.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from transformers import Trainer
from transformers.modeling_outputs import ModelOutput


class WeightedTrainer(Trainer):
    """
    Custom Hugging Face Trainer class-balanced loss weighting for multi-label classification.

    Parameters
    ----------
    *args : Any
        Positional arguments passed to the parent 'Trainer'
    class_weights : Optional[torch.FloatTensor], default=None
        A 1D tensor of positive weights for each class. Used as 'pos_weight' in 'torch.nn.BCEWithLogitsLoss' to
        up- or down-weight positive examples depending on class imbalance
    **kwargs : Any
        Keyword arguments passed to the parent 'Trainer'

    Attributes
    ----------
    loss_fct : torch.nn.BCEWithLogitsLoss
        Loss function configured with optional class weights
    """

    def __init__(
        self,
        *args: Any,
        class_weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)

        self.loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=class_weights,
            reduction="mean",
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, ModelOutput]:
        """
        Compute the weighted BCE loss for multi-label classification with multi-GPU support.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained
        inputs : dict
            Input batch, including 'labels' and other model-specific input tensors
        return_outputs : bool, optional
            If True, return a tuple (loss, outputs), default is False
        num_items_in_batch : int, optional
            Number of items in the current batch, default is None

        Returns
        -------
        loss : torch.Tensor
            The computed loss value
        outputs : ModelOutput, optional
            Returned only if 'return_outputs' is True, contains model outputs
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        num_labels = getattr(model, "num_labels", None)
        if num_labels is None:
            num_labels = getattr(model.module, "num_labels")

        loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))

        return (loss, outputs) if return_outputs else loss
