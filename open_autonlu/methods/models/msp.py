from typing import Union

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_curve


def is_normalized(logits, tol=1e-5):
    """
    Check if the logits vector is normalized (i.e., softmax has been applied).

    Args:
        logits (torch.Tensor): The logits vector to check.
        tol (float): Tolerance for floating-point comparison.

    Returns:
        bool: True if the logits are normalized, False otherwise.
    """
    # Check if all values are between 0 and 1
    in_range = torch.all((logits >= 0) & (logits <= 1)).item()

    # Check if the sum of the logits is approximately 1
    sum_close_to_one = torch.isclose(
        logits.sum(), torch.tensor(1.0, dtype=logits.dtype), atol=tol
    ).item()

    return in_range and sum_close_to_one


class MaxSoftmaxProbabilityDetector(nn.Module):
    """Maximum Softmax Probability (MSP) OOD detector.

    Detects OOD samples based on the maximum softmax probability.
    Samples with max probability below the threshold are classified as OOD.
    """

    def __init__(
        self,
        threshold: Union[np.float32, float],
    ):
        """Initialize the MSP detector.

        Args:
            threshold: Probability threshold for in-scope classification.
        """
        super().__init__()
        self.threshold = nn.Parameter(torch.Tensor([threshold]), requires_grad=False)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Detect OOD samples.

        Args:
            scores: Batch of logits or probabilities.

        Returns:
            Tensor of 0/1 values where 1 indicates OOD.
        """
        if not is_normalized(scores[0]):
            scores = torch.softmax(scores, dim=1)
        max_probs = scores.max(dim=1).values
        return torch.where(max_probs > self.threshold, 0.0, 1.0)  # OOD-1, ID-0


class MaxSoftmaxProbabilityTrainer:
    """Trainer for Maximum Softmax Probability OOD detector.

    Determines optimal threshold using Youden's J statistic on
    in-scope and OOS validation samples.
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def train(
        inscope_logits: Union[np.ndarray, torch.tensor],
        oos_logits: Union[np.ndarray, torch.tensor],
        *args,
        **kwargs,
    ) -> MaxSoftmaxProbabilityDetector:
        """Train the MSP detector.

        Args:
            inscope_logits: Logits from in-scope validation samples.
            oos_logits: Logits from OOS validation samples.

        Returns:
            Trained MaxSoftmaxProbabilityDetector with optimal threshold.
        """
        inscope_logits = torch.tensor(inscope_logits)
        oos_logits = torch.tensor(oos_logits)
        if not is_normalized(inscope_logits[0]):
            inscope_logits = torch.softmax(torch.tensor(inscope_logits), dim=1)
        inscope_scores = np.array(inscope_logits).max(axis=1)
        if not is_normalized(oos_logits[0]):
            oos_logits = torch.softmax(torch.tensor(oos_logits), dim=1)
        oos_scores = np.array(oos_logits).max(axis=1)
        scores = np.hstack([inscope_scores, oos_scores])
        labels = np.hstack([np.ones_like(inscope_scores), np.zeros_like(oos_scores)])
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # Youden's J Statistic balances sensitivity (TPR) and specificity (1 - FPR)
        # The threshold that maximizes Youden's J is considered optimal
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        return MaxSoftmaxProbabilityDetector(threshold=optimal_threshold)  # type: ignore
