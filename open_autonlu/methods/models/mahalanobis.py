from typing import Optional, Union

import numpy as np
from sklearn.metrics import f1_score, make_scorer
import torch
from sklearn.covariance import LedoitWolf
from sklearn.model_selection._classification_threshold import TunedThresholdClassifierCV
from torch import nn


class MarginalMahalanobisDetector(nn.Module):
    """Mahalanobis distance-based OOD detector.

    Detects out-of-distribution samples by computing Mahalanobis distance
    from the in-scope data distribution. Samples with distance exceeding
    the threshold are classified as OOD.

    Attributes:
        mean_vector: Mean of in-scope embeddings.
        precision: Inverse covariance matrix.
        threshold: Distance threshold for OOD classification.
    """

    def __init__(
        self,
        mean_vector: np.ndarray,
        precision: np.ndarray,
        threshold: Union[np.float32, float],
    ):
        """Initialize the Mahalanobis detector.

        Args:
            mean_vector: Mean vector of in-scope embedding distribution.
            precision: Precision (inverse covariance) matrix.
            threshold: Mahalanobis distance threshold for OOD detection.
        """
        super().__init__()
        self.mean_vector = nn.Parameter(
            torch.from_numpy(mean_vector), requires_grad=False
        )
        self.precision = nn.Parameter(torch.from_numpy(precision), requires_grad=False)
        self.threshold = nn.Parameter(torch.Tensor([threshold]), requires_grad=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Detect OOD samples.

        Args:
            embeddings: Batch of embedding vectors.

        Returns:
            Tensor of 0/1 values where 1 indicates OOD.
        """
        zero_f = embeddings - self.mean_vector
        indices = torch.arange(0, zero_f.shape[0])
        term_gau = 0.5 * ((zero_f @ self.precision) @ zero_f.T)[indices, indices]
        return torch.where(term_gau > self.threshold, 1.0, 0.0)


class MarginalMahalanobisEstimator:
    def __init__(self, mean_vector: np.ndarray, precision: np.ndarray):
        self.mean_vector = mean_vector
        self.precision = precision
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"

    def fit(self, embeddings: np.ndarray) -> None: ...

    def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        zero_f = embeddings - self.mean_vector
        term_gau = 0.5 * np.diag((zero_f @ self.precision) @ zero_f.T)
        return term_gau


class MarginalMahalanobisTrainer:
    """Trainer for Mahalanobis-based OOD detection.

    Fits a Mahalanobis distance detector using Ledoit-Wolf covariance
    estimation. Optionally uses OOS samples to tune the detection threshold.
    """

    def __init__(self, seed: int = 42):
        """Initialize the trainer.

        Args:
            seed: Random seed for threshold tuning.
        """
        self.covariance_estimator = LedoitWolf()
        self.seed = seed

    def train(
        self,
        inscope_embeddings: np.ndarray,
        oos_embeddings: Optional[np.ndarray] = None,
        threshold_magnifier: Optional[float] = 1.0,
    ) -> MarginalMahalanobisDetector:
        """Train the Mahalanobis detector.

        Args:
            inscope_embeddings: Embeddings of in-scope training samples.
            oos_embeddings: Optional OOS embeddings for threshold tuning.
            threshold_magnifier: Multiplier for the detection threshold.

        Returns:
            Trained MarginalMahalanobisDetector instance.
        """
        mean_vector = np.mean(inscope_embeddings, axis=0).astype(np.float32)
        covariance = self.covariance_estimator.fit(
            inscope_embeddings
        ).covariance_.astype(np.float32)
        precision = np.linalg.inv(covariance).astype(np.float32)
        estimator = MarginalMahalanobisEstimator(mean_vector, precision)
        if oos_embeddings is not None:
            threshold_estimator = TunedThresholdClassifierCV(
                estimator=estimator,
                refit=False,
                random_state=self.seed,
                scoring=make_scorer(f1_score, response_method="decision_function"),  # type: ignore
                cv="prefit",
            )
            features_with_oos = np.vstack([inscope_embeddings, oos_embeddings])
            labels_with_oos = np.hstack(
                [np.zeros(len(inscope_embeddings)), np.ones(len(oos_embeddings))]
            )
            threshold_estimator.fit(features_with_oos, labels_with_oos)
            threshold = threshold_estimator.best_threshold_ * threshold_magnifier
        else:
            distances = estimator.decision_function(inscope_embeddings)
            threshold = np.max(distances) * threshold_magnifier
        return MarginalMahalanobisDetector(mean_vector, precision, threshold)  # type: ignore
