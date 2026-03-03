from dataclasses import dataclass
from .base_config import BaseMethodWithOODConfig
from .finetuner_config import FinetunerConfig
from ..data_types import OodMethod


@dataclass
class FinetunerOODConfig(FinetunerConfig, BaseMethodWithOODConfig):
    """Configuration for transformer finetuning with Out-of-Distribution detection.

    Extends FinetunerConfig with OOD detection using Marginal Mahalanobis distance
    by default. This method computes class-conditional Gaussian distributions in
    the embedding space and uses Mahalanobis distance to detect samples that fall
    outside known class distributions.

    This is the recommended configuration for OOD detection when sufficient
    training data is available, as Mahalanobis-based detection typically
    outperforms MSP-based methods on larger datasets.

    Attributes:
        threshold_factor: Multiplier for OOD detection threshold. Higher values
            make detection more conservative (fewer samples marked as OOD).
        ood_method: OOD detection method. Defaults to Marginal Mahalanobis,
            which estimates per-class distributions for distance-based detection.
    """

    threshold_factor: float = 1.0
    ood_method: OodMethod = OodMethod.MARGINAL_MAHALANOBIS_OOD
