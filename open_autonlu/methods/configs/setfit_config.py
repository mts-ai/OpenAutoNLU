from dataclasses import dataclass
from .base_config import BaseMethodConfig
from ..data_types import OodMethod


@dataclass
class SetFitMethodConfig(BaseMethodConfig):
    """Configuration for SetFit few-shot learning method.

    Attributes:
        num_iterations: Number of training iterations for contrastive learning.
        body_lr: Learning rate for the sentence transformer body.
        seed: Random seed for reproducibility.
        batch_size: Training batch size.
    """

    num_iterations: int = 20
    body_lr: float = 1e-5
    seed: int = 1223
    batch_size: int = 16
    ood_method: OodMethod = OodMethod.NONE
