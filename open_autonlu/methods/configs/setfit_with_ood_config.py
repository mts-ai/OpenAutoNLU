from dataclasses import dataclass
from .base_config import BaseMethodWithOODConfig
from .setfit_config import SetFitMethodConfig
from ..data_types import OodMethod


@dataclass
class SetFitOODConfig(SetFitMethodConfig, BaseMethodWithOODConfig):
    """Configuration for SetFit few-shot learning with Out-of-Distribution detection.

    Extends SetFitMethodConfig with OOD detection using Maximum Softmax Probability
    (MSP) by default. MSP uses the maximum probability from the softmax output as
    a confidence score - samples with low maximum probability are flagged as OOD.

    Attributes:
        threshold_factor: Multiplier for OOD detection threshold. Higher values
            make detection more conservative (fewer samples marked as OOD).
        ood_method: OOD detection method. Defaults to MSP (Maximum Softmax
            Probability), which is well-suited for few-shot settings.
    """

    threshold_factor: float = 1.0
    ood_method: OodMethod = OodMethod.MSP_OOD
