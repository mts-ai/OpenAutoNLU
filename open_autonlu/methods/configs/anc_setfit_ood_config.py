from dataclasses import dataclass
from .base_config import BaseMethodWithOODConfig
from .anc_setfit_config import AncSetFitConfig
from ..data_types import OodMethod


@dataclass
class AncSetFitOODConfig(AncSetFitConfig, BaseMethodWithOODConfig):
    """Configuration for Anchor-based SetFit with Out-of-Distribution detection.

    Extends AncSetFitConfig with OOD detection capabilities using Maximum Softmax
    Probability (MSP) by default. Combines anchor-based contrastive learning with
    the ability to detect inputs that don't belong to any known class.

    Warning:
        OOD detection does not perform well in few-shot settings with this method.
        Based on benchmarks, the OOD detection metrics are significantly degraded
        when training data is limited. For few-shot scenarios (under 5 samples per class),
        OOD detection is not turned off by default.

    Attributes:
        threshold_factor: Multiplier for OOD detection threshold. Higher values
            make detection more conservative (fewer samples marked as OOD).
        ood_method: OOD detection method to use. Defaults to MSP (Maximum Softmax
            Probability).
    """

    threshold_factor: float = 1.0
    ood_method: OodMethod = OodMethod.MSP_OOD
