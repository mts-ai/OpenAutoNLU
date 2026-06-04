from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..constants import DEFAULT_MODEL_BY_LANGUAGE
from ..data_types import OodMethod


@dataclass
class BaseMethodConfig:
    """Base configuration for all training methods.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path.
            If None, resolved from language by DEFAULT_MODEL_BY_LANGUAGE.
        language: Language code (e.g. "en", "ru"). If model_name_or_path is not set,
            a default model is used only for "en" and "ru"; for other languages
            you must set model_name_or_path.
        tokenizer_kwargs: Additional arguments passed to the tokenizer.
        max_seq_length: Maximum sequence length for tokenization.
        dev_fraction: Fraction of training data to hold out as a dev set
            for early stopping and evaluation. Must be in (0, 1).
    """

    model_name_or_path: Optional[str] = None
    language: str = "en"
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    max_seq_length: int = 512
    dev_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.model_name_or_path is None:
            if self.language not in DEFAULT_MODEL_BY_LANGUAGE:
                raise ValueError(
                    f"Language '{self.language}' has no default model. "
                    f"Set 'model_name_or_path' in config_overrides to a checkpoint that supports this language "
                    f"Default models are only available for: {', '.join(DEFAULT_MODEL_BY_LANGUAGE)}."
                )
            self.model_name_or_path = DEFAULT_MODEL_BY_LANGUAGE[self.language]


@dataclass
class BaseMethodWithOODConfig(BaseMethodConfig):
    """Base configuration with OOD detection parameters.

    Extends BaseMethodConfig with threshold configuration for
    OOD detection methods (Mahalanobis, MSP).

    Attributes:
        ood_method: Out-of-distribution detection method.
            - AUTO: Automatically select based on method type
            - NONE: No OOD detection
            - LOGIT: Use logit-based OOD (adds OOS class)
            - MAHALANOBIS: Use Mahalanobis distance
            - MSP: Use Maximum Softmax Probability
        threshold_factor: Multiplier for OOD detection threshold.
            Higher values make detection more conservative (fewer OOD).
    """

    ood_method: OodMethod = OodMethod.AUTO
    threshold_factor: float = 1.0
