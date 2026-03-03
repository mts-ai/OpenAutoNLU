from dataclasses import dataclass
from .finetuner_config import FinetunerConfig


@dataclass
class TokenClassificationFinetunerConfig(FinetunerConfig):
    """Configuration for token classification (NER) finetuning."""

    pass
