from dataclasses import dataclass, field
from .base_config import BaseMethodConfig
from ..data_types import OodMethod
from ..finetuning_training_args import FinetuningTrainingArguments


@dataclass
class FinetunerConfig(BaseMethodConfig):
    """Configuration for transformer finetuning method.

    Attributes:
        training_arguments: HuggingFace training arguments with
            hyperparameter search space configuration.
        num_hpo_trials: Number of Optuna trials for hyperparameter
            optimization. Set to 0 to disable HPO.
    """

    training_arguments: FinetuningTrainingArguments = field(
        default_factory=FinetuningTrainingArguments  # type: ignore
    )
    num_hpo_trials: int = 10
    ood_method: OodMethod = OodMethod.NONE
