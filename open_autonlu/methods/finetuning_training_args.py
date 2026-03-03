import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy


class OptunaSuggestionDtype(enum.Enum):
    """
    Enum class for Optuna suggestion data types.
    """

    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"


@dataclass
class OptimizableHyperParameter:
    """
    Data class for representing an optimizable hyperparameter.

    Attributes:
        range (Union[tuple[float, float], tuple[int, int], List[Any]]): The range of values for the hyperparameter.
        dtype (OptunaSuggestionDtype): The data type of the hyperparameter.
        optuna_kwargs (Dict[str, Any]): Additional keyword arguments for Optuna.
    """

    range: Union[tuple[float, float], tuple[int, int], List[Any]]
    dtype: OptunaSuggestionDtype
    optuna_kwargs: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class FinetuningTrainingArguments(TrainingArguments):
    auto_find_batch_size: bool = field(
        default=True,
        metadata={
            "help": "Whether to find a batch size that will fit into memory automatically through exponential decay, "
            "avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)"
        },
    )
    output_dir: str = field(
        default="finetuner",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    num_train_epochs: float = field(
        default=10, metadata={"help": "Total number of training epochs to perform."}
    )
    weight_decay: float = field(
        default=1e-3, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    seed: int = field(default=1223)
    hyp_search_space: Dict[str, OptimizableHyperParameter] = field(
        default=None,
        metadata={"help": "A list of OptimizableHyperParameter"},
    )
    hyp_search_defaults: Dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 32,
        },
        metadata={"help": "Default hyperparameters."},
    )

    def configure_for_early_stopping(self):
        self.metric_for_best_model = "f1"
        self.load_best_model_at_end = True
        self.eval_steps = 0.1 / self.num_train_epochs
        self.save_steps = 0.1 / self.num_train_epochs
        self.per_device_eval_batch_size = self.per_device_train_batch_size * 2
        self.eval_strategy = IntervalStrategy.STEPS
        self.save_strategy = IntervalStrategy.STEPS
        self.save_total_limit = 1
        self.greater_is_better = True
