import enum
from dataclasses import dataclass
from typing import Iterable, Optional
from datasets import DatasetDict


class TaskType(enum.Enum):
    MULTICLASS = "multiclass_classification"
    MULTILABEL = "multilabel_classification"
    NER = "token_classification"


@dataclass
class DatasetEvaluatorOutput:
    splits_filtered: DatasetDict
    indices_to_remove: Iterable[int]
    aggregated_report_path: Optional[str]
    reports_by_method_folder: Optional[str]
