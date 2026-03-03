from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import logging

from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
from accelerate.utils import find_executable_batch_size
from open_autonlu.progress_reporter import ProgressReporter
from open_autonlu.methods.constants import BATCH_SIZE


from .data_types import (
    ClassificationEvaluationResult,
    NERFormat,
    RetrievalEvaluationResult,
    TokenClassifierOutput,
    TextClassifierOutput,
    TrainingArtifactInfo,
    SaveFormat,
)
from .configs.base_config import BaseMethodConfig

log = logging.getLogger(__name__)


class Method(ABC):
    @abstractmethod
    def __init__(
        self,
        config: Optional[BaseMethodConfig] = None,
        progress_reporter: Optional[ProgressReporter] = None,
    ) -> None: ...

    @abstractmethod
    def train(self, dataset: Dataset) -> TrainingArtifactInfo:
        """
        Takes in a dataset, that contains train split
        with texts and labels.
        """
        ...

    @abstractmethod
    def test(
        self, dataset: Dataset
    ) -> Union[ClassificationEvaluationResult, RetrievalEvaluationResult]:
        """
        Takes in a dataset, that contains test split
        with texts and labels. Runs inference on texts and counts needed metrics.
        Returns a sklearn classification_report
        """
        ...

    @abstractmethod
    def save(
        self, path_to_package: str, checkpoint_format: SaveFormat = SaveFormat.TORCH
    ) -> None:
        """
        Serializes all the objects and files to a designated path for future recreation.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path_to_package: str):
        """
        Using files from path_to_package reconstructs its own exemplar
        """
        ...

    @staticmethod
    def _get_label_mappings(labels: List[str]):
        """
        Gets the label mappings from a list of labels
        """
        label2id = {label: l_id for l_id, label in enumerate(labels)}
        id2label = {l_id: label for l_id, label in enumerate(labels)}
        return label2id, id2label


class SequenceClassificationMethod(Method):
    @property
    def label_list(self) -> List[str]:
        return [
            value
            for _, value in sorted(self.id2label.items(), key=lambda item: item[0])
        ]

    @staticmethod
    def _extract_scores(logits) -> List[float]:
        return logits.max(-1).tolist()

    @staticmethod
    def _extract_labels(logits, id2label) -> List[str]:
        human_readable_labels = []
        for label_id in logits.argmax(-1):
            human_readable_labels.append(id2label[label_id])
        return human_readable_labels

    @abstractmethod
    def test(
        self, dataset: Dataset, inscope_only: bool = False
    ) -> ClassificationEvaluationResult:
        """
        Takes in a dataset, that contains test split
        with texts and labels. Runs inference on texts and counts needed metrics.
        Returns a sklearn classification_report.

        Args:
            dataset: Dataset with test data
            inscope_only: If True, exclude outOfScope labels from inference predictions,
                         replacing them with the most probable in-scope label
        """
        ...

    @abstractmethod
    def _predict(
        self,
        batch_size: int,
        texts: List[str],
    ) -> List[TextClassifierOutput]:
        """
        Takes in a collection of texts and runs inference on it. Returns a collection of string labels.
        """
        ...

    def predict(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        auto_find_batch_size: bool = False,
        return_all_hypotheses: bool = False,
    ) -> List[TextClassifierOutput]:
        """
        Takes in a collection of texts and runs inference on it. Returns a collection of string labels.
        """
        if return_all_hypotheses:
            log.debug(
                "return_all_hypotheses is not supported for torch-based inference for now and will be ignored."
            )
        if auto_find_batch_size:
            return find_executable_batch_size(
                self._predict, starting_batch_size=batch_size
            )(texts=texts)

        return self._predict(batch_size, texts)


class TokenClassificationMethod(Method):
    """
    This class of methods consumes (through .train)
    and yields (through .predict) the following data structures accordingly:
    .train(Dataset([{"text": "the text to analyze", "labels": ["B-entity", "I-entity", "O", "O"]}]))
    .predict("the text to analyze" or ["the text to analyze", ...]) -> List[TokenClassifierOutput]
    """

    @abstractmethod
    def predict(
        self,
        texts: Union[List[str], str],
        batch_size: int,
        output_format: NERFormat = NERFormat.OFFSETS,
    ) -> List[TokenClassifierOutput]:
        """
        Takes in a collection of texts and runs inference on it. Returns a collection of objects of type TokenClassifierOutput.
        """
        ...

    def train_test_split(
        self,
        dataset: Dataset,
        entity_names: List[str],
        test_size: float = 0.1,
        seed: int = 42,
        bio_tags_key: str = "labels",
    ) -> DatasetDict:
        indices = np.arange(dataset.num_rows, dtype=int)
        entity2id = {name: idx for idx, name in enumerate(entity_names)}
        bag_of_labels = self._get_bag_of_labels(dataset[bio_tags_key], entity2id)
        if bag_of_labels.shape[1] == 1:
            train_indices, test_indices = train_test_split(
                indices, test_size=test_size, random_state=seed, stratify=bag_of_labels
            )
        else:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            train_indices, test_indices = next(splitter.split(indices, bag_of_labels))
        return DatasetDict(
            {
                "train": dataset.select(train_indices),
                "test": dataset.select(test_indices),
            }
        )

    def _get_bag_of_labels(
        self, labels: List[List[str]], label2id: Dict[str, int]
    ) -> np.array:
        outputs = []
        for bio_tags in labels:
            label_names = list({tag[2:] for tag in bio_tags if tag != "O"})
            label_ids = [label2id[label] for label in label_names]
            bag_of_labels = np.zeros(len(label2id), dtype=int)
            bag_of_labels[label_ids] = 1
            outputs.append(bag_of_labels)
        return np.vstack(outputs)
