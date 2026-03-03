from abc import ABC, abstractmethod

import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Optional

from ..constants import TRAIN_KEY, DEV_KEY, TEST_KEY
import logging

log = logging.getLogger(__name__)


class AbstractDataProvider(ABC):
    """Abstract base class for data providers.

    Data providers handle loading and preprocessing of training and
    evaluation datasets from various file formats.
    """

    def __init__(
        self,
        train_path: str,
        dev_path: Optional[str] = None,
        test_path: Optional[str] = None,
        language: str = "en",
    ) -> None:
        """Initialize the data provider.

        Args:
            train_path: Path to training data file.
            dev_path: Optional path to development/validation data.
            test_path: Optional path to test data.
            language: "ru" or "en" for tokenization.
        """
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.language = language

    def load(self) -> DatasetDict:
        """Load and preprocess data from configured paths.

        Returns:
            DatasetDict containing train, dev, and/or test splits.
        """
        data = self._load_data()
        data = self._postprocess(data)

        return data

    @abstractmethod
    def _load_data(self) -> DatasetDict:
        raise NotImplementedError()

    def _postprocess(self, data: DatasetDict) -> DatasetDict:
        return data


class SimpleDataProvider(AbstractDataProvider):
    """Data provider for CSV files with text and label columns.

    Loads CSV files containing 'text' and 'label' columns.
    Optionally supports 'anc_label' column for anchor-based methods.
    """

    def __init__(
        self,
        train_path: str,
        num_examples: Optional[int] = None,
        dev_path: Optional[str] = None,
        test_path: Optional[str] = None,
        language: str = "en",
    ) -> None:
        """Initialize the CSV data provider.

        Args:
            train_path: Path to training CSV file.
            num_examples: Optional limit on number of training examples.
            dev_path: Optional path to development CSV.
            test_path: Optional path to test CSV.
            language: "ru" or "en".
        """
        super().__init__(train_path, dev_path, test_path, language=language)
        self.num_examples = num_examples

    def _load_data(self) -> dict[str, pd.DataFrame]:
        splits = {}
        for split_name in [TRAIN_KEY, DEV_KEY, TEST_KEY]:
            if path := getattr(self, f"{split_name}_path"):
                df = pd.read_csv(path)
                selector = ["text", "label"]
                if split_name == TRAIN_KEY:
                    if "anc_label" in df.keys():
                        selector = ["text", "label", "anc_label"]
                    if self.num_examples is not None:
                        splits[split_name] = df[selector][: self.num_examples]
                        continue
                splits[split_name] = df[selector]
        return splits

    def _postprocess(self, data: dict[str, pd.DataFrame]) -> DatasetDict:
        """
        We remove all examples with OOS_LABEL from the train split, as they are not used for training.
        We remove all examples, that have their columns 'text' or 'label' empty and warn user about removed
        examples.
        """

        for split in data.keys():
            df = data[split]
            indices = df[
                df.label.isna() | df.text.isna() | (df.text == "") | (df.label == "")
            ].index
            if len(indices):
                data[split] = df.drop(index=indices)
                log.warning(
                    f"Some rows contain empty values in {split} split. Rows with indices {indices.to_list()} were removed to avoid any runtime errors."
                )
        return DatasetDict(
            {
                split: Dataset.from_pandas(df, preserve_index=False)
                for split, df in data.items()
            }
        )
