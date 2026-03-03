from typing import List, Optional
import outlines
import pandas as pd

from .model_endpoint import ModelEndpointWrapper
from .processor import Processor
from .configs.data_generator_config import DataGeneratorConfig


class DataGenerator(Processor):
    config_cls = DataGeneratorConfig

    def __init__(
        self,
        prompt: outlines.prompts.Prompt,
        client: ModelEndpointWrapper,
        config: Optional[DataGeneratorConfig] = None,
    ):
        if config is None:
            config = DataGeneratorConfig()
        super().__init__(prompt=prompt, client=client, **config.__dict__)

    def _aggregate_batches(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate a list of DataFrame batches into a single DataFrame, renaming columns and removing duplicates.

        Parameters:
        batches (List[pd.DataFrame]): A list of DataFrame batches to be aggregated.

        Returns:
        pd.DataFrame: A single aggregated DataFrame with duplicates removed.
        """
        all_dfs = []
        for df in batches:
            df = df.rename({"Text": "text"}, axis=1)
            all_dfs.append(df)
        aggregated = pd.concat(all_dfs, ignore_index=True).drop_duplicates("text")
        return aggregated
