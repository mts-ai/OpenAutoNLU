import concurrent.futures
import logging
import threading
from typing import Any, Dict, Optional

import datasets
import pandas as pd
import tqdm
import outlines
from openai import APIConnectionError

from .configs.data_generator_config import ClassificationDataGenerationConfig
from .data_generation import DataGenerator
from .model_endpoint import ModelEndpointWrapper
from .prompts import get_prompts

logging.basicConfig()
log = logging.getLogger(__name__)


class DataGenerationPipeline:
    """Pipeline for generating synthetic data using LLM.

    Generates classification training data by prompting an LLM with
    reference examples from each class. Supports concurrent API requests
    and domain-aware generation.

    Example:
        >>> pipeline = DataGenerationPipeline(
        ...     reference_data=train_df,
        ...     config_overrides={"LlmClientConfig": {"api_key": "..."}}
        ... )
        >>> dataset = pipeline.generate({"num_generations_per_class": 50})
    """

    data_generator_cls = DataGenerator
    client_cls = ModelEndpointWrapper

    def __init__(
        self,
        reference_data: pd.DataFrame,
        text_column: Optional[str] = "text",
        label_column: Optional[str] = "label",
        config_overrides: Optional[Dict[str, Any]] = None,
        prompt: Optional[outlines.prompts.Prompt] = None,
        domain_desc: Optional[str] = None,
        label_descriptions: Optional[Dict[str, str]] = None,
        language: str = "en",
    ):
        """Initialize the data generation pipeline.

        Args:
            reference_data: DataFrame with reference examples for each class.
            text_column: Column name containing text data.
            label_column: Column name containing class labels.
            config_overrides: Configuration overrides for LlmClientConfig
                and DataGeneratorConfig.
            prompt: Custom prompt template. Defaults to generate_texts.
            domain_desc: Optional domain description for context.
            label_descriptions: Optional dict mapping labels to descriptions.
        """
        self.config_overrides = config_overrides
        self.reference_data = reference_data
        self.text_column = text_column
        self.label_column = label_column
        self.domain_desc = domain_desc
        self.label_descriptions = label_descriptions or {}

        client_config = self._resolve_config_overrides(self.client_cls)
        client = self.client_cls(config=client_config)

        if prompt is None:
            prompt = get_prompts(language).generate_texts

        data_generator_config = self._resolve_config_overrides(self.data_generator_cls)
        self.data_generator = self.data_generator_cls(
            prompt, client, config=data_generator_config
        )

    # TODO: move method to separate Mixin
    def _resolve_config_overrides(self, method_cls):
        if self.config_overrides is None:
            return method_cls.config_cls()

        possible_keys = [method_cls.__name__]
        config_cls = getattr(method_cls, "config_cls", None)
        if config_cls is not None:
            possible_keys.append(config_cls.__name__)

        method_kwargs = None
        for key in possible_keys:
            if key in self.config_overrides:
                method_kwargs = self.config_overrides.get(key)
                break

        if method_kwargs is None:
            method_kwargs = {
                k: v
                for k, v in self.config_overrides.items()
                if hasattr(method_cls.config_cls, k)
            }
        return method_cls.config_cls(**method_kwargs)

    def _execute_api_call(self, texts, topic, n_generations, stop_event):
        if stop_event.is_set():
            return None
        result = self.data_generator(
            texts=texts,
            topic=topic,
            data_size=n_generations,  # how many samples should we generate
            batch_size=len(texts),  # desired number of examples for prompt
        )
        return result

    def generate(self, generation_kwargs: Optional[Dict] = None):
        """
        Generates classification dataset based on samples from reference data.

        Parameters:
        generation_kwargs: Parameters supported by ClassificationDataGenerationConfig.

        Returns:
        Dataset: Dataset of generated samples or None.
        """
        generation_config = (
            ClassificationDataGenerationConfig(**generation_kwargs)
            if generation_kwargs
            else ClassificationDataGenerationConfig()
        )

        new_data: Dict[str, list] = {"text": [], "label": []}

        topic_to_samples = {}
        for topic, group in self.reference_data.groupby(self.label_column):
            # Use all available examples for sampling per batch
            all_examples = group[self.text_column].to_list()
            topic_to_samples[topic] = all_examples

        num_classes = len(topic_to_samples)
        max_concurrent_from_config = self.data_generator.client.num_concurrent_requests

        max_workers_limit = 10

        if max_concurrent_from_config and max_concurrent_from_config > 1:
            num_concurrent_requests = min(
                max(max_concurrent_from_config, num_classes), max_workers_limit
            )
        else:
            num_concurrent_requests = min(num_classes, max_workers_limit)

        def get_adaptive_batch_size(generated_size):
            if generated_size <= 10:
                return 5
            else:
                return 10

        generation_batch_size = get_adaptive_batch_size(
            generation_config.num_generations_per_class
        )

        stop_event = threading.Event()
        pbar = tqdm.tqdm(total=len(topic_to_samples.keys()), desc="Generating data")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_concurrent_requests
        ) as executor:
            future_to_topic = {
                executor.submit(
                    self.data_generator,
                    texts=[],
                    topic=topic,
                    data_size=generation_config.num_generations_per_class,
                    batch_size=generation_batch_size,
                    example_texts=all_examples,
                    examples_per_batch=generation_config.num_shot,
                    stop_event=stop_event,
                    domain_desc=self.domain_desc,
                    label_desc=self.label_descriptions.get(topic),
                ): topic
                for topic, all_examples in topic_to_samples.items()
            }
            for future in concurrent.futures.as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    new_texts = future.result()
                    if new_texts is not None:
                        new_data["text"].extend(new_texts.text)
                        new_data["label"].extend([topic] * len(new_texts.text))
                except APIConnectionError as e:
                    log.error(
                        "Unable to connect to LLM. Exiting generation pipeline.",
                        exc_info=e,
                    )
                    stop_event.set()
                    break
                finally:
                    pbar.update(1)
            pbar.close()

        if len(new_data["text"]) == 0:
            return None
        return datasets.Dataset.from_dict(new_data)
