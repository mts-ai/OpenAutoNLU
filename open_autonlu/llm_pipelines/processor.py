import re
import random
from typing import Any, Iterable, Iterator, List, Union, Optional, Callable
import mdpd
import numpy as np
import outlines
import pandas as pd
import tqdm
from .model_endpoint import ModelEndpointWrapper
from openai import APIConnectionError, BadRequestError
import threading
import logging

log = logging.getLogger(__name__)


class WrongFormatError(Exception):
    pass


class Processor:
    def __init__(
        self,
        prompt: outlines.prompts.Prompt,
        client: ModelEndpointWrapper,
        seed: int,
        prompt_takes_batches: bool,
        num_retries: int = 3,
        batch_size: int = None,
    ):
        self.prompt = prompt
        self.batch_size = batch_size
        self.client = client
        self.seed = seed
        self.prompt_takes_batches = prompt_takes_batches
        self.num_retries = num_retries

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Process texts either in batches or individually, handling retries for wrongly formatted responses.

        Parameters:
        *args: Variable length argument list.
        **kwds: Arbitrary keyword arguments.

        Returns:
        Any: The aggregated or processed responses.
        """
        if self.prompt_takes_batches:
            return self._process_batches(**kwds)
        else:
            return self._process_individual(**kwds)

    def _api_call(
        self,
        prompt: Union[str, outlines.prompt],
        postprocessor: Optional[Callable] = lambda x: x,
        stop_flag: Optional[threading.Event] = None,
    ) -> Any:
        """
        Processes a request to the model's API and handles any errors that occur.

        Args:
            prompt (str): The instruction prompt for the generative model.
            postprocessor (Callable): Function to process response from generative model.
            stop_flag (threading.Event): A flag used to stop API calls running in different threads.

        Returns:
            Any: Post-processed response from the generative model or None in case of failure.
        """
        for trial in range(self.num_retries):
            if stop_flag is not None and stop_flag.is_set():
                return None
            try:
                response = self.client(prompt)
                response = postprocessor(response)
                return response
            except WrongFormatError:
                log.info(
                    f"Trial {trial + 1}. LLM has output wrongly formatted text, retrying."
                )
            except APIConnectionError as e:
                raise e
            except BadRequestError as e:
                log.error("Bad request. Skipping this request.", exc_info=e)
                return None
            except Exception as e:
                log.error(
                    f"{e}\nTrial {trial + 1}. An exception occurred while calling LLM, retrying.",
                    exc_info=e,
                )
        else:
            log.info("Out of retries, skipping request.")
            return None

    def _process_batches(self, **kwds: Any) -> Any:
        """
        Process texts in batches, handling retries for wrongly formatted responses.

        Parameters:
        **kwds: Arbitrary keyword arguments.

        Returns:
        Any: The aggregated batches.
        """
        postprocessed_batches = []
        texts = kwds.pop("texts")
        batch_size = kwds.pop("batch_size", self.batch_size)
        stop_event = kwds.pop("stop_event", None)
        example_texts = kwds.pop("example_texts", None)
        examples_per_batch = kwds.pop("examples_per_batch", None)

        assert (
            batch_size is not None
        ), "You have to specify batch size either in call args, or in constructor args"

        if example_texts is not None and examples_per_batch is not None:
            data_size = kwds.get("data_size", batch_size)
            num_batches = (data_size + batch_size - 1) // batch_size

            rng = random.Random(self.seed)

            for batch_idx in range(num_batches):
                # Calculate how many texts to generate in this batch
                remaining = data_size - (batch_idx * batch_size)
                current_batch_size = min(batch_size, remaining)
                if current_batch_size <= 0:
                    break

                num_examples = min(examples_per_batch, len(example_texts))
                batch_examples = rng.sample(example_texts, num_examples)
                rng.shuffle(batch_examples)

                # Pass current_batch_size instead of total data_size
                batch_kwds = kwds.copy()
                batch_kwds["data_size"] = current_batch_size

                response = self._api_call(
                    self.prompt(texts=batch_examples, **batch_kwds),
                    postprocessor=self._postprocess_one_batch,
                    stop_flag=stop_event,
                )
                if response is not None:
                    postprocessed_batches.append(response)
        else:
            for batch in self._window(texts, batch_size, self.seed):
                response = self._api_call(
                    self.prompt(texts=batch, **kwds),
                    postprocessor=self._postprocess_one_batch,
                    stop_flag=stop_event,
                )
                if response is not None:
                    postprocessed_batches.append(response)

        if len(postprocessed_batches) == 0:
            return None
        else:
            return self._aggregate_batches(postprocessed_batches)

    def _process_individual(self, **kwds: Any) -> Any:
        """
        Process texts individually.

        Parameters:
        **kwds: Arbitrary keyword arguments.

        Returns:
        Any: The aggregated individual responses.
        """
        postprocessed_responses = []
        texts = kwds.pop("texts")
        stop_event = kwds.pop("stop_event", None)

        for text in tqdm.tqdm(texts, desc="Processing texts"):
            response = self._api_call(
                self.prompt(text=text, **kwds),
                postprocessor=self._postprocess_one,
                stop_flag=stop_event,
            )
            if response is not None:
                postprocessed_responses.append(response)
        if len(postprocessed_responses) == 0:
            return None
        else:
            return self._aggregate_single_responses(postprocessed_responses, texts)

    @staticmethod
    def _window(
        texts: Iterable[str], subset_size: int, seed: int
    ) -> Iterator[List[str]]:
        """
        Divides an iterable of text strings into subsets of a specified size.

        The method uses a random number generator to ensure the subsets are
        distributed randomly. It yields each subset as a list of strings.

        Parameters:
        -----------
        texts : Iterable[str]
            An iterable of text strings that need to be divided into subsets.
        subset_size : int
            The size of each subset. This value must be less than or equal to
            the number of text samples provided.
        seed : int
            A seed value for the random number generator to ensure reproducibility
            of the subset distribution.

        Returns:
        --------
        Iterator[List[str]]
            An iterator that yields lists of strings, where each list represents
            a subset of the input texts.

        Raises:
        -------
        AssertionError
            If the `subset_size` is greater than the number of text samples provided.
        """
        df = pd.DataFrame({"text": texts})
        # TODO: process this assertion error in self._process_batches() and self._process_individual()
        assert (
            subset_size <= df.shape[0]
        ), f"Choose a smaller window, your data is only {df.shape[0]} samples"
        rng = np.random.default_rng(seed=seed)
        n_samples = df.shape[0] // subset_size
        df["subset"] = rng.choice(n_samples, df.shape[0])
        for _, group in df.groupby("subset"):
            yield group.text.tolist()

    @staticmethod
    def _postprocess_one_batch(text_response: str):
        # response should contain only a table
        raw_table = re.search(r"\|.*\|", text_response, re.S)
        if raw_table:
            return mdpd.from_md(raw_table.group(0))
        else:
            raise WrongFormatError(text_response)

    def _aggregate_batches(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError()

    def _aggregate_single_responses(
        self, responses: List[str], texts: List[str]
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def _postprocess_one(self, response: List[str]) -> Any:
        raise NotImplementedError()
