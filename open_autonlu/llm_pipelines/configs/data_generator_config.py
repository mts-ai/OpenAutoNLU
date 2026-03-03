from dataclasses import dataclass
from typing import Optional


@dataclass
class DataGeneratorConfig:
    """Configuration for LLM-based data generation.

    Attributes:
        seed: Random seed for reproducibility.
        prompt_takes_batches: If True, prompts receive batched examples.
        num_retries: Number of retry attempts on API failure.
        batch_size: Number of examples to generate per API call.
    """

    seed: Optional[int] = 42
    prompt_takes_batches: Optional[bool] = True
    num_retries: Optional[int] = 3
    batch_size: Optional[int] = None


@dataclass
class ClassificationDataGenerationConfig:
    """Configuration for classification data generation.

    Attributes:
        num_generations_per_class: Number of samples to generate per class.
        num_shot: Number of example samples to include in the prompt.
    """

    num_generations_per_class: Optional[int] = 100
    num_shot: Optional[int] = 5
