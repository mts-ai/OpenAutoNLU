from dataclasses import dataclass
from typing import Optional
import os
from ..prompts import DEFAULT_SYSTEM_PROMPT


@dataclass
class LlmClientConfig:
    """Configuration for LLM API client.

    Used for data augmentation and synthetic test generation pipelines.
    Defaults are read from environment variables if available.

    Attributes:
        api_key: API key for authentication. Reads from MODEL_API_KEY env var.
        base_url: Base URL of the LLM API endpoint. Reads from BASE_URL env var.
        model_id: Model identifier to use. Reads from MODEL_ID env var.
        default_system_prompt: System prompt for the LLM.
        default_temperature: Sampling temperature (0.0-1.0).
        default_presence_penalty: Presence penalty for generation.
        default_frequency_penalty: Frequency penalty for generation.
        num_concurrent_requests: Maximum concurrent API requests.
    """

    api_key: str = os.environ.get("MODEL_API_KEY", "")
    base_url: str = os.environ.get("BASE_URL", "")
    model_id: str = os.environ.get("MODEL_ID", "")
    default_system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    default_temperature: Optional[float] = 0.5
    default_presence_penalty: Optional[float] = 0.0
    default_frequency_penalty: Optional[float] = 0.1
    num_concurrent_requests: Optional[int] = 1
