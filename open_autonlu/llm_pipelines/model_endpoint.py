from typing import Any, Optional
import openai
from .configs.llm_config import LlmClientConfig


class ModelEndpointWrapper:
    """Wrapper for OpenAI-compatible LLM API endpoints.

    Provides a callable interface for making streaming chat completion requests
    to LLM APIs. Supports customizable generation parameters per-request while
    maintaining sensible defaults from configuration.

    Attributes:
        config_cls: The configuration class used for this wrapper.
        client: The OpenAI client instance.
        system_prompt: Default system prompt for conversations.
        model_name: Model identifier for API requests.
        default_temperature: Default sampling temperature.
        default_presence: Default presence penalty.
        default_frequency_penalty: Default frequency penalty.
        num_concurrent_requests: Maximum concurrent API requests allowed.

    Example:
        >>> config = LlmClientConfig(api_key="your-key", model_id="gpt-4")
        >>> llm = ModelEndpointWrapper(config)
        >>> response = llm("What is machine learning?")
        >>> print(response)
    """

    config_cls = LlmClientConfig

    def __init__(
        self,
        config: Optional[LlmClientConfig] = None,
    ) -> None:
        """Initialize the model endpoint wrapper.

        Args:
            config: Configuration for the LLM client. If None, uses default
                configuration with values from environment variables.
        """
        if config is None:
            config = LlmClientConfig()

        self.client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.system_prompt = config.default_system_prompt
        self.model_name = config.model_id
        self.default_temperature = config.default_temperature
        self.default_presence = config.default_presence_penalty
        self.default_frequency_penalty = config.default_frequency_penalty
        self.num_concurrent_requests = config.num_concurrent_requests

    def __call__(
        self,
        prompt,
        system_prompt=None,
        temperature=None,
        presence_penalty=None,
        frequency_penalty=None,
    ) -> Any:
        """Send a prompt to the LLM and return the generated response.

        Makes a streaming chat completion request and concatenates the response
        chunks into a single string.

        Args:
            prompt: The user message to send to the model.
            system_prompt: Override the default system prompt for this request.
            temperature: Sampling temperature (0.0-1.0). Higher values produce
                more random outputs.
            presence_penalty: Penalty for repeating topics already mentioned.
            frequency_penalty: Penalty for repeating the same tokens.

        Returns:
            The complete generated response as a string.
        """
        params = {
            "temperature": (
                self.default_temperature if temperature is None else temperature
            ),
            "presence_penalty": (
                self.default_presence if presence_penalty is None else presence_penalty
            ),
            "frequency_penalty": (
                self.default_frequency_penalty
                if frequency_penalty is None
                else frequency_penalty
            ),
        }

        stream = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt or self.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
            model=self.model_name,
            **params,
        )
        result = []
        for chunk in stream:
            result.append(chunk.choices[0].delta.content or "")
        return "".join(result)
