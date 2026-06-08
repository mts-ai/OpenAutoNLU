"""Default, language-keyed prompt provider.

Delegates to the existing ``llm_pipelines.prompts.get_prompts`` so prompt text
stays in one place; the plugin only provides the protocol-shaped access point so
the router/LLM subsystems depend on an interface, not on Russian/English string
constants directly.
"""

from __future__ import annotations

from typing import Any


class DefaultPromptProvider:
    """PromptProvider backed by ``get_prompts(language)``."""

    def prompts(self, language: str = "en") -> Any:
        from ...llm_pipelines.prompts import get_prompts

        return get_prompts(language)

    def system_prompt(self, language: str, task: str = "") -> str:
        ns = self.prompts(language)
        return getattr(ns, "system_prompt", "") or ""
