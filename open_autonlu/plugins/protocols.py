"""Plugin interfaces for locale/domain-specific behavior.

The router never imports concrete language/domain logic; it depends only on
these protocols and resolves implementations via ``plugins.registry`` using
``TaskSpec.language`` / script. Concrete plugins (char-noise, LLM augmentation,
gibberish/tiered OOD sampling, prompts) are wired in Phase 4.
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable


@runtime_checkable
class TextAugmenter(Protocol):
    """Produce additional surface forms for a labeled example."""

    def augment(self, text: str, label: str, ctx: Any = None) -> List[str]: ...


@runtime_checkable
class OodSampler(Protocol):
    """Produce out-of-distribution examples for a given tier."""

    def sample(self, n: int, tier: str, ctx: Any = None) -> List[str]: ...


@runtime_checkable
class PromptProvider(Protocol):
    """Provide language-keyed prompts for LLM subsystems."""

    def system_prompt(self, language: str, task: str) -> str: ...


@runtime_checkable
class GibberishGenerator(Protocol):
    """Produce script-appropriate nonsensical text (very-far OOD)."""

    def generate(self, n: int, script: str = "latin") -> List[str]: ...
