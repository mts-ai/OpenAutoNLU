"""Plugin system for language/domain-specific behavior (Phase 0 scaffolding)."""

from . import registry
from .protocols import (
    GibberishGenerator,
    OodSampler,
    PromptProvider,
    TextAugmenter,
)

__all__ = [
    "registry",
    "TextAugmenter",
    "OodSampler",
    "PromptProvider",
    "GibberishGenerator",
]
