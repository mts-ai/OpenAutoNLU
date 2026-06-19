"""Prompt-provider plugins."""

from __future__ import annotations

from .. import registry
from .default import DefaultPromptProvider

registry.register("prompts", "default", lambda **kw: DefaultPromptProvider(**kw))

__all__ = ["DefaultPromptProvider"]
