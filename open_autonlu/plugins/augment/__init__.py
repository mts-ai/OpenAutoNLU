"""Augmentation plugins.

Importing this module registers the char-noise augmenter. The LLM augmenter is
a dataset-level delegation (``llm_upsample``) and is imported lazily to avoid
pulling the LLM stack at import time.
"""

from __future__ import annotations

from .. import registry
from .char_noise import CharNoiseAugmenter

registry.register("augment", "char_noise", lambda **kw: CharNoiseAugmenter(**kw))


def _llm_factory(**kw):
    from .llm import llm_upsample

    return llm_upsample


registry.register("augment", "llm", _llm_factory)

__all__ = ["CharNoiseAugmenter"]
