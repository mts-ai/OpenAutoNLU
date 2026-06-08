"""LLM-based augmentation plugin.

Phase 4 keeps the existing dataset-level LLM upsampling logic intact and simply
exposes it behind the plugin namespace as a thin delegation. The heavy lifting
still lives in ``method_resolver._augment_with_llm`` (unchanged) to avoid
behavior drift; this wrapper only provides a stable, plugin-style entry point.
"""

from __future__ import annotations

from typing import Any


def llm_upsample(*args: Any, **kwargs: Any):
    """Delegate to the existing dataset-level LLM augmentation routine."""
    from ...method_resolver import _augment_with_llm  # lazy: pulls llm stack

    return _augment_with_llm(*args, **kwargs)
