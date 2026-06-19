"""Character/word-level augmentation plugin (wraps WordCharAugmentation)."""

from __future__ import annotations

from typing import List, Optional

from ...data.augmentations import WordCharAugmentation


class CharNoiseAugmenter:
    """TextAugmenter that applies char/word-level corruptions via augmentex."""

    def __init__(self, **kwargs):
        self._aug = WordCharAugmentation(**kwargs)

    def augment(self, text: str, label: Optional[str] = None, ctx=None) -> List[str]:
        return self._aug(text)
