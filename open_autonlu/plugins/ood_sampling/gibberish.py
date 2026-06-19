"""Gibberish OOD sampler plugin.

Thin wrapper around the existing ``GibberishDatasetGenerator`` so trainers stop
importing it directly. The default (no language) reproduces the *legacy*
generator byte-for-byte for a given seed -- this is the backward-compatible
"very-far" OOD source used by ``_generate_synthetic_oos``.
"""

from __future__ import annotations

from typing import List, Optional

from ...methods.utils.random_sentence_generator import (
    GibberishDatasetGenerator,
    GibberishDatasetGeneratorEN,
)


class GibberishOodSampler:
    """Produces nonsensical (very-far OOD) text.

    Args:
        seed: RNG seed (parity with the legacy ``GibberishDatasetGenerator(seed)``).
        language: ``"en"`` selects the Latin-script generator; any other value
            (including None) uses the legacy generator for backward compatibility.
        config: optional generator config override.
    """

    def __init__(self, seed: int = 0, language: Optional[str] = None, config=None):
        self.seed = seed
        self.language = language
        if language == "en":
            self._gen = GibberishDatasetGeneratorEN(seed=seed, config=config)
        else:
            self._gen = GibberishDatasetGenerator(seed=seed, config=config)

    def sample(self, n: int, tier: str = "very_far", ctx=None) -> List[str]:
        """Return ``n`` gibberish utterances (tier is ignored; always very-far)."""
        return self._gen(num_rows=n)
