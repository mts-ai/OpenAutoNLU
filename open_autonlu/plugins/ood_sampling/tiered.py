"""Tiered OOD sampler (close / mid / far / very-far).

Implements the OOD taxonomy described in the paper's evaluation appendix:

- close: held-out classes from the same macro-category (needs hierarchy).
- mid:   held-out classes from the same dataset.
- far:   examples from a different (related) dataset.
- very_far: synthetic gibberish.

close/mid/far require an injected pool of candidate utterances (the router/
benchmark supplies them via ``ctx``); very_far is self-contained via the
gibberish sampler. p95 sizing and the 1:2 OOD:ID calibration ratio are exposed
as helpers so the benchmark harness and the trainer share one implementation.

This is a minimal, dependency-light port -- it does NOT import any benchmarking
package. Cross-corpus pools are passed in, not loaded here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .gibberish import GibberishOodSampler

VALID_TIERS = ("close", "mid", "far", "very_far")


def p95_budget(class_sizes: Sequence[int]) -> int:
    """OOD test budget = 95th percentile of in-distribution class sizes."""
    if not class_sizes:
        return 0
    return int(np.percentile(np.asarray(class_sizes), 95))


def split_equally(total: int, tiers: Sequence[str]) -> Dict[str, int]:
    """Split a budget as evenly as possible across the given tiers."""
    if not tiers:
        return {}
    base, rem = divmod(total, len(tiers))
    out = {}
    for i, t in enumerate(tiers):
        out[t] = base + (1 if i < rem else 0)
    return out


class TieredOodSampler:
    """Samples OOD text per semantic-distance tier.

    Args:
        seed: RNG seed.
        pools: mapping tier -> list of candidate utterances for close/mid/far.
            Missing tiers fall back to whatever is available; very_far never
            needs a pool.
        language: passed through to the gibberish sampler.
    """

    def __init__(
        self,
        seed: int = 0,
        pools: Optional[Dict[str, Sequence[str]]] = None,
        language: Optional[str] = None,
    ):
        self.seed = seed
        self.pools = {k: list(v) for k, v in (pools or {}).items()}
        self._rng = np.random.default_rng(seed)
        self._gibberish = GibberishOodSampler(seed=seed, language=language)

    def sample(self, n: int, tier: str = "very_far", ctx=None) -> List[str]:
        if tier not in VALID_TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Valid: {VALID_TIERS}")
        if tier == "very_far":
            return self._gibberish.sample(n)
        pool = self.pools.get(tier, [])
        if not pool:
            raise ValueError(
                f"No pool provided for tier '{tier}'. Pass it via pools=..."
            )
        if len(pool) >= n:
            idx = self._rng.choice(len(pool), size=n, replace=False)
        else:
            idx = self._rng.choice(len(pool), size=n, replace=True)
        return [pool[i] for i in idx]

    def sample_mixed(self, total: int, tiers: Sequence[str] = VALID_TIERS) -> Dict[str, List[str]]:
        """Sample ``total`` examples split equally across ``tiers``."""
        usable = [t for t in tiers if t == "very_far" or self.pools.get(t)]
        budget = split_equally(total, usable)
        return {t: self.sample(k, tier=t) for t, k in budget.items() if k > 0}
