"""OOD sampling plugins + resolver.

Importing this module registers the built-in samplers ("gibberish", "tiered")
in the plugin registry. ``resolve_ood_sampler`` is the single entry point used
by trainers to obtain an :class:`OodSampler`, defaulting to legacy gibberish.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import registry
from .gibberish import GibberishOodSampler
from .tiered import TieredOodSampler

# Register built-ins (idempotent on re-import).
registry.register("ood_sampling", "gibberish", lambda seed=0, **kw: GibberishOodSampler(seed=seed, **kw))
registry.register("ood_sampling", "tiered", lambda seed=0, **kw: TieredOodSampler(seed=seed, **kw))


def resolve_ood_sampler(spec: Optional[Any], seed: int = 0):
    """Return an OOD sampler from a spec.

    Args:
        spec: one of
            - None -> legacy gibberish sampler (backward compatible);
            - an object with a ``sample`` method (an OodSampler instance);
            - a registered plugin name, e.g. "gibberish" / "tiered".
        seed: seed passed to plugin factories.
    """
    if spec is None:
        return GibberishOodSampler(seed=seed)
    if isinstance(spec, str):
        return registry.get("ood_sampling", spec)(seed=seed)
    if hasattr(spec, "sample"):
        return spec
    raise TypeError(
        f"Unsupported ood_sampler spec: {spec!r}. Use None, a plugin name, "
        f"or an object with a .sample(n) method."
    )


__all__ = [
    "GibberishOodSampler",
    "TieredOodSampler",
    "resolve_ood_sampler",
]
