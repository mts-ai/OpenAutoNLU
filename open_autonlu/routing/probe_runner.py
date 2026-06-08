"""ProbeRunner: cheap empirical validation of candidate recipes.

The default probe is a *frozen-capability* probe: it grounds a candidate's
expected in-scope quality in the encoder's separability / linear-head ceiling on
the current data (computed once, shared across candidates). This is the cheap,
honest signal -- it measures the ``(data, model)`` pair.

A heavier ``microtrain`` probe (1-2 epochs of the real Method) is left as an
injectable strategy: pass a custom ``probe_fn`` to :class:`ProbeRunner`. Tests
inject deterministic probes to exercise the scorer/compiler without any encoder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .recipe import Recipe


@dataclass
class ProbeContext:
    """Everything a probe needs about the current routing problem."""

    dataset: Any = None
    profile: Any = None
    capability: Any = None
    model_config: Any = None


@dataclass
class ProbeResult:
    """Outcome of probing a single recipe."""

    recipe_id: str
    in_scope_f1: Optional[float]
    ood_score: Optional[float] = None
    wall_time_s: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "in_scope_f1": self.in_scope_f1,
            "ood_score": self.ood_score,
            "wall_time_s": round(self.wall_time_s, 4),
            "notes": self.notes,
        }


ProbeFn = Callable[[Recipe, ProbeContext], ProbeResult]


def frozen_capability_probe(recipe: Recipe, ctx: ProbeContext) -> ProbeResult:
    """Default probe: read in-scope quality from the CapabilityProfile.

    Recipe-independent on the empirical axis (separability is a property of the
    data+encoder); the PlanScorer differentiates recipes via regime fit and cost.
    Returns ``in_scope_f1=None`` when no capability probe is available.
    """
    start = time.perf_counter()
    cap = ctx.capability
    f1: Optional[float] = None
    if cap is not None:
        f1 = (
            cap.linear_head_ceiling
            if cap.linear_head_ceiling is not None
            else cap.separability_score
        )
    return ProbeResult(
        recipe_id=recipe.id,
        in_scope_f1=f1,
        wall_time_s=time.perf_counter() - start,
        notes={"source": "frozen_capability"},
    )


class ProbeRunner:
    """Runs a probe function over candidate recipes."""

    def __init__(self, probe_fn: Optional[ProbeFn] = None):
        self.probe_fn = probe_fn or frozen_capability_probe

    def run(self, recipe: Recipe, ctx: ProbeContext) -> ProbeResult:
        return self.probe_fn(recipe, ctx)

    def run_many(self, recipes: List[Recipe], ctx: ProbeContext) -> List[ProbeResult]:
        return [self.run(r, ctx) for r in recipes]
