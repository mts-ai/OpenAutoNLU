"""Constraint engine: hard guardrails + soft preference scoring.

This is the "guardrails, not intelligence" layer. It filters the recipe set
*before* any expensive probe runs, using TaskSpec hard constraints and cheap
profile signals. Phase 0/3 implements the hard filters and a light soft score;
the empirical decision happens later in the probe runner / plan scorer.

User overrides act as hard constraints here: an explicit ``recipe_id`` pins the
candidate set to a single recipe (still validated against constraints, with a
warning rather than a hard failure when violated -- "explicit beats implicit").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .recipe import Recipe
from .registry import RecipeRegistry
from .task_spec import (
    OOD_POLICY_DETECTOR,
    OOD_POLICY_LOGIT_CLASS,
    OOD_POLICY_NONE,
    TaskSpec,
)

log = logging.getLogger(__name__)


@dataclass
class ScoredRecipe:
    recipe: Recipe
    score: float


class ConstraintEngine:
    """Filters and softly ranks recipes for a given context."""

    def __init__(self, registry: RecipeRegistry):
        self.registry = registry

    def filter(
        self,
        task_spec: TaskSpec,
        min_class_size: Optional[int] = None,
        has_anc_label: bool = False,
        user_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredRecipe]:
        """Return constraint-satisfying recipes, ranked by soft score (desc).

        Raises:
            ValueError: if ``min_class_size`` is below the hard minimum (2),
                or if a pinned ``recipe_id`` is unknown.
        """
        user_overrides = user_overrides or {}

        # Hard: explicit user pin wins.
        pinned = user_overrides.get("recipe_id")
        if pinned is not None:
            recipe = self.registry.get(pinned)  # raises if unknown
            violations = self._violations(
                recipe, task_spec, min_class_size, has_anc_label
            )
            if violations:
                log.warning(
                    "Pinned recipe '%s' violates constraints %s; honoring user "
                    "override anyway (explicit beats implicit).",
                    pinned,
                    violations,
                )
            return [ScoredRecipe(recipe, score=1.0)]

        # Hard: absolute floor on data.
        if min_class_size is not None and min_class_size < 2:
            raise ValueError(
                f"At least 2 samples per label required; found {min_class_size}."
            )

        candidates: List[ScoredRecipe] = []
        for recipe in self.registry.all():
            if self._violations(recipe, task_spec, min_class_size, has_anc_label):
                continue
            candidates.append(
                ScoredRecipe(recipe, self._soft_score(recipe, min_class_size))
            )

        candidates.sort(key=lambda sr: sr.score, reverse=True)
        return candidates

    def _violations(
        self,
        recipe: Recipe,
        task_spec: TaskSpec,
        min_class_size: Optional[int],
        has_anc_label: bool,
    ) -> List[str]:
        """Return a list of hard-constraint violations (empty == eligible)."""
        v: List[str] = []

        # OOD policy gating (mirrors legacy: LOGIT/NONE -> no-OOD trainer).
        if task_spec.ood_policy == OOD_POLICY_NONE and recipe.ood:
            v.append("ood_policy=none excludes OOD recipes")
        if task_spec.ood_policy == OOD_POLICY_LOGIT_CLASS and recipe.ood:
            v.append("ood_policy=logit_class uses no-OOD trainer")
        if task_spec.ood_policy == OOD_POLICY_DETECTOR and not recipe.ood:
            v.append("ood_policy=detector requires an OOD recipe")

        # Anchor requirement.
        if recipe.requires_anc_label and not has_anc_label:
            v.append("recipe requires anc_label column")

        return v

    def _soft_score(self, recipe: Recipe, min_class_size: Optional[int]) -> float:
        """Neutral tie-breaker; empirical probes decide ranking."""
        del recipe, min_class_size
        return 0.0
