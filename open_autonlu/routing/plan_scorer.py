"""PlanScorer: multi-objective selection over probed candidates.

Combines the empirical probe signal (in-scope F1, OOD score) with structural
priors (data-regime fit, cost tier). The objective weights come from
``TaskSpec.objective``. Returns the winning plan plus the score margin to the
runner-up (used as ``ExecutionPlan.selection_margin``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .probe_runner import ProbeResult
from .recipe import Recipe

# Normalized cost penalty per recipe tier.
DEFAULT_COST_MAP: Dict[str, float] = {"low": 0.0, "medium": 0.5, "high": 1.0}

# Small structural bonus for a recipe whose declared regime fits the data.
REGIME_FIT_BONUS = 0.1


@dataclass
class ScoredPlan:
    recipe: Recipe
    result: ProbeResult
    score: float


class PlanScorer:
    """Scores and ranks candidate plans for a given objective."""

    def __init__(
        self,
        cost_map: Optional[Dict[str, float]] = None,
        regime_fit_bonus: float = REGIME_FIT_BONUS,
    ):
        self.cost_map = cost_map or DEFAULT_COST_MAP
        self.regime_fit_bonus = regime_fit_bonus

    def score_one(
        self, recipe: Recipe, result: ProbeResult, objective, profile=None
    ) -> float:
        f1 = result.in_scope_f1 or 0.0
        score = objective.in_scope_f1 * f1

        if result.ood_score is not None:
            ood_weight = objective.ood_recall_close + objective.ood_recall_far
            score += ood_weight * result.ood_score

        # Structural prior: prefer a recipe whose regime matches the data.
        if profile is not None and recipe.matches_class_size(profile.min_class_size):
            score += self.regime_fit_bonus

        # Cost penalty.
        score -= objective.train_cost * self.cost_map.get(recipe.cost_tier, 0.5)
        return score

    def rank(
        self,
        pairs: List[Tuple[Recipe, ProbeResult]],
        objective,
        profile=None,
    ) -> List[ScoredPlan]:
        scored = [
            ScoredPlan(recipe, result, self.score_one(recipe, result, objective, profile))
            for recipe, result in pairs
        ]
        scored.sort(key=lambda sp: sp.score, reverse=True)
        return scored

    def select(
        self,
        pairs: List[Tuple[Recipe, ProbeResult]],
        objective,
        profile=None,
    ) -> Tuple[ScoredPlan, float]:
        """Return the best plan and its margin to the runner-up.

        Raises:
            ValueError: if ``pairs`` is empty.
        """
        if not pairs:
            raise ValueError("No candidate plans to score.")
        ranked = self.rank(pairs, objective, profile)
        margin = ranked[0].score - ranked[1].score if len(ranked) > 1 else ranked[0].score
        return ranked[0], float(margin)
