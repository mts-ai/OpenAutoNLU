"""PlanScorer: multi-objective selection over probed candidates.

Combines empirical probe signals (in-scope F1, OOD score) with the user
objective (cost penalty). Selection is probe-driven only -- no sample-count
regime bonuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .probe_runner import ProbeResult
from .recipe import Recipe

# Normalized cost penalty per recipe tier.
DEFAULT_COST_MAP: Dict[str, float] = {"low": 0.0, "medium": 0.5, "high": 1.0}


@dataclass
class ScoredPlan:
    recipe: Recipe
    result: ProbeResult
    score: float


class PlanScorer:
    """Scores and ranks candidate plans for a given objective."""

    def __init__(self, cost_map: Optional[Dict[str, float]] = None):
        self.cost_map = cost_map or DEFAULT_COST_MAP

    def score_one(
        self, recipe: Recipe, result: ProbeResult, objective, profile=None
    ) -> float:
        del profile  # kept for API compatibility; not used for scoring
        f1 = result.in_scope_f1 or 0.0
        score = objective.in_scope_f1 * f1

        if result.ood_score is not None:
            ood_weight = objective.ood_recall_close + objective.ood_recall_far
            score += ood_weight * result.ood_score

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
