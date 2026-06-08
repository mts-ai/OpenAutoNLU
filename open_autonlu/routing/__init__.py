"""Routing layer: profile -> constraints -> probes -> recipe.

See ``arch_suggestion.md`` for the design. Phase 0 ships the declarative
scaffolding (TaskSpec, Recipe, RecipeRegistry, ConstraintEngine, ExecutionPlan)
plus a parity-preserving ``legacy_adapter``. Profiles, probes, the scorer and
the compiler land in later phases.

Note: the top-level package keeps imports light (no torch). ``legacy_adapter``
pulls the heavy ``methods`` stack and is imported explicitly when needed.
"""

from .constraints import ConstraintEngine, ScoredRecipe
from .execution_plan import ExecutionPlan, hash_profile
from .recipe import Recipe
from .registry import RecipeRegistry
from .task_spec import (
    BudgetPolicy,
    ModelConfig,
    ObjectiveWeights,
    TaskSpec,
)

__all__ = [
    "ConstraintEngine",
    "ScoredRecipe",
    "ExecutionPlan",
    "hash_profile",
    "Recipe",
    "RecipeRegistry",
    "TaskSpec",
    "ModelConfig",
    "BudgetPolicy",
    "ObjectiveWeights",
]
