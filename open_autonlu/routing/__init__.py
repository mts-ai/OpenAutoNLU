"""Routing layer: profile -> constraints -> probes -> recipe.

Declarative scaffolding (TaskSpec, Recipe, RecipeRegistry, ConstraintEngine,
ExecutionPlan) plus empirical selection via encoder probes and PlanScorer.

Note: the top-level package keeps imports light (no torch). Heavy trainer imports
are lazy inside ``plan_adapter``.
"""

from .capability_profile import (
    CapabilityProfile,
    Embedder,
    HFEmbedder,
    extract_capability_profile,
)
from .compiler import compile_plan
from .constraints import ConstraintEngine, ScoredRecipe
from .dataset_profile import DatasetProfile, extract_dataset_profile
from .execution_plan import ExecutionPlan, hash_profile
from .plan_scorer import PlanScorer, ScoredPlan
from .probe_runner import ProbeContext, ProbeResult, ProbeRunner, frozen_capability_probe, recipe_encoder_probe
from .recipe import ProbeConfig, Recipe
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
    "DatasetProfile",
    "extract_dataset_profile",
    "CapabilityProfile",
    "extract_capability_profile",
    "Embedder",
    "HFEmbedder",
    "compile_plan",
    "PlanScorer",
    "ScoredPlan",
    "ProbeRunner",
    "ProbeContext",
    "ProbeResult",
    "frozen_capability_probe",
    "recipe_encoder_probe",
    "ProbeConfig",
]
