"""compile_plan: profile -> constraints -> (probes) -> ExecutionPlan.

Three routing modes (from ``TaskSpec.routing_mode``):

- ``legacy`` / ``compile_only``: deterministic *parity* path. Reproduces the
  existing resolver's decision exactly (via ``legacy_adapter``), then wraps it in
  an ExecutionPlan. Used to validate the new pipeline against the old one.
- ``full``: empirical path. Filters recipes by constraints, probes the top-K, and
  selects with the PlanScorer. May diverge from legacy -- that's the point.

The compiler is additive: nothing calls it until Phase 6 wires the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .constraints import ConstraintEngine
from .dataset_profile import extract_dataset_profile
from .execution_plan import ExecutionPlan, hash_profile
from .legacy_adapter import resolve_method_family
from .plan_scorer import PlanScorer
from .probe_runner import ProbeContext, ProbeRunner
from .recipe import Recipe
from .registry import RecipeRegistry
from .task_spec import (
    OOD_POLICY_DETECTOR,
    ROUTING_MODE_COMPILE_ONLY,
    ROUTING_MODE_FULL,
    ROUTING_MODE_LEGACY,
    TaskSpec,
)

log = logging.getLogger(__name__)


def _build_plan(
    recipe: Recipe,
    model_config: Any,
    profile_hash: str,
    compiled_at: Optional[str],
    notes: dict,
    probe_scores: Optional[dict] = None,
    selection_margin: float = 0.0,
) -> ExecutionPlan:
    components = {"trainer": recipe.trainer, "method_family": recipe.method_family}
    if recipe.ood_scorer_default:
        components["ood_scorer"] = recipe.ood_scorer_default
    for key, value in recipe.components.items():
        components.setdefault(key, value)
    return ExecutionPlan(
        recipe_id=recipe.id,
        model_id=getattr(model_config, "encoder", None),
        components=components,
        probe_scores=probe_scores or {},
        dataset_profile_hash=profile_hash,
        selection_margin=selection_margin,
        compiled_at=compiled_at,
        notes=notes,
    )


def compile_plan(
    dataset: Any,
    task_spec: Optional[TaskSpec] = None,
    *,
    registry: Optional[RecipeRegistry] = None,
    model_config: Any = None,
    min_class_size: Optional[int] = None,
    has_anc_label: Optional[bool] = None,
    embedder: Any = None,
    probe_runner: Optional[ProbeRunner] = None,
    plan_scorer: Optional[PlanScorer] = None,
    compiled_at: Optional[str] = None,
) -> ExecutionPlan:
    """Compile a routing decision into an :class:`ExecutionPlan`.

    Args:
        dataset: labeled data (``datasets.Dataset`` or ``pandas.DataFrame``).
        task_spec: routing intent; ``routing_mode`` selects the path.
        model_config: encoder config (defaults to ``task_spec.model``).
        min_class_size / has_anc_label: override the values derived from the
            DatasetProfile (used by the pipeline to pass post-resampling counts).
        embedder: inject an Embedder for the probe stage (``full`` mode).
        probe_runner / plan_scorer: inject for testing or custom strategies.
    """
    task_spec = task_spec or TaskSpec()
    registry = registry or RecipeRegistry.load()
    model_config = model_config if model_config is not None else task_spec.model

    profile = extract_dataset_profile(dataset, task_spec)
    if min_class_size is None:
        min_class_size = profile.min_class_size
    if has_anc_label is None:
        has_anc_label = profile.has_anc_label
    profile_hash = hash_profile(profile.to_dict())

    mode = task_spec.routing_mode

    # ---- Parity path (legacy / compile_only) -------------------------------
    if mode in (ROUTING_MODE_LEGACY, ROUTING_MODE_COMPILE_ONLY):
        ood_enabled = task_spec.ood_policy == OOD_POLICY_DETECTOR
        family = resolve_method_family(min_class_size, has_anc_label)
        recipe = registry.find(method_family=family, ood=ood_enabled)
        return _build_plan(
            recipe,
            model_config,
            profile_hash,
            compiled_at,
            notes={"routing_mode": mode},
        )

    # ---- Empirical path (full) ---------------------------------------------
    if mode != ROUTING_MODE_FULL:
        raise ValueError(f"Unknown routing_mode '{mode}'.")

    candidates = [
        sr.recipe
        for sr in ConstraintEngine(registry).filter(
            task_spec, min_class_size=min_class_size, has_anc_label=has_anc_label
        )
    ]
    if not candidates:
        raise ValueError("No recipes satisfy the constraints for this task.")
    candidates = candidates[: task_spec.budget.max_candidates]

    capability = None
    run_probes = not task_spec.budget.skip_probes or embedder is not None
    if run_probes and (embedder is not None or getattr(model_config, "encoder", None)):
        from .capability_profile import extract_capability_profile

        capability = extract_capability_profile(
            dataset, model_config, embedder=embedder
        )

    ctx = ProbeContext(
        dataset=dataset,
        profile=profile,
        capability=capability,
        model_config=model_config,
    )
    runner = probe_runner or ProbeRunner()
    results = runner.run_many(candidates, ctx)

    scorer = plan_scorer or PlanScorer()
    best, margin = scorer.select(
        list(zip(candidates, results)), task_spec.objective, profile
    )
    return _build_plan(
        best.recipe,
        model_config,
        profile_hash,
        compiled_at,
        notes={"routing_mode": ROUTING_MODE_FULL, "n_candidates": len(candidates)},
        probe_scores={res.recipe_id: res.to_dict() for res in results},
        selection_margin=margin,
    )
