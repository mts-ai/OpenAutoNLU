"""Bridge between the new routing layer and the existing ``Method`` classes.

This adapter guarantees *parity*: given the same inputs, it resolves the exact
same ``Method`` class that ``TextClassificationTrainingPipeline._synergize_data_
and_method`` would resolve today. It reuses the existing ``resolve_method`` and
the ``OOD_METHOD_MAP`` / ``NO_OOD_METHOD_MAP`` constants so behavior cannot drift.

It deliberately covers the *method-class resolution* path (the part recipes
replace). Data upsample/downsample stays in the pipeline for now and is wired in
a later phase.
"""

from __future__ import annotations

import logging
from typing import Type

from ..constants import ANCSETFIT_NAME, SETFIT_NAME
from ..method_resolver import resolve_method
from ..methods.data_types import OodMethod
from .recipe import Recipe
from .registry import RecipeRegistry

log = logging.getLogger(__name__)

# Legacy semantics: OOD is "enabled" for every method except NONE and LOGIT.
# (LOGIT adds an out-of-scope class but uses the *no-OOD* trainer class.)
_OOD_DISABLED = (OodMethod.NONE, OodMethod.LOGIT)


def is_ood_enabled(ood_method: OodMethod) -> bool:
    """Mirror of the legacy ``ood_method in (NONE, LOGIT)`` branch."""
    return ood_method not in _OOD_DISABLED


def resolve_method_family(min_class_size: int, has_anc_label: bool) -> str:
    """Reproduce the legacy family selection.

    Uses the existing ``resolve_method`` for the size-based decision, then
    applies the AncSetFit->SetFit fallback when no ``anc_label`` column exists
    (matching ``_synergize_data_and_method``).
    """
    family = resolve_method(min_class_size)
    if family == ANCSETFIT_NAME and not has_anc_label:
        log.debug("No anc_label; falling back ancsetfit -> setfit (legacy parity).")
        family = SETFIT_NAME
    return family


def resolve_recipe(
    registry: RecipeRegistry,
    min_class_size: int,
    ood_method: OodMethod = OodMethod.AUTO,
    has_anc_label: bool = False,
) -> Recipe:
    """Resolve the recipe a legacy run would have used for these inputs."""
    family = resolve_method_family(min_class_size, has_anc_label)
    return registry.find(method_family=family, ood=is_ood_enabled(ood_method))


def trainer_class_name(
    registry: RecipeRegistry,
    min_class_size: int,
    ood_method: OodMethod = OodMethod.AUTO,
    has_anc_label: bool = False,
) -> str:
    """Return the ``Method`` class *name* (no heavy import)."""
    return resolve_recipe(registry, min_class_size, ood_method, has_anc_label).trainer


def load_method_class(trainer_name: str) -> Type:
    """Resolve a trainer class name to the actual class (heavy import, lazy)."""
    from ..auto_classes.abstract_pipeline import get_method_from_string

    return get_method_from_string(trainer_name)


def to_method_and_data(plan, training_data):
    """Map an ``ExecutionPlan`` to ``(Method class, training data)``.

    ``plan.components["trainer"]`` must hold the trainer class name. Data is
    passed through unchanged in Phase 0.
    """
    trainer_name = plan.components.get("trainer")
    if not trainer_name:
        raise ValueError("ExecutionPlan.components is missing 'trainer'.")
    return load_method_class(trainer_name), training_data
