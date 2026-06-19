"""Map compiled ExecutionPlans to Method classes."""

from __future__ import annotations

from typing import Type


def load_method_class(trainer_name: str) -> Type:
    """Resolve a trainer class name to the actual class (lazy import)."""
    from ..auto_classes.abstract_pipeline import get_method_from_string

    return get_method_from_string(trainer_name)


def to_method_and_data(plan, training_data):
    """Map an ``ExecutionPlan`` to ``(Method class, training data)``."""
    trainer_name = plan.components.get("trainer")
    if not trainer_name:
        raise ValueError("ExecutionPlan.components is missing 'trainer'.")
    return load_method_class(trainer_name), training_data
