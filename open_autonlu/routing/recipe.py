"""Declarative training recipes.

A ``Recipe`` is a composable plan fragment: a method family, its trainer
class name, OOD wiring, soft data-regime preferences, and cost tier. Recipes
replace the ``if/else`` logic in ``resolve_method`` + ``OOD_METHOD_MAP``.

Adding a new method == adding a YAML file here. The central router is not
touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Recipe:
    """A single declarative recipe.

    Attributes:
        id: Unique recipe identifier (also the YAML file stem).
        method_family: Legacy method-name key ("setfit"/"ancsetfit"/"finetuning").
            Used for parity mapping with the existing resolver.
        trainer: ``Method`` subclass name resolved via ``get_method_from_string``.
        ood: Whether this recipe wires an OOD detector into training.
        ood_scorer_default: Default OOD score ("msp"/"mahalanobis"/...), or None.
        cost_tier: "low" | "medium" | "high" -- consumed by BudgetPolicy/scorer.
        min_class_size: Soft lower bound on per-class samples (inclusive), or None.
        max_class_size: Soft upper bound on per-class samples (inclusive), or None.
        requires_anc_label: If True, recipe needs an ``anc_label`` column.
        requires: Raw "requires" block (hard/soft constraints) for the engine.
        components: Extra component slots (augmenter, ood_sampler, ...).
    """

    id: str
    method_family: str
    trainer: str
    ood: bool = False
    ood_scorer_default: Optional[str] = None
    cost_tier: str = "medium"
    min_class_size: Optional[int] = None
    max_class_size: Optional[int] = None
    requires_anc_label: bool = False
    requires: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)

    def matches_class_size(self, min_class_size: int) -> bool:
        """Soft check: does ``min_class_size`` fall within this recipe's regime?"""
        if self.min_class_size is not None and min_class_size < self.min_class_size:
            return False
        if self.max_class_size is not None and min_class_size > self.max_class_size:
            return False
        return True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recipe":
        known = {
            "id",
            "method_family",
            "trainer",
            "ood",
            "ood_scorer_default",
            "cost_tier",
            "min_class_size",
            "max_class_size",
            "requires_anc_label",
            "requires",
            "components",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Recipe '{data.get('id', '<unknown>')}' has unknown keys: {sorted(unknown)}"
            )
        return cls(
            id=data["id"],
            method_family=data["method_family"],
            trainer=data["trainer"],
            ood=bool(data.get("ood", False)),
            ood_scorer_default=data.get("ood_scorer_default"),
            cost_tier=data.get("cost_tier", "medium"),
            min_class_size=data.get("min_class_size"),
            max_class_size=data.get("max_class_size"),
            requires_anc_label=bool(data.get("requires_anc_label", False)),
            requires=data.get("requires") or {},
            components=data.get("components") or {},
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "Recipe":
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Recipe file {path} must contain a mapping.")
        data.setdefault("id", path.stem)
        return cls.from_dict(data)
