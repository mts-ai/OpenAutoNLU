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

from .data_prep import DataPrepConfig, data_prep_for_trainer


@dataclass
class ProbeConfig:
    """Encoder probe settings for empirical recipe selection."""

    probe_type: str = "knn"  # few_shot_knn | knn | linear
    max_samples: int = 500
    n_shot: int = 5
    n_neighbors: int = 5
    max_folds: int = 3

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ProbeConfig":
        if not data:
            return cls()
        return cls(
            probe_type=str(data.get("probe_type", "knn")),
            max_samples=int(data.get("max_samples", 500)),
            n_shot=int(data.get("n_shot", 5)),
            n_neighbors=int(data.get("n_neighbors", 5)),
            max_folds=int(data.get("max_folds", 3)),
        )


_DEFAULT_PROBE_BY_TRAINER_PREFIX = (
    ("AncSetFit", "few_shot_knn"),
    ("SetFit", "knn"),
    ("Finetuner", "linear"),
    ("TokenClassification", "linear"),
)


def default_probe_type_for_trainer(trainer: str) -> str:
    for prefix, probe_type in _DEFAULT_PROBE_BY_TRAINER_PREFIX:
        if trainer.startswith(prefix):
            return probe_type
    return "knn"


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
        requires_anc_label: If True, recipe needs an ``anc_label`` column.
        requires: Raw "requires" block (hard/soft constraints) for the engine.
        components: Extra component slots (augmenter, ood_sampler, ...).
        probe: Encoder probe config for empirical selection.
    """

    id: str
    method_family: str
    trainer: str
    ood: bool = False
    ood_scorer_default: Optional[str] = None
    cost_tier: str = "medium"
    requires_anc_label: bool = False
    requires: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    data_prep: DataPrepConfig = field(default_factory=DataPrepConfig)

    def effective_probe_type(self) -> str:
        return self.probe.probe_type

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recipe":
        known = {
            "id",
            "method_family",
            "trainer",
            "ood",
            "ood_scorer_default",
            "cost_tier",
            "requires_anc_label",
            "requires",
            "components",
            "probe",
            "data_prep",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Recipe '{data.get('id', '<unknown>')}' has unknown keys: {sorted(unknown)}"
            )
        trainer = data["trainer"]
        if data.get("probe") is None:
            probe = ProbeConfig(probe_type=default_probe_type_for_trainer(trainer))
        else:
            probe = ProbeConfig.from_dict(data["probe"])
        if data.get("data_prep") is None:
            data_prep = data_prep_for_trainer(trainer)
        else:
            data_prep = DataPrepConfig.from_dict(data["data_prep"])
        return cls(
            id=data["id"],
            method_family=data["method_family"],
            trainer=trainer,
            ood=bool(data.get("ood", False)),
            ood_scorer_default=data.get("ood_scorer_default"),
            cost_tier=data.get("cost_tier", "medium"),
            requires_anc_label=bool(data.get("requires_anc_label", False)),
            requires=data.get("requires") or {},
            components=data.get("components") or {},
            probe=probe,
            data_prep=data_prep,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "Recipe":
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Recipe file {path} must contain a mapping.")
        data.setdefault("id", path.stem)
        return cls.from_dict(data)
