"""User-facing routing intent declarations.

These dataclasses describe *what the user wants* (task, objective, budget,
base model). They contain requirements and knobs only -- no routing decision
is made here. The compiler consumes a ``TaskSpec`` together with the dataset
and produces an ``ExecutionPlan``.

Design principle: this module is model- and language-neutral. Locale/domain
specifics live in plugins selected via ``TaskSpec.language``, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# OOD policy literals (kept as plain strings to avoid a hard enum dependency
# in user configs / YAML).
OOD_POLICY_NONE = "none"
OOD_POLICY_LOGIT_CLASS = "logit_class"
OOD_POLICY_DETECTOR = "detector"
VALID_OOD_POLICIES = frozenset(
    {OOD_POLICY_NONE, OOD_POLICY_LOGIT_CLASS, OOD_POLICY_DETECTOR}
)

# Routing modes (mirrors arch_suggestion.md migration story).
ROUTING_MODE_LEGACY = "legacy"
ROUTING_MODE_COMPILE_ONLY = "compile_only"
ROUTING_MODE_FULL = "full"
VALID_ROUTING_MODES = frozenset(
    {ROUTING_MODE_LEGACY, ROUTING_MODE_COMPILE_ONLY, ROUTING_MODE_FULL}
)


@dataclass
class ObjectiveWeights:
    """Multi-objective weights used by the plan scorer.

    Higher is better for the positive terms; cost/variance are penalties.
    Defaults reproduce a "maximise in-scope F1" objective.
    """

    in_scope_f1: float = 1.0
    ood_recall_close: float = 0.0
    ood_recall_far: float = 0.0
    train_cost: float = 0.0
    variance: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "in_scope_f1": self.in_scope_f1,
            "ood_recall_close": self.ood_recall_close,
            "ood_recall_far": self.ood_recall_far,
            "train_cost": self.train_cost,
            "variance": self.variance,
        }


@dataclass
class BudgetPolicy:
    """Resource guardrails for probing/compilation.

    Attributes:
        skip_probes: If True, do not run empirical probes (rank-only routing).
        max_probe_minutes: Soft wall-clock cap for the whole probe stage.
        max_candidates: Max number of recipes that may enter the probe pool.
        max_train_minutes: Optional hard cap surfaced to the constraint engine.
    """

    skip_probes: bool = True
    max_probe_minutes: float = 10.0
    max_candidates: int = 3
    max_train_minutes: Optional[float] = None


@dataclass
class ModelConfig:
    """The user-chosen base encoder. Always the source of truth for probes."""

    encoder: Optional[str] = None
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_seq_length: int = 512


@dataclass
class TaskSpec:
    """Stable, explicit declaration of routing intent.

    Overrides everything else when set. ``ood_policy`` and ``anchors`` gate
    which recipe families are eligible; ``language`` selects plugins.
    """

    task_type: str = "multiclass_classification"
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    ood_policy: str = OOD_POLICY_NONE
    label_schema: str = "flat"  # "flat" | "hierarchical"
    anchors: bool = False  # enables anchor-based recipes
    language: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    routing_mode: str = ROUTING_MODE_LEGACY
    constraints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ood_policy not in VALID_OOD_POLICIES:
            raise ValueError(
                f"Unknown ood_policy '{self.ood_policy}'. "
                f"Valid: {sorted(VALID_OOD_POLICIES)}"
            )
        if self.routing_mode not in VALID_ROUTING_MODES:
            raise ValueError(
                f"Unknown routing_mode '{self.routing_mode}'. "
                f"Valid: {sorted(VALID_ROUTING_MODES)}"
            )
