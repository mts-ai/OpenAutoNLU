"""Serializable output of the routing compiler.

The router does not return a class name; it returns a versioned, serializable
``ExecutionPlan`` that is persisted alongside the trained model. On model swap
or drift, re-running probes produces a new plan; a plan diff shows what changed
(e.g. "OOD scorer flipped MSP -> energy after encoder change").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

SCHEMA_VERSION = 1


@dataclass
class ExecutionPlan:
    """A compiled routing decision.

    Attributes:
        recipe_id: Selected recipe id.
        model_id: Encoder the plan was compiled for (user's base model).
        components: Resolved component slots (trainer, ood_scorer, ...).
        probe_scores: Empirical probe metrics, if probes were run.
        dataset_profile_hash: Hash of the dataset profile used for selection.
        selection_margin: Score gap to the runner-up recipe (0.0 if not probed).
        hpo_scope: Optional hyperparameter search ranges.
        compiled_at: ISO timestamp; injected by the caller (env has no clock).
        schema_version: Plan schema version for forward compatibility.
        notes: Free-form provenance (e.g. "routing_mode=compile_only").
    """

    recipe_id: str
    model_id: Optional[str] = None
    components: Dict[str, Any] = field(default_factory=dict)
    probe_scores: Dict[str, Any] = field(default_factory=dict)
    dataset_profile_hash: Optional[str] = None
    selection_margin: float = 0.0
    hpo_scope: Dict[str, Any] = field(default_factory=dict)
    compiled_at: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExecutionPlan":
        return cls.from_dict(json.loads(Path(path).read_text()))


def hash_profile(profile: Dict[str, Any]) -> str:
    """Stable hash of a (JSON-serializable) profile dict."""
    blob = json.dumps(profile, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
