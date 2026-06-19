"""ProbeRunner: cheap empirical validation of candidate recipes.

Each recipe is scored with a *recipe-specific* encoder probe (few-shot kNN,
full kNN, or linear head) on embeddings from the user's encoder. Probes share
one encoding pass per compile; scores differ by probe type, not sample-count
regimes.

Pass a custom ``probe_fn`` to :class:`ProbeRunner` for tests or heavier
micro-training strategies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from ._eval_utils import few_shot_knn_macro_f1, knn_macro_f1, stratified_cv_macro_f1
from .recipe import Recipe

log = logging.getLogger(__name__)


@dataclass
class ProbeContext:
    """Everything a probe needs about the current routing problem."""

    dataset: Any = None
    profile: Any = None
    capability: Any = None
    model_config: Any = None
    embedder: Any = None
    embeddings: Optional[np.ndarray] = None
    labels: Optional[List[str]] = None
    seed: int = 42


@dataclass
class ProbeResult:
    """Outcome of probing a single recipe."""

    recipe_id: str
    in_scope_f1: Optional[float]
    ood_score: Optional[float] = None
    wall_time_s: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "in_scope_f1": self.in_scope_f1,
            "ood_score": self.ood_score,
            "wall_time_s": round(self.wall_time_s, 4),
            "notes": self.notes,
        }


ProbeFn = Callable[[Recipe, ProbeContext], ProbeResult]


def _encode_probe_samples(ctx: ProbeContext, max_samples: int) -> None:
    """Populate ``ctx.embeddings`` / ``ctx.labels`` once per probe batch."""
    if ctx.embeddings is not None and ctx.labels is not None:
        return

    from .capability_profile import _subsample, _to_dataframe

    if ctx.dataset is None:
        return

    df = _to_dataframe(ctx.dataset, "text", "label")
    work = _subsample(df, max_per_class=50, max_total=max_samples, seed=ctx.seed)

    embedder = ctx.embedder
    if embedder is None:
        model_id = getattr(ctx.model_config, "encoder", None)
        if not model_id:
            return
        from .capability_profile import HFEmbedder

        embedder = HFEmbedder(
            model_id,
            max_seq_length=getattr(ctx.model_config, "max_seq_length", 256),
        )

    texts = work["text"].tolist()
    ctx.labels = work["label"].tolist()
    ctx.embeddings = embedder.encode(texts)


def _score_probe(recipe: Recipe, ctx: ProbeContext) -> Optional[float]:
    if ctx.embeddings is None or ctx.labels is None:
        return None

    X = ctx.embeddings
    y = ctx.labels
    cfg = recipe.probe
    probe_type = recipe.effective_probe_type()

    if probe_type == "few_shot_knn":
        return few_shot_knn_macro_f1(
            X,
            y,
            n_shot=cfg.n_shot,
            seed=ctx.seed,
            max_folds=cfg.max_folds,
            n_neighbors=cfg.n_neighbors,
        )
    if probe_type == "linear":
        return stratified_cv_macro_f1(
            X,
            y,
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            seed=ctx.seed,
            max_folds=cfg.max_folds,
        )
    return knn_macro_f1(
        X,
        y,
        seed=ctx.seed,
        max_folds=cfg.max_folds,
        n_neighbors=cfg.n_neighbors,
    )


def recipe_encoder_probe(recipe: Recipe, ctx: ProbeContext) -> ProbeResult:
    """Score a recipe with its declared encoder probe type."""
    start = time.perf_counter()
    f1 = _score_probe(recipe, ctx)
    return ProbeResult(
        recipe_id=recipe.id,
        in_scope_f1=f1,
        wall_time_s=time.perf_counter() - start,
        notes={
            "source": "recipe_encoder",
            "probe_type": recipe.effective_probe_type(),
        },
    )


def frozen_capability_probe(recipe: Recipe, ctx: ProbeContext) -> ProbeResult:
    """Fallback when embeddings are unavailable (no encoder / budget skip)."""
    start = time.perf_counter()
    cap = ctx.capability
    f1: Optional[float] = None
    if cap is not None:
        f1 = (
            cap.linear_head_ceiling
            if cap.linear_head_ceiling is not None
            else cap.separability_score
        )
    return ProbeResult(
        recipe_id=recipe.id,
        in_scope_f1=f1,
        wall_time_s=time.perf_counter() - start,
        notes={"source": "frozen_capability"},
    )


class ProbeRunner:
    """Runs a probe function over candidate recipes."""

    def __init__(self, probe_fn: Optional[ProbeFn] = None):
        self.probe_fn = probe_fn or recipe_encoder_probe

    def run(self, recipe: Recipe, ctx: ProbeContext) -> ProbeResult:
        return self.probe_fn(recipe, ctx)

    def run_many(self, recipes: List[Recipe], ctx: ProbeContext) -> List[ProbeResult]:
        if recipes and self.probe_fn is recipe_encoder_probe:
            max_samples = max(r.probe.max_samples for r in recipes)
            try:
                _encode_probe_samples(ctx, max_samples)
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("Encoder probe failed: %s", exc)
        if ctx.embeddings is None and self.probe_fn is recipe_encoder_probe:
            return [frozen_capability_probe(r, ctx) for r in recipes]
        return [self.run(r, ctx) for r in recipes]
