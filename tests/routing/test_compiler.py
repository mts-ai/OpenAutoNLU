"""Compiler integration: empirical recipe selection."""

import numpy as np
import pandas as pd
import pytest

from open_autonlu.routing import (
    ModelConfig,
    PlanScorer,
    ProbeResult,
    ProbeRunner,
    RecipeRegistry,
    TaskSpec,
    compile_plan,
)
from open_autonlu.routing.plan_adapter import to_method_and_data
from open_autonlu.routing.task_spec import (
    OOD_POLICY_DETECTOR,
    OOD_POLICY_NONE,
)


@pytest.fixture(scope="module")
def registry():
    return RecipeRegistry.load()


def _df(n_per_class, classes=("transfer", "weather", "music"), anc=False):
    rows = []
    for c in classes:
        for i in range(n_per_class):
            row = {"text": f"{c} request number {i}", "label": c}
            if anc:
                row["anc_label"] = f"the {c} skill"
            rows.append(row)
    return pd.DataFrame(rows)


class SeparableStub:
    """One-hot-ish embedding keyed on the first token (== label)."""

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def encode(self, texts):
        labels = [t.split()[0] for t in texts]
        uniq = sorted(set(labels))
        idx = {c: i for i, c in enumerate(uniq)}
        X = np.zeros((len(texts), len(uniq)), dtype=np.float32)
        for r, lab in enumerate(labels):
            X[r, idx[lab]] = 1.0
        return X + 0.01 * self.rng.randn(*X.shape).astype(np.float32)


@pytest.mark.parametrize(
    "n,anc,expected_recipe",
    [
        (120, False, "setfit"),
        (30, False, "setfit"),
        (5, True, "anc_setfit"),
    ],
)
def test_compile_selects_by_probe_and_cost(registry, n, anc, expected_recipe):
    df = _df(n, anc=anc)
    spec = TaskSpec(ood_policy=OOD_POLICY_NONE)
    plan = compile_plan(df, spec, registry=registry, embedder=SeparableStub())
    assert plan.recipe_id == expected_recipe
    assert plan.probe_scores
    assert plan.selection_margin >= 0.0
    assert plan.dataset_profile_hash


def test_compile_plan_loads_real_class(registry):
    plan = compile_plan(_df(120), TaskSpec(), registry=registry, embedder=SeparableStub())
    cls, data = to_method_and_data(plan, training_data="DATA")
    assert cls.__name__ == "SetFitMethod"
    assert data == "DATA"


def test_full_mode_detector_policy_selects_ood_recipe(registry):
    spec = TaskSpec(ood_policy=OOD_POLICY_DETECTOR)
    plan = compile_plan(_df(120), spec, registry=registry, embedder=SeparableStub())
    assert plan.recipe_id == "setfit_ood"
    assert plan.components.get("ood_scorer") == "msp"


def test_equal_probe_scores_prefer_cheaper_recipe(registry):
    fixed = ProbeRunner(probe_fn=lambda r, ctx: ProbeResult(r.id, in_scope_f1=0.9))
    spec = TaskSpec(ood_policy=OOD_POLICY_NONE)
    plan = compile_plan(_df(120), spec, registry=registry, probe_runner=fixed)
    assert plan.recipe_id == "setfit"


def test_pinned_recipe_id(registry):
    spec = TaskSpec(ood_policy=OOD_POLICY_NONE)
    plan = compile_plan(
        _df(120),
        spec,
        registry=registry,
        user_overrides={"recipe_id": "finetuner"},
    )
    assert plan.recipe_id == "finetuner"
    assert plan.notes.get("selection") == "pinned"


def test_model_id_recorded(registry):
    spec = TaskSpec()
    spec.model = ModelConfig(encoder="bert-base-uncased")
    plan = compile_plan(
        _df(120), spec, registry=registry, model_config=spec.model, embedder=SeparableStub()
    )
    assert plan.model_id == "bert-base-uncased"


def test_plan_scorer_select_empty_raises():
    with pytest.raises(ValueError):
        PlanScorer().select([], TaskSpec().objective)


def test_plan_scorer_margin(registry):
    setfit = registry.get("setfit")
    finetuner = registry.get("finetuner")
    pairs = [
        (setfit, ProbeResult("setfit", in_scope_f1=0.5)),
        (finetuner, ProbeResult("finetuner", in_scope_f1=0.9)),
    ]
    best, margin = PlanScorer().select(pairs, TaskSpec().objective)
    assert best.recipe.id == "finetuner"
    assert margin == pytest.approx(0.35, abs=1e-6)
