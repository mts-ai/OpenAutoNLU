"""Compiler integration (Phase 5): parity + empirical selection."""

import numpy as np
import pandas as pd
import pytest

from open_autonlu.methods.data_types import OodMethod
from open_autonlu.routing import (
    ModelConfig,
    PlanScorer,
    ProbeResult,
    ProbeRunner,
    RecipeRegistry,
    TaskSpec,
    compile_plan,
)
from open_autonlu.routing import legacy_adapter as la
from open_autonlu.routing.task_spec import (
    OOD_POLICY_DETECTOR,
    OOD_POLICY_NONE,
    ROUTING_MODE_COMPILE_ONLY,
    ROUTING_MODE_FULL,
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


# --------------------------------------------------------------------------- #
# compile_only parity                                                         #
# --------------------------------------------------------------------------- #

PARITY_CASES = [
    # (n_per_class, anc, ood_policy, ood_method_for_legacy)
    (5, True, OOD_POLICY_NONE, OodMethod.NONE),
    (5, False, OOD_POLICY_NONE, OodMethod.NONE),  # anc fallback -> setfit
    (30, False, OOD_POLICY_NONE, OodMethod.NONE),
    (120, False, OOD_POLICY_NONE, OodMethod.NONE),
    (30, False, OOD_POLICY_DETECTOR, OodMethod.AUTO),
    (120, False, OOD_POLICY_DETECTOR, OodMethod.MARGINAL_MAHALANOBIS_OOD),
    (5, True, OOD_POLICY_DETECTOR, OodMethod.AUTO),
]


@pytest.mark.parametrize("n,anc,policy,ood_method", PARITY_CASES)
def test_compile_only_matches_legacy_trainer(registry, n, anc, policy, ood_method):
    df = _df(n, anc=anc)
    spec = TaskSpec(routing_mode=ROUTING_MODE_COMPILE_ONLY, ood_policy=policy)
    plan = compile_plan(df, spec, registry=registry)

    expected = la.trainer_class_name(
        registry, min_class_size=n, ood_method=ood_method, has_anc_label=anc
    )
    assert plan.components["trainer"] == expected
    assert plan.notes["routing_mode"] == ROUTING_MODE_COMPILE_ONLY
    assert plan.dataset_profile_hash


def test_compile_only_plan_loads_real_class(registry):
    plan = compile_plan(
        _df(120), TaskSpec(routing_mode=ROUTING_MODE_COMPILE_ONLY), registry=registry
    )
    cls, data = la.to_method_and_data(plan, training_data="DATA")
    assert cls.__name__ == "Finetuner"
    assert data == "DATA"


# --------------------------------------------------------------------------- #
# full mode (empirical)                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n,anc,expected_recipe",
    [
        (120, False, "finetuner"),   # ample -> finetune regime
        (30, False, "setfit"),       # moderate -> setfit regime
        (5, True, "anc_setfit"),     # scarce + anchors -> ancsetfit regime
    ],
)
def test_full_mode_picks_regime_recipe(registry, n, anc, expected_recipe):
    df = _df(n, anc=anc)
    spec = TaskSpec(routing_mode=ROUTING_MODE_FULL, ood_policy=OOD_POLICY_NONE)
    plan = compile_plan(df, spec, registry=registry, embedder=SeparableStub())
    assert plan.recipe_id == expected_recipe
    assert plan.notes["routing_mode"] == ROUTING_MODE_FULL
    assert plan.probe_scores  # populated
    assert plan.selection_margin >= 0.0


def test_full_mode_detector_policy_selects_ood_recipe(registry):
    spec = TaskSpec(routing_mode=ROUTING_MODE_FULL, ood_policy=OOD_POLICY_DETECTOR)
    plan = compile_plan(_df(120), spec, registry=registry, embedder=SeparableStub())
    assert plan.recipe_id == "finetuner_ood"
    assert plan.components.get("ood_scorer") == "mahalanobis"


def test_full_mode_with_injected_probe_runner(registry):
    """No embedder; deterministic probe -> scorer selects by regime fit."""
    fixed = ProbeRunner(probe_fn=lambda r, ctx: ProbeResult(r.id, in_scope_f1=0.9))
    spec = TaskSpec(routing_mode=ROUTING_MODE_FULL, ood_policy=OOD_POLICY_NONE)
    plan = compile_plan(_df(120), spec, registry=registry, probe_runner=fixed)
    assert plan.recipe_id == "finetuner"


def test_model_id_recorded(registry):
    spec = TaskSpec(routing_mode=ROUTING_MODE_COMPILE_ONLY)
    spec.model = ModelConfig(encoder="bert-base-uncased")
    plan = compile_plan(_df(120), spec, registry=registry, model_config=spec.model)
    assert plan.model_id == "bert-base-uncased"


# --------------------------------------------------------------------------- #
# scorer unit                                                                  #
# --------------------------------------------------------------------------- #


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
    assert margin == pytest.approx(0.4, abs=1e-6)
