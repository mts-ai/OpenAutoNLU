"""ConstraintEngine hard-filtering + soft ranking (Phase 0)."""

import pytest

from open_autonlu.routing import ConstraintEngine, RecipeRegistry, TaskSpec
from open_autonlu.routing.task_spec import (
    OOD_POLICY_DETECTOR,
    OOD_POLICY_LOGIT_CLASS,
    OOD_POLICY_NONE,
)


@pytest.fixture(scope="module")
def engine():
    return ConstraintEngine(RecipeRegistry.load())


def _ids(scored):
    return {sr.recipe.id for sr in scored}


def test_ood_policy_none_excludes_ood_recipes(engine):
    spec = TaskSpec(ood_policy=OOD_POLICY_NONE)
    out = engine.filter(spec, min_class_size=100, has_anc_label=True)
    assert all(not sr.recipe.ood for sr in out)


def test_ood_policy_detector_keeps_only_ood(engine):
    spec = TaskSpec(ood_policy=OOD_POLICY_DETECTOR)
    out = engine.filter(spec, min_class_size=100, has_anc_label=True)
    assert out and all(sr.recipe.ood for sr in out)


def test_logit_class_uses_no_ood_recipes(engine):
    spec = TaskSpec(ood_policy=OOD_POLICY_LOGIT_CLASS)
    out = engine.filter(spec, min_class_size=100, has_anc_label=True)
    assert all(not sr.recipe.ood for sr in out)


def test_anc_recipes_excluded_without_anc_label(engine):
    spec = TaskSpec(ood_policy=OOD_POLICY_NONE)
    out = engine.filter(spec, min_class_size=3, has_anc_label=False)
    assert "anc_setfit" not in _ids(out)


def test_soft_score_prefers_matching_regime(engine):
    """For a full-data size, the finetuner recipe should rank top."""
    spec = TaskSpec(ood_policy=OOD_POLICY_NONE)
    out = engine.filter(spec, min_class_size=500, has_anc_label=True)
    assert out[0].recipe.id == "finetuner"


def test_pinned_recipe_bypasses_ranking(engine):
    spec = TaskSpec(ood_policy=OOD_POLICY_NONE)
    out = engine.filter(
        spec, min_class_size=500, has_anc_label=True,
        user_overrides={"recipe_id": "setfit_ood"},
    )
    assert _ids(out) == {"setfit_ood"}


def test_pinned_unknown_recipe_raises(engine):
    with pytest.raises(KeyError):
        engine.filter(TaskSpec(), user_overrides={"recipe_id": "nope"})


def test_min_size_below_floor_raises(engine):
    with pytest.raises(ValueError):
        engine.filter(TaskSpec(), min_class_size=1)
