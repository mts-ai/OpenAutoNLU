"""Recipe loading + registry lookups (Phase 0)."""

import pytest

from open_autonlu.constants import (
    ANCSETFIT_NAME,
    FINETUNING_NAME,
    NO_OOD_METHOD_MAP,
    OOD_METHOD_MAP,
    SETFIT_NAME,
)
from open_autonlu.routing import Recipe, RecipeRegistry


@pytest.fixture(scope="module")
def registry():
    return RecipeRegistry.load()


def test_loads_all_six_recipes(registry):
    ids = {r.id for r in registry.all()}
    assert ids == {
        "finetuner",
        "finetuner_ood",
        "setfit",
        "setfit_ood",
        "anc_setfit",
        "anc_setfit_ood",
    }


def test_find_by_family_and_ood(registry):
    assert registry.find(FINETUNING_NAME, ood=False).id == "finetuner"
    assert registry.find(FINETUNING_NAME, ood=True).id == "finetuner_ood"
    assert registry.find(SETFIT_NAME, ood=False).id == "setfit"
    assert registry.find(SETFIT_NAME, ood=True).id == "setfit_ood"
    assert registry.find(ANCSETFIT_NAME, ood=False).id == "anc_setfit"
    assert registry.find(ANCSETFIT_NAME, ood=True).id == "anc_setfit_ood"


def test_get_unknown_raises(registry):
    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_recipes_mirror_legacy_method_maps(registry):
    """Each recipe's trainer must equal the legacy map entry it replaces."""
    for family in (SETFIT_NAME, ANCSETFIT_NAME, FINETUNING_NAME):
        assert registry.find(family, ood=False).trainer == NO_OOD_METHOD_MAP[family]
        assert registry.find(family, ood=True).trainer == OOD_METHOD_MAP[family]


def test_class_size_regime_bounds():
    r = Recipe(
        id="x", method_family="setfit", trainer="SetFitMethod",
        min_class_size=6, max_class_size=80,
    )
    assert not r.matches_class_size(5)
    assert r.matches_class_size(6)
    assert r.matches_class_size(80)
    assert not r.matches_class_size(81)


def test_unknown_recipe_key_rejected():
    with pytest.raises(ValueError):
        Recipe.from_dict(
            {"id": "x", "method_family": "setfit", "trainer": "SetFitMethod",
             "bogus_key": 1}
        )
