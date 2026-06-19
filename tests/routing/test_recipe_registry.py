"""Recipe loading + registry lookups."""

import pytest

from open_autonlu.constants import ANCSETFIT_NAME, FINETUNING_NAME, SETFIT_NAME
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


def test_recipes_have_expected_trainers(registry):
    assert registry.get("finetuner").trainer == "Finetuner"
    assert registry.get("setfit_ood").trainer == "SetFitOOD"
    assert registry.get("anc_setfit").trainer == "AncSetFitMethod"


def test_probe_types_assigned_per_trainer(registry):
    assert registry.get("anc_setfit").probe.probe_type == "few_shot_knn"
    assert registry.get("setfit").probe.probe_type == "knn"
    assert registry.get("finetuner").probe.probe_type == "linear"


def test_unknown_recipe_key_rejected():
    with pytest.raises(ValueError):
        Recipe.from_dict(
            {"id": "x", "method_family": "setfit", "trainer": "SetFitMethod",
             "bogus_key": 1}
        )
