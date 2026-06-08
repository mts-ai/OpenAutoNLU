"""Parity: routing legacy_adapter must resolve the same Method class as the
existing pipeline logic for every (min_class_size, ood_method, anc) combo.
"""

import pytest

from open_autonlu.constants import (
    ANCSETFIT_NAME,
    NO_OOD_METHOD_MAP,
    OOD_METHOD_MAP,
    SETFIT_NAME,
)
from open_autonlu.method_resolver import resolve_method
from open_autonlu.methods.data_types import OodMethod
from open_autonlu.routing import RecipeRegistry
from open_autonlu.routing import legacy_adapter as la


@pytest.fixture(scope="module")
def registry():
    return RecipeRegistry.load()


def _legacy_expected_trainer(min_size, ood_method, anc_present):
    """Reproduce _synergize_data_and_method's class resolution verbatim."""
    family = resolve_method(min_size)
    if family == ANCSETFIT_NAME and not anc_present:
        family = SETFIT_NAME
    if ood_method in (OodMethod.NONE, OodMethod.LOGIT):
        return NO_OOD_METHOD_MAP[family]
    return OOD_METHOD_MAP[family]


@pytest.mark.parametrize("min_size", [2, 3, 5, 6, 10, 40, 80, 81, 100, 500])
@pytest.mark.parametrize("ood_method", list(OodMethod))
@pytest.mark.parametrize("anc_present", [True, False])
def test_trainer_class_name_matches_legacy(registry, min_size, ood_method, anc_present):
    expected = _legacy_expected_trainer(min_size, ood_method, anc_present)
    got = la.trainer_class_name(registry, min_size, ood_method, anc_present)
    assert got == expected


def test_documented_regime_boundaries(registry):
    """Pin the true thresholds (note: 80 -> setfit, 81 -> finetuning)."""
    name = lambda n: la.trainer_class_name(  # noqa: E731
        registry, n, OodMethod.NONE, has_anc_label=True
    )
    assert name(5) == "AncSetFitMethod"
    assert name(10) == "SetFitMethod"
    assert name(80) == "SetFitMethod"
    assert name(81) == "Finetuner"


def test_ancsetfit_ood_keeps_legacy_bug_class(registry):
    """OOD ancsetfit currently maps to AncSetFitMethod, not AncSetFitOOD."""
    assert OOD_METHOD_MAP[ANCSETFIT_NAME] == "AncSetFitMethod"
    got = la.trainer_class_name(registry, 3, OodMethod.MSP_OOD, has_anc_label=True)
    assert got == "AncSetFitMethod"


def test_is_ood_enabled():
    assert not la.is_ood_enabled(OodMethod.NONE)
    assert not la.is_ood_enabled(OodMethod.LOGIT)
    assert la.is_ood_enabled(OodMethod.AUTO)
    assert la.is_ood_enabled(OodMethod.MARGINAL_MAHALANOBIS_OOD)
    assert la.is_ood_enabled(OodMethod.MSP_OOD)


def test_min_size_below_floor_raises(registry):
    with pytest.raises(ValueError):
        la.trainer_class_name(registry, 1, OodMethod.NONE, has_anc_label=True)
