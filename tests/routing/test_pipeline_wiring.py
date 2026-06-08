"""Phase 6: routing wired into TextClassificationTrainingPipeline.

Exercises method resolution (no model training) to prove compile_only resolves
the same Method class as legacy and that an ExecutionPlan is produced/persisted.
"""

import pandas as pd
import pytest

from open_autonlu.auto_classes import TextClassificationTrainingPipeline
from open_autonlu.methods.data_types import OodMethod
from open_autonlu.routing import ExecutionPlan


def _write_csv(tmp_path, n_per_class=90, classes=("transfer", "balance", "card")):
    rows = []
    for c in classes:
        for i in range(n_per_class):
            rows.append({"text": f"{c} request number {i}", "label": c})
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _pipeline(tmp_path, overrides):
    return TextClassificationTrainingPipeline(
        _write_csv(tmp_path), config_overrides=overrides
    )


def test_routing_mode_default_is_legacy(tmp_path):
    p = _pipeline(tmp_path, {"language": "en"})
    assert p._routing_mode() == "legacy"


def test_routing_mode_override(tmp_path):
    p = _pipeline(tmp_path, {"language": "en", "routing_mode": "compile_only"})
    assert p._routing_mode() == "compile_only"


@pytest.mark.parametrize(
    "ood_method,expected_cls",
    [
        (OodMethod.NONE, "Finetuner"),
        (OodMethod.AUTO, "FinetunerWithOOD"),
        (OodMethod.LOGIT, "Finetuner"),
    ],
)
def test_compile_only_resolves_same_class_as_legacy(tmp_path, ood_method, expected_cls):
    legacy = _pipeline(tmp_path, {"language": "en", "ood_method": ood_method})
    legacy_cls, _ = legacy._synergize_data_and_method()

    routed = _pipeline(
        tmp_path,
        {"language": "en", "ood_method": ood_method, "routing_mode": "compile_only"},
    )
    routed_cls, _ = routed._synergize_data_and_method()

    assert legacy_cls is routed_cls
    assert legacy_cls.__name__ == expected_cls


def test_execution_plan_populated(tmp_path):
    p = _pipeline(
        tmp_path,
        {"language": "en", "ood_method": OodMethod.NONE, "routing_mode": "compile_only"},
    )
    p._synergize_data_and_method()
    assert isinstance(p.execution_plan, ExecutionPlan)
    assert p.execution_plan.recipe_id == "finetuner"
    assert p.execution_plan.components["trainer"] == "Finetuner"
    assert p.execution_plan.notes["routing_mode"] == "compile_only"


def test_legacy_mode_also_records_plan(tmp_path):
    """Even legacy mode records a plan for observability/persistence."""
    p = _pipeline(tmp_path, {"language": "en", "ood_method": OodMethod.NONE})
    p._synergize_data_and_method()
    assert isinstance(p.execution_plan, ExecutionPlan)
    assert p.execution_plan.recipe_id == "finetuner"


def test_build_execution_plan_maps_ood_variants(tmp_path):
    p = _pipeline(tmp_path, {"language": "en"})
    plan_no = p._build_execution_plan("setfit", OodMethod.NONE)
    plan_ood = p._build_execution_plan("setfit", OodMethod.MSP_OOD)
    assert plan_no.components["trainer"] == "SetFitMethod"
    assert plan_ood.components["trainer"] == "SetFitOOD"


def test_execution_plan_save_load_roundtrip(tmp_path):
    plan = ExecutionPlan(
        recipe_id="finetuner",
        model_id="bert-base-uncased",
        components={"trainer": "Finetuner"},
        notes={"routing_mode": "compile_only"},
    )
    path = tmp_path / "execution_plan.json"
    plan.save(path)
    loaded = ExecutionPlan.load(path)
    assert loaded.recipe_id == "finetuner"
    assert loaded.components["trainer"] == "Finetuner"
