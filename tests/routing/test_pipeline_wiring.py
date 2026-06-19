"""Phase 6: routing wired into TextClassificationTrainingPipeline."""

import numpy as np
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


class SeparableStub:
    def encode(self, texts):
        labels = [t.split()[0] for t in texts]
        uniq = sorted(set(labels))
        idx = {c: i for i, c in enumerate(uniq)}
        X = np.zeros((len(texts), len(uniq)), dtype=np.float32)
        for r, lab in enumerate(labels):
            X[r, idx[lab]] = 1.0
        return X


@pytest.fixture(autouse=True)
def _stub_encoder_probe(monkeypatch):
    """Avoid loading HF weights in pipeline routing tests."""

    def _fake_compile(dataset, task_spec=None, **kwargs):
        kwargs.setdefault("embedder", SeparableStub())
        from open_autonlu.routing.compiler import compile_plan as real_compile

        return real_compile(dataset, task_spec, **kwargs)

    monkeypatch.setattr(
        "open_autonlu.auto_classes.text_classification_pipeline.compile_plan",
        _fake_compile,
    )


def test_routing_mode_default_is_full(tmp_path):
    p = _pipeline(tmp_path, {"language": "en"})
    assert p._routing_mode() == "full"


@pytest.mark.parametrize(
    "ood_method,expected_cls",
    [
        (OodMethod.NONE, "SetFitMethod"),
        (OodMethod.AUTO, "SetFitOOD"),
        (OodMethod.LOGIT, "SetFitMethod"),
    ],
)
def test_routing_resolves_method_class(tmp_path, ood_method, expected_cls):
    routed = _pipeline(tmp_path, {"language": "en", "ood_method": ood_method})
    routed_cls, _ = routed._synergize_data_and_method()
    assert routed_cls.__name__ == expected_cls


def test_execution_plan_populated(tmp_path):
    p = _pipeline(tmp_path, {"language": "en", "ood_method": OodMethod.NONE})
    p._synergize_data_and_method()
    assert isinstance(p.execution_plan, ExecutionPlan)
    assert p.execution_plan.recipe_id == "setfit"
    assert p.execution_plan.components["trainer"] == "SetFitMethod"


def test_pinned_recipe_id(tmp_path):
    p = _pipeline(
        tmp_path,
        {"language": "en", "ood_method": OodMethod.NONE, "recipe_id": "finetuner"},
    )
    p._synergize_data_and_method()
    assert p.execution_plan.recipe_id == "finetuner"
    cls, _ = p._synergize_data_and_method()
    assert cls.__name__ == "Finetuner"


def test_execution_plan_save_load_roundtrip(tmp_path):
    plan = ExecutionPlan(
        recipe_id="finetuner",
        model_id="bert-base-uncased",
        components={"trainer": "Finetuner"},
        notes={"selection": "probe"},
    )
    path = tmp_path / "execution_plan.json"
    plan.save(path)
    loaded = ExecutionPlan.load(path)
    assert loaded.recipe_id == "finetuner"
    assert loaded.components["trainer"] == "Finetuner"
