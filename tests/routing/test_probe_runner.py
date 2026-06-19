"""Recipe-specific encoder probes (Stage 1)."""

import numpy as np
import pandas as pd

from open_autonlu.routing import RecipeRegistry
from open_autonlu.routing.probe_runner import (
    ProbeContext,
    recipe_encoder_probe,
)
from open_autonlu.routing.recipe import ProbeConfig, Recipe


def _df(n_per_class=40):
    rows = []
    for c in ("alpha", "beta", "gamma"):
        for i in range(n_per_class):
            rows.append({"text": f"{c} item {i}", "label": c})
    return pd.DataFrame(rows)


class SeparableStub:
    def encode(self, texts):
        labels = [t.split()[0] for t in texts]
        uniq = sorted(set(labels))
        idx = {c: i for i, c in enumerate(uniq)}
        X = np.zeros((len(texts), len(uniq)), dtype=np.float32)
        for r, lab in enumerate(labels):
            X[r, idx[lab]] = 1.0
        return X


class NoisyStub:
    def __init__(self, dim=16, seed=0):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def encode(self, texts):
        n = len(texts)
        return self.rng.randn(n, self.dim).astype(np.float32)


def _recipe(recipe_id, probe_type):
    base = RecipeRegistry.load().get(recipe_id)
    return Recipe(
        id=base.id,
        method_family=base.method_family,
        trainer=base.trainer,
        ood=base.ood,
        cost_tier=base.cost_tier,
        requires_anc_label=base.requires_anc_label,
        probe=ProbeConfig(probe_type=probe_type),
    )


def test_recipe_probes_differ_on_same_embeddings():
    df = _df()
    emb = SeparableStub()
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    ctx = ProbeContext(
        embeddings=emb.encode(texts),
        labels=labels,
    )
    few_shot = recipe_encoder_probe(
        _recipe("anc_setfit", "few_shot_knn"), ctx
    )
    knn = recipe_encoder_probe(_recipe("setfit", "knn"), ctx)
    linear = recipe_encoder_probe(_recipe("finetuner", "linear"), ctx)
    assert few_shot.in_scope_f1 is not None
    assert knn.in_scope_f1 is not None
    assert linear.in_scope_f1 is not None
    # On separable one-hot data, full kNN/linear beat capped few-shot training.
    assert knn.in_scope_f1 >= few_shot.in_scope_f1
    assert linear.in_scope_f1 >= few_shot.in_scope_f1


def test_probe_types_diverge_on_noisy_embeddings():
    df = _df(n_per_class=25)
    emb = NoisyStub()
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    ctx = ProbeContext(embeddings=emb.encode(texts), labels=labels)
    scores = {
        recipe_encoder_probe(_recipe(rid, ptype), ctx).in_scope_f1
        for rid, ptype in (
            ("anc_setfit", "few_shot_knn"),
            ("setfit", "knn"),
            ("finetuner", "linear"),
        )
    }
    assert len({round(s, 3) for s in scores if s is not None}) > 1
