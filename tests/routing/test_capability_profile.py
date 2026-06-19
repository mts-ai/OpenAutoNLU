"""CapabilityProfile probes (Phase 2).

Unit tests inject a stub Embedder so the probe logic (kNN, linear head,
bucketing, budget gating) runs offline and deterministically. A separate test
exercises a real tiny HF model but is skipped if it can't be loaded (offline).
"""

import numpy as np
import pandas as pd
import pytest

from open_autonlu.routing import (
    BudgetPolicy,
    ModelConfig,
    extract_capability_profile,
)
from open_autonlu.routing.capability_profile import CapabilityProfile


def _df(n_per_class=30, classes=("a", "b", "c")):
    rows = []
    for c in classes:
        for i in range(n_per_class):
            rows.append({"text": f"{c} text example {i}", "label": c})
    return pd.DataFrame(rows)


class SeparableStub:
    """Embeds each label near a distinct one-hot centroid -> high separability."""

    def __init__(self, noise=0.01, seed=0):
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self._label_for = {}

    def encode(self, texts):
        # First whitespace token is the label in the fixtures above.
        labels = [t.split()[0] for t in texts]
        uniq = sorted(set(labels))
        idx = {c: i for i, c in enumerate(uniq)}
        dim = len(uniq)
        X = np.zeros((len(texts), dim), dtype=np.float32)
        for r, lab in enumerate(labels):
            X[r, idx[lab]] = 1.0
        return X + self.noise * self.rng.randn(*X.shape).astype(np.float32)


class RandomStub:
    """Embeds everything randomly -> no class signal -> low separability."""

    def __init__(self, dim=8, seed=0):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def encode(self, texts):
        return self.rng.randn(len(texts), self.dim).astype(np.float32)


def test_separable_stub_high_scores():
    p = extract_capability_profile(_df(), embedder=SeparableStub())
    assert isinstance(p, CapabilityProfile)
    assert p.probed is True
    assert p.embedding_dim == 3
    assert p.separability_score is not None and p.separability_score > 0.9
    assert p.linear_head_ceiling is not None and p.linear_head_ceiling > 0.9
    assert p.separability_bucket == "high"


def test_random_stub_low_scores():
    p = extract_capability_profile(_df(), embedder=RandomStub())
    assert p.separability_score is not None
    assert p.separability_score < 0.6
    assert p.separability_bucket in {"low", "medium"}


def test_different_model_yields_different_separability():
    """Acceptance: same data, different encoder -> different separability."""
    good = extract_capability_profile(_df(), embedder=SeparableStub())
    bad = extract_capability_profile(_df(), embedder=RandomStub())
    assert good.separability_score > bad.separability_score


def test_budget_skip_probes_returns_unprobed():
    p = extract_capability_profile(
        _df(),
        model_config=ModelConfig(encoder="some/model"),
        budget=BudgetPolicy(skip_probes=True),
    )
    assert p.probed is False
    assert p.separability_score is None
    assert p.model_id == "some/model"
    assert p.notes.get("reason") == "budget.skip_probes"


def test_no_embedder_and_no_model_raises():
    with pytest.raises(ValueError):
        extract_capability_profile(_df(), model_config=ModelConfig(encoder=None))


def test_model_id_recorded_from_config():
    p = extract_capability_profile(
        _df(), model_config=ModelConfig(encoder="my/enc"), embedder=SeparableStub()
    )
    assert p.model_id == "my/enc"


@pytest.mark.slow
def test_real_tiny_hf_model():
    """Optional end-to-end with a real (tiny) encoder; skip if unavailable."""
    from open_autonlu.routing import HFEmbedder

    try:
        emb = HFEmbedder("hf-internal-testing/tiny-random-BertModel", max_seq_length=32)
    except Exception as exc:  # noqa: BLE001 - offline / download failure
        pytest.skip(f"tiny HF model unavailable: {exc}")
    p = extract_capability_profile(_df(10), embedder=emb)
    assert p.probed is True
    assert p.embedding_dim and p.embedding_dim > 0
