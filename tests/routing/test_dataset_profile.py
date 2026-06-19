"""DatasetProfile extraction (Phase 1) -- offline, deterministic."""

import glob

import pandas as pd
import pytest

from open_autonlu.routing import TaskSpec, extract_dataset_profile
from open_autonlu.routing.dataset_profile import DatasetProfile


def _balanced_df(n_per_class=20):
    rows = []
    for i in range(n_per_class):
        rows.append({"text": f"transfer money to my account number {i}", "label": "transfer"})
        rows.append({"text": f"what is the weather like today {i}", "label": "weather"})
        rows.append({"text": f"play some jazz music for me {i}", "label": "music"})
    return pd.DataFrame(rows)


def test_basic_counts_and_buckets():
    df = _balanced_df(20)
    p = extract_dataset_profile(df)
    assert isinstance(p, DatasetProfile)
    assert p.n_samples == 60
    assert p.n_classes == 3
    assert p.min_class_size == 20 and p.max_class_size == 20
    assert p.imbalance_ratio == 1.0
    assert p.imbalance_bucket == "low"
    assert p.size_bucket == "moderate"  # 10 <= 20 < 100
    assert 0.99 <= p.label_entropy <= 1.0  # perfectly balanced


def test_imbalance_detected():
    rows = [{"text": f"a sample {i}", "label": "big"} for i in range(100)]
    rows += [{"text": f"rare case {i}", "label": "small"} for i in range(4)]
    p = extract_dataset_profile(pd.DataFrame(rows))
    assert p.min_class_size == 4 and p.max_class_size == 100
    assert p.imbalance_ratio == 25.0
    assert p.imbalance_bucket == "high"
    assert p.size_bucket == "scarce"


def test_separable_vs_random():
    sep = extract_dataset_profile(_balanced_df(25)).tfidf_separability
    # Random text -> labels carry no signal -> low separability.
    import numpy as np
    rng = np.random.RandomState(0)
    vocab = [f"w{i}" for i in range(50)]
    rows = [
        {"text": " ".join(rng.choice(vocab, 5)), "label": rng.choice(["x", "y"])}
        for _ in range(200)
    ]
    rand = extract_dataset_profile(pd.DataFrame(rows)).tfidf_separability
    assert sep is not None and rand is not None
    assert sep > 0.8
    assert sep > rand


def test_anc_and_oos_flags():
    df = _balanced_df(10)
    df["anc_label"] = "some description"
    df.loc[0, "label"] = "outOfScope"
    p = extract_dataset_profile(df)
    assert p.has_anc_label is True
    assert p.has_oos_label is True
    assert p.has_hierarchy is True  # anc_label implies hierarchy signal


def test_hierarchical_taskspec_sets_hierarchy():
    p = extract_dataset_profile(_balanced_df(10), TaskSpec(label_schema="hierarchical"))
    assert p.has_hierarchy is True


def test_deterministic():
    df = _balanced_df(15)
    a = extract_dataset_profile(df, seed=7).to_dict()
    b = extract_dataset_profile(df, seed=7).to_dict()
    assert a == b


def test_missing_column_raises():
    with pytest.raises(KeyError):
        extract_dataset_profile(pd.DataFrame({"text": ["a"], "intent": ["x"]}))


def test_runs_on_fixture_csv():
    paths = glob.glob("examples/test_data/noise_n_shot_data/*train*.csv")
    if not paths:
        pytest.skip("fixture CSV not present")
    p = extract_dataset_profile(pd.read_csv(sorted(paths)[0]))
    assert p.n_classes >= 2
    assert p.tfidf_separability is None or 0.0 <= p.tfidf_separability <= 1.0
