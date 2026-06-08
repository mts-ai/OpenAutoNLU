"""OOD-sampling plugins (Phase 4) -- gibberish parity is the key gate."""

import pytest

from open_autonlu.methods.utils.random_sentence_generator import (
    GibberishDatasetGenerator,
)
from open_autonlu.plugins import registry
from open_autonlu.plugins.ood_sampling import (
    GibberishOodSampler,
    resolve_ood_sampler,
)
from open_autonlu.plugins.ood_sampling.tiered import (
    TieredOodSampler,
    p95_budget,
    split_equally,
)


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("n", [1, 20, 137])
def test_gibberish_byte_parity_with_legacy(seed, n):
    """The default sampler must reproduce the legacy generator exactly.

    This is what guarantees `_generate_synthetic_oos` is unchanged by Phase 4.
    """
    legacy = GibberishDatasetGenerator(seed)(num_rows=n)
    via_plugin = GibberishOodSampler(seed=seed).sample(n)
    via_resolver = resolve_ood_sampler(None, seed).sample(n)
    assert via_plugin == legacy
    assert via_resolver == legacy


def test_resolve_string_matches_default():
    assert resolve_ood_sampler("gibberish", 1).sample(5) == GibberishOodSampler(
        seed=1
    ).sample(5)


def test_resolve_instance_passthrough():
    s = GibberishOodSampler(seed=3)
    assert resolve_ood_sampler(s, 99) is s


def test_resolve_unknown_name_raises():
    with pytest.raises(KeyError):
        resolve_ood_sampler("not_a_sampler", 0)


def test_resolve_bad_type_raises():
    with pytest.raises(TypeError):
        resolve_ood_sampler(1234, 0)


def test_registry_registers_builtins():
    assert "gibberish" in registry.available("ood_sampling")
    assert "tiered" in registry.available("ood_sampling")


def test_en_gibberish_differs_from_legacy():
    legacy = GibberishOodSampler(seed=5).sample(10)
    en = GibberishOodSampler(seed=5, language="en").sample(10)
    assert en != legacy  # different alphabet


def test_tiered_very_far_self_contained():
    t = TieredOodSampler(seed=0)
    assert len(t.sample(5, "very_far")) == 5


def test_tiered_mid_requires_pool():
    with pytest.raises(ValueError):
        TieredOodSampler(seed=0).sample(3, "mid")


def test_tiered_mid_with_pool():
    t = TieredOodSampler(seed=0, pools={"mid": list("abcde")})
    out = t.sample(3, "mid")
    assert len(out) == 3 and set(out) <= set("abcde")


def test_tiered_unknown_tier_raises():
    with pytest.raises(ValueError):
        TieredOodSampler(seed=0).sample(2, "nonsense")


def test_p95_budget_and_split_equally():
    assert p95_budget([10, 10, 10, 100]) >= 10
    split = split_equally(10, ["a", "b", "c"])
    assert sum(split.values()) == 10
    assert set(split) == {"a", "b", "c"}
