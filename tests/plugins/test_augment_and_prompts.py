"""Augmentation + prompt plugins (Phase 4)."""

import pytest

from open_autonlu.plugins import registry
from open_autonlu.plugins.augment.char_noise import CharNoiseAugmenter


def test_char_noise_wraps_word_char_augmentation():
    from open_autonlu.data.augmentations import WordCharAugmentation

    text = "transfer money to my account today"
    plugin_out = CharNoiseAugmenter().augment(text)
    direct_out = WordCharAugmentation()(text)
    assert isinstance(plugin_out, list)
    assert plugin_out == direct_out  # pure delegation


def test_augment_registry():
    assert "char_noise" in registry.available("augment")
    assert "llm" in registry.available("augment")


def test_char_noise_factory_builds_instance():
    factory = registry.get("augment", "char_noise")
    inst = factory()
    assert isinstance(inst, CharNoiseAugmenter)


def test_prompts_registered():
    from open_autonlu.plugins.prompts import DefaultPromptProvider  # noqa: F401

    assert "default" in registry.available("prompts")


def test_default_prompt_provider_delegates_get_prompts():
    from open_autonlu.plugins.prompts import DefaultPromptProvider

    provider = DefaultPromptProvider()
    try:
        ns = provider.prompts("en")
    except Exception as exc:  # noqa: BLE001 - optional heavy dep (outlines)
        pytest.skip(f"prompt backend unavailable: {exc}")
    assert ns is not None
