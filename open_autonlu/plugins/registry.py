"""Plugin lookup by name (+ optional language/script hints).

Phase 0 ships a minimal name-keyed registry. Concrete implementations are
registered in Phase 4 when gibberish/char-noise/LLM/prompt plugins are
extracted from the trainers and ``method_resolver``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

_REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {
    "augment": {},
    "ood_sampling": {},
    "prompts": {},
    "gibberish": {},
}


def register(kind: str, name: str, factory: Callable[..., Any]) -> None:
    """Register a plugin ``factory`` under ``(kind, name)``."""
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown plugin kind '{kind}'. Known: {sorted(_REGISTRY)}")
    _REGISTRY[kind][name] = factory


def get(kind: str, name: str) -> Callable[..., Any]:
    """Look up a registered plugin factory."""
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown plugin kind '{kind}'. Known: {sorted(_REGISTRY)}")
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(
            f"No plugin '{name}' of kind '{kind}'. "
            f"Registered: {sorted(_REGISTRY[kind])}"
        )


def available(kind: str) -> list[str]:
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown plugin kind '{kind}'. Known: {sorted(_REGISTRY)}")
    return sorted(_REGISTRY[kind])
