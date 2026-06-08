"""Recipe registry: discover and look up recipes from YAML."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .recipe import Recipe

log = logging.getLogger(__name__)

DEFAULT_RECIPES_DIR = Path(__file__).parent / "recipes"


class RecipeRegistry:
    """In-memory registry of recipes keyed by id.

    Built from a directory of YAML files. Provides lookups by id and by
    ``(method_family, ood)`` -- the latter is what the legacy adapter uses
    to reproduce ``OOD_METHOD_MAP`` / ``NO_OOD_METHOD_MAP`` behavior.
    """

    def __init__(self, recipes: Dict[str, Recipe]):
        self._recipes = dict(recipes)

    @classmethod
    def load(cls, recipes_dir: Optional[Path] = None) -> "RecipeRegistry":
        recipes_dir = Path(recipes_dir) if recipes_dir else DEFAULT_RECIPES_DIR
        recipes: Dict[str, Recipe] = {}
        for path in sorted(recipes_dir.glob("*.yaml")):
            recipe = Recipe.from_yaml(path)
            if recipe.id in recipes:
                raise ValueError(f"Duplicate recipe id '{recipe.id}' in {path}")
            recipes[recipe.id] = recipe
        if not recipes:
            raise ValueError(f"No recipes found in {recipes_dir}")
        log.debug("Loaded %d recipes from %s", len(recipes), recipes_dir)
        return cls(recipes)

    def all(self) -> List[Recipe]:
        return list(self._recipes.values())

    def get(self, recipe_id: str) -> Recipe:
        try:
            return self._recipes[recipe_id]
        except KeyError:
            raise KeyError(
                f"Unknown recipe id '{recipe_id}'. Known: {sorted(self._recipes)}"
            )

    def has(self, recipe_id: str) -> bool:
        return recipe_id in self._recipes

    def find(self, method_family: str, ood: bool) -> Recipe:
        """Return the unique recipe for a ``(method_family, ood)`` pair."""
        matches = [
            r
            for r in self._recipes.values()
            if r.method_family == method_family and r.ood == ood
        ]
        if not matches:
            raise KeyError(
                f"No recipe for method_family='{method_family}', ood={ood}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous recipes for method_family='{method_family}', ood={ood}: "
                f"{[r.id for r in matches]}"
            )
        return matches[0]
