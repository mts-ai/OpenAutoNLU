"""Recipe-driven data preparation."""

import pandas as pd
import pytest
from datasets import Dataset

from open_autonlu.routing import RecipeRegistry
from open_autonlu.routing.data_prep import apply_data_prep


def _df(n_per_class=100):
    rows = []
    for c in ("a", "b"):
        for i in range(n_per_class):
            rows.append({"text": f"{c} sample {i}", "label": c})
    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def test_setfit_recipe_downsamples():
    recipe = RecipeRegistry.load().get("setfit")
    out = apply_data_prep(_df(100), recipe)
    counts = out.to_pandas().groupby("label").size()
    assert counts.max() <= 80
