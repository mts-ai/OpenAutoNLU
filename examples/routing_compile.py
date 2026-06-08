"""Example: inspect the routing layer's compiled ExecutionPlan.

Shows the new routing layer (arch_suggestion.md) without training a model:
profile a dataset, compile a plan in each routing mode, and print the result.

Run:
    python examples/routing_compile.py
"""

import pandas as pd

from open_autonlu.routing import (
    TaskSpec,
    compile_plan,
    extract_dataset_profile,
)
from open_autonlu.routing.task_spec import (
    ROUTING_MODE_COMPILE_ONLY,
    ROUTING_MODE_FULL,
)


def _toy_dataset(n_per_class=120):
    rows = []
    for c in ("transfer", "balance", "card_lost"):
        for i in range(n_per_class):
            rows.append({"text": f"{c} request example number {i}", "label": c})
    return pd.DataFrame(rows)


def main():
    df = _toy_dataset()

    profile = extract_dataset_profile(df)
    print("DatasetProfile:")
    print(f"  classes={profile.n_classes} min/class={profile.min_class_size} "
          f"size_bucket={profile.size_bucket} "
          f"separability={profile.tfidf_separability:.3f} "
          f"({profile.separability_bucket})")

    # compile_only: deterministic, parity with the legacy resolver.
    plan = compile_plan(df, TaskSpec(routing_mode=ROUTING_MODE_COMPILE_ONLY))
    print("\nExecutionPlan (compile_only):")
    print(plan.to_json())

    # full: constraint-filtered + (optionally) probe-driven selection.
    plan_full = compile_plan(df, TaskSpec(routing_mode=ROUTING_MODE_FULL))
    print("\nExecutionPlan (full, no encoder probe):")
    print(f"  recipe_id={plan_full.recipe_id} "
          f"margin={plan_full.selection_margin:.3f} "
          f"candidates={plan_full.notes.get('n_candidates')}")


if __name__ == "__main__":
    main()
