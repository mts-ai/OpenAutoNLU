"""Example: inspect the routing layer's compiled ExecutionPlan.

Run:
    python examples/routing_compile.py
"""

import pandas as pd

from open_autonlu.routing import (
    TaskSpec,
    compile_plan,
    extract_dataset_profile,
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
    print(
        f"  classes={profile.n_classes} min/class={profile.min_class_size} "
        f"separability={profile.tfidf_separability:.3f} "
        f"({profile.separability_bucket})"
    )

    plan = compile_plan(df, TaskSpec())
    print("\nExecutionPlan:")
    print(plan.to_json())


if __name__ == "__main__":
    main()
