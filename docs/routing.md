# Routing layer: profile → constraints → probes → recipe

The routing layer decides *how* to train a model for a given
`(task, data, model)` triple using **empirical encoder probes** — not
sample-count heuristics.

```
TaskSpec ─┐
Dataset ──┼─▶ DatasetProfile ─┐
Model ────┘                   ├─▶ ConstraintEngine ─▶ ProbeRunner ─▶ PlanScorer ─▶ ExecutionPlan
              CapabilityProfile┘
```

---

## Quick start

```python
from open_autonlu.auto_classes import TextClassificationTrainingPipeline

pipeline = TextClassificationTrainingPipeline(
    "train.csv", "test.csv",
    config_overrides={"language": "en", "model_name_or_path": "bert-base-uncased"},
)
pipeline.train()
print(pipeline.execution_plan.recipe_id)
```

Inspect a decision without training:

```python
import pandas as pd
from open_autonlu.routing import TaskSpec, compile_plan, ModelConfig

df = pd.read_csv("train.csv")
plan = compile_plan(
    df,
    TaskSpec(model=ModelConfig(encoder="bert-base-uncased")),
)
print(plan.to_json())
```

Pin a recipe explicitly:

```python
config_overrides = {"recipe_id": "setfit", "ood_method": OodMethod.NONE}
```

---

## Concepts

### TaskSpec
User intent: `ood_policy`, `language`, `model`, `objective` weights, `budget`.

### DatasetProfile
Cheap data-only signals (class stats, imbalance, TF-IDF separability). Used for
metadata and constraints — **not** for sample-count routing.

### CapabilityProfile / ProbeRunner
Frozen encoder embeddings + **recipe-specific** probes:
- `anc_setfit` → few-shot kNN
- `setfit` → kNN
- `finetuner` → linear head CV

### Recipes
YAML files in `open_autonlu/routing/recipes/`. Declare trainer, OOD wiring,
`probe`, and `data_prep` (downsample caps). Adding a method = adding a YAML file.

### ConstraintEngine
Hard filters only: OOD policy, `anc_label` requirement, ≥2 samples/class floor,
user `recipe_id` pin.

### ExecutionPlan
Persisted as `execution_plan.json` next to the saved model.

---

## Plugins (language/domain isolation)

Locale-specific behavior lives in `open_autonlu/plugins`, selected by name —
the router never imports domain strings directly.

---

## Design principles

1. **Never route on absolute sample counts** — encoder probes decide.
2. **Never commit a recipe without validation** unless the user pins one.
3. **Never embed locale/domain in the router** — only in plugins.
4. **Explicit user overrides beat compiled plans.**
