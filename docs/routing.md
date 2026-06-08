# Routing layer: profile → constraints → probes → recipe

The routing layer decides *how* to train a model for a given
`(task, data, model)` triple. Historically this was a set of hard sample-count
thresholds inside `resolve_method`. The routing layer reorganizes that decision
into a small, declarative, empirically-grounded pipeline so it stays correct
across **any domain, language, and user-chosen base model**.

It is **fully backward compatible**: the default routing mode is `legacy`, which
reproduces the previous behavior byte-for-byte. Nothing changes unless you opt in.

```
TaskSpec ─┐
Dataset ──┼─▶ DatasetProfile ─┐
Model ────┘                   ├─▶ ConstraintEngine ─▶ ProbeRunner ─▶ PlanScorer ─▶ ExecutionPlan
              CapabilityProfile┘
```

---

## Quick start

Nothing is required — existing code keeps working:

```python
from open_autonlu.auto_classes import TextClassificationTrainingPipeline

# Legacy behavior (default). Identical to before the routing layer existed.
pipeline = TextClassificationTrainingPipeline("train.csv", "test.csv",
                                              config_overrides={"language": "en"})
pipeline.train()
```

Opt into the routing layer with a single config key:

```python
# compile_only: route the decision through the new layer.
# Produces the SAME method as legacy, plus a persisted ExecutionPlan.
pipeline = TextClassificationTrainingPipeline(
    "train.csv", "test.csv",
    config_overrides={"language": "en", "routing_mode": "compile_only"},
)
pipeline.train()
print(pipeline.execution_plan.recipe_id)        # e.g. "setfit"
pipeline.save("./model", save_format)           # writes execution_plan.json too
```

### Routing modes

| `routing_mode` | Behavior |
|----------------|----------|
| `legacy` (default) | Original resolver. Byte-identical to pre-routing behavior. |
| `compile_only` | Resolve the method through the routing layer. Provably the **same** method class as legacy; additionally records and persists an `ExecutionPlan`. |
| `full` | Empirical, probe-driven selection (see the compiler API). Standalone today via `compile_plan`; pipeline execution currently uses the parity path. |

---

## Inspecting a decision without training

`compile_plan` runs the whole pipeline and returns a serializable
`ExecutionPlan` — no model is trained:

```python
import pandas as pd
from open_autonlu.routing import TaskSpec, compile_plan
from open_autonlu.routing.task_spec import ROUTING_MODE_COMPILE_ONLY

df = pd.read_csv("train.csv")
plan = compile_plan(df, TaskSpec(routing_mode=ROUTING_MODE_COMPILE_ONLY))
print(plan.to_json())
```

```json
{
  "recipe_id": "setfit",
  "components": {"trainer": "SetFitMethod", "method_family": "setfit"},
  "dataset_profile_hash": "077e0ab0e15901f1",
  "notes": {"routing_mode": "compile_only"}
}
```

See `examples/routing_compile.py` for a runnable demo.

---

## Concepts

### TaskSpec — what you declare
`open_autonlu.routing.TaskSpec` captures intent and knobs (never a decision):
`task_type`, `objective` (multi-objective weights), `ood_policy`
(`none` / `logit_class` / `detector`), `anchors`, `language`, `model`
(`ModelConfig`), `budget` (`BudgetPolicy`), and `routing_mode`.

### DatasetProfile — cheap, data-only signals
`extract_dataset_profile(data)` computes class stats, imbalance ratio +
normalized entropy, text-length distribution, duplicate rate, `has_oos/anc/
hierarchy`, and a **TF-IDF separability proxy** — no encoder, deterministic.
Signals are *relative* (`low`/`medium`/`high`), never absolute router thresholds.

### CapabilityProfile — cheap, model-aware probes
`extract_capability_profile(data, model_config)` runs a frozen forward pass on
the **user's** encoder and reports kNN `separability_score` and a logistic-head
`linear_head_ceiling`. This is the model-aware replacement for sample-count
heuristics: the same data on a different encoder yields a different score. Inject
a custom `Embedder` (or use `HFEmbedder`); `BudgetPolicy.skip_probes` disables it.

### Recipes — declarative method definitions
Each YAML file in `open_autonlu/routing/recipes/` is a `Recipe`: a method family,
its `trainer` class, OOD wiring, soft data-regime preferences, and cost tier.
The shipped recipes mirror the existing methods exactly:

| Recipe | Trainer | Regime (`n_min`) | OOD |
|--------|---------|------------------|-----|
| `anc_setfit` / `anc_setfit_ood` | `AncSetFitMethod` | 2–5 | needs `anc_label` |
| `setfit` / `setfit_ood` | `SetFitMethod` / `SetFitOOD` | 6–80 | MSP |
| `finetuner` / `finetuner_ood` | `Finetuner` / `FinetunerWithOOD` | >80 | Mahalanobis |

**Adding a method = adding a YAML file.** The central router is not edited.

### ConstraintEngine — guardrails
Hard filters before any probe: OOD policy gating, `anc_label` requirement, the
2-samples-per-class floor, and user pins (`config_overrides={"recipe_id": ...}`
wins — "explicit beats implicit"). Soft-ranks survivors by regime fit.

### ProbeRunner + PlanScorer — empirical selection (`full` mode)
`ProbeRunner` validates candidates (default: a cheap frozen-capability probe;
inject a `probe_fn` for micro-training). `PlanScorer` combines the probe signal
with structural priors (regime fit, cost tier) under `TaskSpec.objective` and
returns the winner plus the margin to the runner-up.

### ExecutionPlan — the persisted artifact
The output of routing is a versioned `ExecutionPlan` (`recipe_id`, `model_id`,
`components`, `probe_scores`, `dataset_profile_hash`, `selection_margin`). It is
saved as `execution_plan.json` next to the model, so a routing decision is
reproducible and diffable across model swaps.

---

## Plugins (language/domain isolation)

Locale- and domain-specific behavior lives behind protocols in
`open_autonlu.plugins`, selected by name — the router never imports Russian/
English strings or generators directly.

| Kind | Protocol | Built-ins |
|------|----------|-----------|
| `ood_sampling` | `OodSampler` | `gibberish` (legacy default), `tiered` (close/mid/far/very-far) |
| `augment` | `TextAugmenter` | `char_noise`, `llm` |
| `prompts` | `PromptProvider` | `default` (language-keyed) |

Swap the OOD sampler via config (the default reproduces the legacy gibberish
generator byte-for-byte):

```python
config_overrides = {"ood_method": OodMethod.MSP_OOD, "ood_sampler": "tiered"}
# or inject an instance:
from open_autonlu.plugins.ood_sampling import GibberishOodSampler
config_overrides = {"ood_sampler": GibberishOodSampler(seed=0, language="en")}
```

---

## Design principles

1. **Never route on absolute sample counts alone** — normalize by the
   separability probe on the chosen encoder.
2. **Never commit a recipe without validation** (probes) unless the user pins one.
3. **Never embed locale/domain in the router** — only in plugins.
4. **Explicit user overrides beat compiled plans.**
5. **Incremental & reversible** — `legacy` stays the default; the existing
   `Method` classes are untouched; selection simply moved out.

---

## Known limitations / roadmap

- `full`-mode **pipeline execution** currently uses the parity path; probe-driven
  method override with matching data processing is the next increment.
- `OOD_METHOD_MAP["ancsetfit"]` historically maps to `AncSetFitMethod` (not
  `AncSetFitOOD`). The routing layer reproduces this for parity
  (`anc_setfit_ood.yaml`); fixing it is a deliberate, separately-versioned change.
- NER routing is stubbed (`TaskSpec.task_type`); text classification is wired
  first.
