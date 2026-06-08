# OpenAutoNLU Pipeline

[![arXiv](https://img.shields.io/badge/arXiv-2603.01824-b31b1b.svg)](https://arxiv.org/abs/2603.01824)
[![PyPI](https://img.shields.io/pypi/v/open-autonlu?label=PyPI%20package&color=green)](https://pypi.org/project/open-autonlu/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)

OpenAutoNLU is an open-source pipeline for training natural language understanding (NLU) models for **text classification** (multiclass) and **named entity recognition (NER)**. It supports few-shot learning (SetFit, AncSetFit with optional anchor labels), classic fine-tuning, data quality diagnostics, out-of-distribution (OOD) detection, optional LLM-based augmentation and synthetic test generation, and ONNX export for deployment.

You provide train (and optionally test) data; high-level pipelines (`TextClassificationTrainingPipeline`, `TokenClassificationTrainingPipeline`) load it, run optional data-quality checks, then **automatically choose the training method** from the data: **AncSetFit** for very small datasets (2–5 samples per class), **SetFit** for medium size (6–80), and **fine-tuning** for larger data. You can override configs (batch size, OOD method, augmentation, etc.) and save models in ONNX format. A Streamlit app and Docker images (CPU/GPU) are included for interactive use.

Built by MWS AI and contributors (see [pyproject.toml](pyproject.toml) for authors). Aimed at practitioners and researchers who want a single, data-driven workflow for few-shot and full-size NLU training without manually picking methods or tuning low-level knobs.

!requires Python >=3.12, <3.13

Usage examples are located in the `examples` folder.

## Installation

Install from PyPI:
```bash
pip install "open-autonlu[cpu]"
# or with CUDA support:
pip install "open-autonlu[cuda]"
```

Development mode:

To work with the repository in developer mode, install it as an editable package:
```bash
pip install -e .
```

This way you don't need to reinstall the package after code changes. To install all dependencies (two configurations are available: cpu and cuda) for development, run:
```bash
uv sync --extra cuda
```

## Documentation

To build and view the documentation locally:
```bash
uv sync
cd docs && uv run make html
open build/html/index.html
```
## Running with Docker


**With GPU** (recommended host: 16GB RAM, 8 CPU, A100 40GB, ~30GB disk):

```bash
docker-compose up -d
```

**Without GPU** (macOS or CPU-only):

```bash
docker build --build-arg EXTRA=cpu -t open-autonlu .
docker run -p 8501:8501 open-autonlu
```

## Code example with default parameters:

### Training
```python
from open_autonlu.auto_classes import (
    TextClassificationTrainingPipeline,
    TokenClassificationTrainingPipeline
)
from open_autonlu.methods.data_types import SaveFormat

# Text Classification training
pipeline = TextClassificationTrainingPipeline(
    train_path="train.csv",
    test_path="test.csv",
    config_overrides={"language": "en"}  # for non-en/ru also set "model_name_or_path"
)
result = pipeline.train()
pipeline.save("./model", SaveFormat.ONNX)

# NER training
pipeline = TokenClassificationTrainingPipeline(
    train_path="train.json",
    test_path="test.json",
    config_overrides={"language": "en"}  # for non-en/ru also set "model_name_or_path"
)
result = pipeline.train()
pipeline.save("./model", SaveFormat.ONNX)
```


### Inference
```python
from open_autonlu.auto_classes import (
    TextClassificationInferenceManager,
    TokenClassificationInferenceManager
)

# Text Classification inference
inferer = TextClassificationInferenceManager("./model")
results = inferer.predict(["Hello world", "Goodbye"], batch_size=32)
for r in results:
    print(f"{r.most_probable.label}: {r.most_probable.score:.3f}")

# NER inference
ner_inferer = TokenClassificationInferenceManager("./ner_model")
results = ner_inferer.predict(["John works at Google"], batch_size=1)
for r in results:
    for entity in r.labels:
        print(f"{entity.text}: {entity.label}")
```

## Data Quality Diagnostics

The `diagnose()` method evaluates training data quality using multiple evaluators:
- `cartography` (MulticlassCLF) [Dataset Cartography](https://aclanthology.org/2020.emnlp-main.746.pdf)
- `vinfo` (MulticlassCLF) [V-Usable information](https://arxiv.org/abs/2110.08420)
- `uncertainty` (MulticlassCLF, NER)
- `retag` (MulticlassCLF, NER)
- `label aggregation` (NER)

Run the data quality stage:
```python
from open_autonlu.auto_classes import TextClassificationTrainingPipeline

pipeline = TextClassificationTrainingPipeline(train_path="train.csv")
evaluation_result = pipeline.diagnose()
```

## Configuration Overrides

The `config_overrides` parameter allows you to customize training behavior with modifying default configurations.

### Basic Usage

```python
from open_autonlu.auto_classes import TextClassificationTrainingPipeline
from open_autonlu.methods.data_types import OodMethod, SaveFormat

pipeline = TextClassificationTrainingPipeline(
    train_path="train.csv",
    config_overrides={
        "language": "en",                # for non-en/ru also set "model_name_or_path"
        "ood_method": OodMethod.LOGIT,   # OOD detection method
        "batch_size": 32,                # Batch size
    }
)
result = pipeline.train()
pipeline.save("./model", SaveFormat.ONNX)
```

### OOD Detection Methods

Out-of-Distribution detection identifies inputs that don't belong to any trained class.

| Method | Description | Best for |
|--------|-------------|----------|
| `OodMethod.AUTO` | Auto-select based on training method | Default |
| `OodMethod.MARGINAL_MAHALANOBIS_OOD` | Mahalanobis distance from embedding distribution | Finetuning |
| `OodMethod.MSP_OOD` | Maximum Softmax Probability threshold | SetFit, AncSetFit |
| `OodMethod.LOGIT` | Adds `outOfScope` class during training | Alternative approach |
| `OodMethod.NONE` | Disable OOD detection | When not needed |

The `threshold_factor` parameter controls OOD detection sensitivity. It is a multiplier applied to the OOD detection threshold. Higher values make detection more conservative (fewer samples are marked as OOD), while lower values make it more aggressive (more samples are flagged as OOD). Default value is `1.0`.

```python
from open_autonlu.methods.data_types import OodMethod

# Override ood_method and adjust sensitivity
config_overrides = {
    "ood_method": OodMethod.MARGINAL_MAHALANOBIS_OOD,
    "threshold_factor": 1.5,  # More conservative OOD detection
}
```

### Routing layer (method selection)

By default the library auto-selects the training method (AncSetFit / SetFit /
Finetuning) from the data. The optional **routing layer** reorganizes that
decision into a declarative `profile → constraints → probes → recipe` pipeline
that stays robust across domains, languages, and base models. It is fully
backward compatible — `routing_mode` defaults to `legacy`.

```python
pipeline = TextClassificationTrainingPipeline(
    train_path="train.csv",
    config_overrides={
        "language": "en",
        "routing_mode": "compile_only",  # legacy | compile_only | full
    },
)
pipeline.train()
print(pipeline.execution_plan.recipe_id)   # selected recipe, persisted as execution_plan.json
```

Inspect a decision without training via `compile_plan` (see
`examples/routing_compile.py`). Full details and the plugin system
(`ood_sampling`, `augment`, `prompts`) are documented in
[`docs/routing.md`](docs/routing.md).

### LLM Data Augmentation

Automatically augment underrepresented classes using LLM generation. The `language` parameter controls which prompts are sent to the LLM (`"en"` for English, `"ru"` for Russian). For other languages, English prompts are used with an instruction to generate text in the language of the provided examples.

```python
import os
from open_autonlu.auto_classes import TextClassificationTrainingPipeline

pipeline = TextClassificationTrainingPipeline(
    train_path="train.csv",
    config_overrides={
        "language": "en",
        "llm_augmentation": {
            "enabled": True,
            "use_domain_analysis": True,  # Analyze domain for better prompts
            "threshold": 81,               # Augment classes with < 81 samples
            "max_attempts": 10,            # Max generation attempts
            "num_shot": 5,                 # Examples in prompt
            "config_overrides": {
                "LlmClientConfig": {
                    "api_key": os.environ["MODEL_API_KEY"],
                    "model_id": "gpt-4",
                }
            }
        }
    }
)
```

### Synthetic Test Generation

Generate synthetic test data using LLM when no test set is provided.

```python
import os
from open_autonlu.auto_classes import TextClassificationTrainingPipeline

pipeline = TextClassificationTrainingPipeline(
    train_path="train.csv",  # No test_path provided
    config_overrides={
        "language": "en",  
        "llm_test_generation": {
            "enabled": True,
            "num_samples_per_class": 100,
            "use_domain_analysis": True,
            "synthetic_test_path": "./synthetic_test.csv",  # Save generated data
            "config_overrides": {
                "LlmClientConfig": {
                    "api_key": os.environ["MODEL_API_KEY"],
                    "model_id": "gpt-4",
                }
            }
        }
    }
)
result = pipeline.train()  # Test data generated automatically
```

### Method-Specific Overrides

You can override default parameters for a specific training method by using the method class name as a key. Note that **overriding a method's defaults does not force the pipeline to use that method** — the pipeline always selects the method automatically based on the dataset. The overrides will only take effect if the pipeline selects that particular method.

Available method keys:
- Base methods: `SetFitMethod`, `AncSetFitMethod`, `Finetuner`, `TokenClassificationFinetuner`
- OOD methods: `SetFitOOD`, `AncSetFitOOD`, `FinetunerWithOOD`

All overridable parameters are defined in the corresponding config classes in `open_autonlu/methods/configs/`.

```python
# SetFit configuration
config_overrides = {
    "SetFitMethod": {
        "num_iterations": 25,
        "body_lr": 2e-5,
        "batch_size": 16,
    }
}

# Finetuner configuration
config_overrides = {
    "Finetuner": {
        "num_hpo_trials": 15,  # Hyperparameter optimization trials
    }
}
```

## Multilingual Support

The pipeline has been tested on **English (en), Russian (ru), French (fr), Chinese (zh), Arabic (ar), and Hindi (hi)**. Correct tokenization and NER behavior is guaranteed for these languages. Other languages are also supported but have not been explicitly validated.

### Model selection for non-default languages

Default models are only available for English (`bert-base-uncased`) and Russian (`ai-forever/ruBert-base`). For any other language you **must** set `model_name_or_path` in `config_overrides`:

```python
pipeline = TextClassificationTrainingPipeline(
    train_path="train.csv",
    config_overrides={
        "language": "fr",
        "model_name_or_path": "MODEL_NAME",
    }
)
```

Any HuggingFace checkpoint that supports your target language can be used.

### AncSetFit template

When the pipeline selects AncSetFit (2-5 examples per class), it prepends a `template` string to each `anc_label` to form anchor sentences. Default templates exist only for English and Russian. For other languages a custom `template` **must** be provided, otherwise the pipeline will raise an error. Even for English/Russian, setting a domain-specific template is recommended for best results:

```python
config_overrides={
    "language": "fr",
    "model_name_or_path": "camembert-base",
    "AncSetFitMethod": {
        "template": "User asks the bot to perform a request using the skill: ",  # write in your target language
    }
}
```

## Data Formats

### Text Classification (CSV)

```csv
text,label,anc_label
"Remove my meeting tomorrow",calendar_remove,remove calendar event
"Add a dentist appointment on Friday",calendar_set,add calendar event
```

The `anc_label` column is optional. It contains a natural language description of what the class means. It is a human-readable explanation of the label.

### NER (JSON)

The package supports two NER data formats:

**Offsets format** — entities are defined by character spans with `start` and `end` positions:

```json
[
  {"text": "What time is it in Australia", "spans": [{"start": 19, "end": 28, "label": "place_name"}]},
  {"text": "What is the forecast today for Moscow", "spans": [{"start": 21, "end": 26, "label": "date"}, {"start": 31, "end": 37, "label": "place_name"}]}
]
```

**Brackets format** — entities are marked inline using `[label : entity]` notation:

```json
[
  {"text": "play a track by [artist : the rolling stones]"},
  {"text": "play [song : hello] by [artist : adele]"}
]
```

## Example data

The files in `examples/test_data/noise_n_shot_data/` (text classification) and `examples/test_data/noise_n_shot_data_ner/` (NER) were made with external sampling scripts.

- **Text classification:** the scripts use the [**SNIPS**](https://huggingface.co/datasets/DeepPavlov/snips) dataset (intent/slot-style). They build train/test splits with optional n-shot sampling and label noise. In the included example, 1% of training labels were noised (randomly flipped to another class). The resulting CSVs follow the formats described above.
- **NER:** the scripts use the [**MASSIVE**](https://huggingface.co/datasets/DeepPavlov/massive) dataset. They produce few-shot train/test subsets with optional label noise (1% of labels noised) and export data in the offsets/BIO-style JSON expected by the NER pipeline.