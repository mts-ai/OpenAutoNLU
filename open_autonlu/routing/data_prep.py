"""Recipe-driven training data preparation.

Data caps and upsampling are declared per recipe (not chosen by sample-count
heuristics in the pipeline). Applied after routing selects a recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from datasets import Dataset

from ..constants import MAX_ANC_SET_FIT_SIZE, MAX_SET_FIT_SIZE, UPSAMPLE_AUGMENTEX
from ..method_resolver import count_labels, downsample, upsample


@dataclass
class DataPrepConfig:
    """Declarative data transforms for a recipe."""

    downsample_to: Optional[int] = None
    upsample_to: Optional[int] = None
    upsample_method: str = UPSAMPLE_AUGMENTEX

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DataPrepConfig":
        if not data:
            return cls()
        return cls(
            downsample_to=data.get("downsample_to"),
            upsample_to=data.get("upsample_to"),
            upsample_method=data.get("upsample_method", UPSAMPLE_AUGMENTEX),
        )


_DEFAULT_DATA_PREP_BY_TRAINER = {
    "AncSetFitMethod": DataPrepConfig(downsample_to=MAX_ANC_SET_FIT_SIZE),
    "AncSetFitOOD": DataPrepConfig(downsample_to=MAX_ANC_SET_FIT_SIZE),
    "SetFitMethod": DataPrepConfig(downsample_to=MAX_SET_FIT_SIZE),
    "SetFitOOD": DataPrepConfig(downsample_to=MAX_SET_FIT_SIZE),
    "Finetuner": DataPrepConfig(),
    "FinetunerWithOOD": DataPrepConfig(),
}


def data_prep_for_trainer(trainer: str) -> DataPrepConfig:
    return _DEFAULT_DATA_PREP_BY_TRAINER.get(trainer, DataPrepConfig())


def apply_data_prep(
    training_data: Dataset,
    recipe: Any,
    *,
    upsample_method: Optional[str] = None,
    llm_config_overrides: Optional[dict] = None,
    domain_desc: Optional[str] = None,
    label_descriptions: Optional[dict] = None,
    language: str = "en",
) -> Dataset:
    """Apply the recipe's declared data preparation to training data."""
    prep = getattr(recipe, "data_prep", None)
    if prep is None or not isinstance(prep, DataPrepConfig):
        prep = data_prep_for_trainer(getattr(recipe, "trainer", ""))

    data = training_data
    label_counts, _ = count_labels(data)

    if prep.upsample_to is not None:
        method = upsample_method or prep.upsample_method
        data = upsample(
            data,
            label_counts,
            prep.upsample_to,
            upsample_method=method,
            llm_config_overrides=llm_config_overrides,
            domain_desc=domain_desc,
            label_descriptions=label_descriptions,
            language=language,
        )
        label_counts, _ = count_labels(data)

    if prep.downsample_to is not None:
        data = downsample(data, label_counts, prep.downsample_to)

    return data
