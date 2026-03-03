from typing import Any, Dict, Optional
import torch
from transformers import AutoTokenizer
from collections import defaultdict
import datasets


def batch(iterable, size=1):
    iterable_length = len(iterable)
    for idx in range(0, iterable_length, size):
        yield iterable[idx : min(idx + size, iterable_length)]


def initialize_transformers_tokenizer(
    model_name_or_path: str,
    max_seq_length: int,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
):
    default_tokenizer_kwargs = {"model_max_length": max_seq_length}
    if not tokenizer_kwargs:
        tokenizer_kwargs = default_tokenizer_kwargs
    else:
        default_tokenizer_kwargs.update(tokenizer_kwargs)
        tokenizer_kwargs = default_tokenizer_kwargs
    return AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, **tokenizer_kwargs
    )


def resolve_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def sample_ner_dataset(dataset: datasets.Dataset, n_shot: int) -> datasets.Dataset:
    """
    This function should take in a datasets.Dataset with columns
    text (a string with text) and labels (a list of BIO labels)
    and return a datasets.Dataset with same structure but each
    entity type should occur approximately n_shot times. All
    empty (all BIO labels are O) samples should be excluded.
    """

    # Filter out examples with all 'O' labels
    def has_entities(example):
        return any(label != "O" for label in example["labels"])

    filtered_dataset = dataset.filter(has_entities)

    # If no examples left after filtering, return empty dataset
    if len(filtered_dataset) == 0:
        return datasets.Dataset.from_dict({"text": [], "labels": []})

    # Compute entity counts for each example
    def compute_entity_counts(example):
        labels = example["labels"]
        entity_counts = defaultdict(int)
        current_entity = None
        for label in labels:
            if label.startswith("B-"):
                entity_type = label[2:]
                entity_counts[entity_type] += 1
                current_entity = entity_type
            elif label.startswith("I-"):
                entity_type = label[2:]
                if current_entity == entity_type:
                    continue  # Part of the same entity
                else:
                    current_entity = None  # Invalid I- tag, treat as O
            else:
                current_entity = None
        # Convert to regular dict for serialization
        example["entity_counts"] = dict(entity_counts)
        return example

    def normalize_dicts(examples):
        new_data = {"entity_counts": []}
        for counts in examples["entity_counts"]:
            new_data["entity_counts"].append(
                {
                    key: value if value is not None else 0
                    for key, value in counts.items()
                }
            )
        return new_data

    filtered_dataset = filtered_dataset.map(compute_entity_counts).map(
        normalize_dicts, batched=True
    )
    # return filtered_dataset

    # Collect all entity types present in the dataset
    entity_types = list(filtered_dataset[0]["entity_counts"].keys())
    # for example in filtered_dataset:
    #     entity_types.update(example['entity_counts'].keys())
    # entity_types = list(entity_types)

    if not entity_types:
        return datasets.Dataset.from_dict({"text": [], "labels": []})

    # Shuffle entity types to randomize processing order
    # random.shuffle(entity_types)
    print(entity_types)

    # Initialize remaining counts for each entity type
    remaining_counts = {et: n_shot for et in entity_types}
    selected_indices = set()

    # Process each entity type to collect examples
    for entity_type in entity_types:
        # Collect candidate examples that have the current entity type and are not selected yet
        candidates = []
        for idx in range(len(filtered_dataset)):
            if idx not in selected_indices:
                example = filtered_dataset[idx]
                if entity_type in example["entity_counts"]:
                    count = example["entity_counts"][entity_type]
                    candidates.append((idx, count))

        # Sort candidates by count descending to maximize contribution per example
        try:
            candidates.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(candidates)
            raise (e)

        for idx, count in candidates:
            if remaining_counts[entity_type] <= 0:
                break
            if idx not in selected_indices:
                selected_indices.add(idx)
                # Update remaining counts for all entity types in this example
                example = filtered_dataset[idx]
                for et, cnt in example["entity_counts"].items():
                    if et in remaining_counts:
                        remaining_counts[et] = max(0, remaining_counts[et] - cnt)

    # Create the sampled dataset
    selected_indices = sorted(selected_indices)
    sampled_dataset = filtered_dataset.select(selected_indices)

    # Remove the temporary 'entity_counts' column
    sampled_dataset = sampled_dataset.remove_columns("entity_counts")

    return sampled_dataset
