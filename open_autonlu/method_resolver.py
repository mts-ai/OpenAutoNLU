import logging
from collections import Counter
from typing import Any, Dict, Optional

import pandas as pd
from datasets import Dataset

from open_autonlu.llm_pipelines import DataGenerationPipeline, analyze_domain_and_labels
from open_autonlu.llm_pipelines.prompts import get_prompts

from .constants import (
    ANCSETFIT_NAME,
    FINETUNING_NAME,
    LLM_AUGMENTATION_THRESHOLD,
    MAX_ANC_SET_FIT_SIZE,
    MAX_SET_FIT_SIZE,
    MIN_ANC_SET_FIT_SIZE,
    SETFIT_NAME,
    TOKEN_CLF_FINETUNING,
    UPSAMPLE_AUGMENTEX,
    UPSAMPLE_LLM,
)
from .data import WordCharAugmentation

log = logging.getLogger(__name__)


def downsample(
    dataset: Dataset, label_counts: pd.DataFrame, downsample_to: int
) -> Dataset:
    """
    Downsamples a given dataset to the specified size for each class.

    Args:
        dataset (Dataset): The input dataset.
        label_counts (pd.DataFrame): A DataFrame containing the count of each label in the dataset.
        downsample_to (int): The target size for each class.

    Returns:
        Dataset: The downsampled dataset.
    """
    df: pd.DataFrame = dataset.to_pandas()  # type: ignore
    labels_to_downsample = (
        label_counts.where(lambda x: x > downsample_to, pd.NA).dropna().index.tolist()
    )
    sub_dfs = []
    for label_name, sub_df in df.groupby("label"):
        if label_name in labels_to_downsample:
            sub_dfs.append(sub_df.sample(downsample_to, random_state=42))
        else:
            sub_dfs.append(sub_df)
    df = pd.concat(sub_dfs, ignore_index=True)
    return Dataset.from_pandas(df, preserve_index=False)


def upsample(
    dataset: Dataset,
    label_counts: pd.DataFrame,
    upsample_to: int,
    upsample_method: str = UPSAMPLE_AUGMENTEX,
    llm_config_overrides: Optional[Dict[str, Any]] = None,
    domain_desc: Optional[str] = None,
    label_descriptions: Optional[Dict[str, str]] = None,
    language: str = "en",
) -> Dataset:
    """
    Upsamples a given dataset to the specified size for each class using the specified method.

    Args:
        dataset (Dataset): The input dataset.
        label_counts (pd.DataFrame): A DataFrame containing the count of each label in the dataset.
        upsample_to (int): The target size for each class.
        upsample_method (str, optional): The method to use for upsampling. Defaults to UPSAMPLE_AUGMENTEX.
        llm_config_overrides: Optional config overrides for LLM augmentation.
        domain_desc: Optional domain description for LLM augmentation.
        label_descriptions: Optional label descriptions for LLM augmentation.

    Returns:
        Dataset: The upsampled dataset.
    """
    df: pd.DataFrame = dataset.to_pandas()  # type: ignore
    low_resource_classes = [
        str(label)  # Ensure labels are strings
        for label in label_counts.where(lambda x: x < upsample_to, pd.NA)
        .dropna()
        .index.tolist()
    ]
    # Ensure dataset labels are strings for consistent comparison
    if df["label"].dtype != "object":
        df = df.copy()
        df["label"] = df["label"].astype(str)
    if upsample_method == UPSAMPLE_AUGMENTEX:
        df = _augment_with_augmentex(df, low_resource_classes, upsample_to)
    elif upsample_method == UPSAMPLE_LLM:
        df = _augment_with_llm(
            dataset_df=df,
            label_counts=label_counts,
            upsample_to=upsample_to,
            llm_config_overrides=llm_config_overrides or {},
            domain_desc=domain_desc,
            label_descriptions=label_descriptions,
            language=language,
        )

    return Dataset.from_pandas(df, preserve_index=False)


def _augment_with_augmentex(
    df: pd.DataFrame, low_resource_classes: list[str], upsample_to: int
) -> pd.DataFrame:
    augmentation = WordCharAugmentation()
    sub_dfs = []
    for label_name, sub_df in df.groupby("label"):
        # Ensure label_name is string for comparison
        label_name_str = str(label_name)
        if label_name_str in low_resource_classes:
            augmented_pool = []
            for _, row in sub_df.iterrows():
                augmented_texts = augmentation(row["text"])
                for text in augmented_texts:
                    clone = row.copy()
                    clone["text"] = text
                    augmented_pool.append(clone)
            num_samples_to_aug = upsample_to - sub_df.shape[0]
            augmented_df = pd.DataFrame(augmented_pool).drop_duplicates(subset=["text"])
            if num_samples_to_aug > augmented_df.shape[0]:
                if augmented_df.shape[0] > 0:
                    sub_dfs.append(pd.concat([sub_df, augmented_df]))
                else:
                    sub_dfs.append(sub_df)
            else:
                augmented_df = augmented_df.sample(
                    n=num_samples_to_aug, random_state=42
                )
                sub_dfs.append(pd.concat([sub_df, augmented_df]))
        else:
            sub_dfs.append(sub_df)
    return pd.concat(sub_dfs, ignore_index=True)


def _augment_with_llm(
    dataset_df: pd.DataFrame,
    label_counts: pd.DataFrame,
    upsample_to: int,
    llm_config_overrides: Dict[str, Any],
    domain_desc: Optional[str],
    label_descriptions: Optional[Dict[str, str]],
    language: str = "en",
) -> pd.DataFrame:
    threshold = llm_config_overrides.get("threshold", LLM_AUGMENTATION_THRESHOLD)
    classes_to_augment = [
        str(label)
        for label, stats in label_counts.iterrows()
        if stats["text"] < threshold and str(label).lower() != "outofscope"
    ]

    if not classes_to_augment:
        log.info("No classes requiring LLM augmentation")
        return dataset_df

    if dataset_df["label"].dtype != "object":
        dataset_df = dataset_df.copy()
        dataset_df["label"] = dataset_df["label"].astype(str)

    domain_desc, label_descriptions = _ensure_domain_context(
        dataset_df,
        domain_desc,
        label_descriptions,
        llm_config_overrides.get("use_domain_analysis", False),
        llm_config_overrides.get("config_overrides"),
        language=language,
    )

    prompt = llm_config_overrides.get("prompt") or get_prompts(language).generate_texts
    llm_pipeline_overrides = llm_config_overrides.get("config_overrides")

    llm_classes_data = {}
    fallback_rows = []

    for label in classes_to_augment:
        sub_df = dataset_df[dataset_df["label"] == label]
        current_size = sub_df.shape[0]
        num_needed = max(0, upsample_to - current_size)
        if num_needed <= 0:
            continue

        if current_size < 2:
            log.warning(
                "Label '%s' has less than 2 examples. Using character-level augmentation.",
                label,
            )
            fallback_df = _augment_with_augmentex(
                sub_df, low_resource_classes=[label], upsample_to=upsample_to
            )
            new_rows = fallback_df.iloc[sub_df.shape[0] :]
            if not new_rows.empty:
                fallback_rows.append(new_rows)
            continue

        llm_classes_data[label] = {
            "reference_data": sub_df,
            "current_size": current_size,
            "num_needed": num_needed,
        }

    augmented_rows = []

    if llm_classes_data:
        all_reference_data = pd.concat(
            [data["reference_data"] for data in llm_classes_data.values()],
            ignore_index=True,
        )

        pipeline = DataGenerationPipeline(
            reference_data=all_reference_data,
            text_column="text",
            label_column="label",
            prompt=prompt,
            config_overrides=llm_pipeline_overrides,
            domain_desc=domain_desc,
            label_descriptions=label_descriptions,
            language=language,
        )

        all_generated_texts_per_class = {label: [] for label in llm_classes_data.keys()}
        max_attempts = llm_config_overrides.get("max_attempts", 10)
        attempt = 0

        log.info("Generating data for %d classes in parallel", len(llm_classes_data))

        while attempt < max_attempts:
            classes_still_needing = {
                label: data
                for label, data in llm_classes_data.items()
                if len(all_generated_texts_per_class[label]) < data["num_needed"]
            }

            if not classes_still_needing:
                break

            max_num_needed = max(
                data["num_needed"] - len(all_generated_texts_per_class[label])
                for label, data in classes_still_needing.items()
            )
            min_request_size = llm_config_overrides.get("min_request_size", 20)
            request_size = max(int(max_num_needed * 1.2) + 1, int(min_request_size))
            num_shot = min(
                llm_config_overrides.get("num_shot", 5),
                min(data["current_size"] for data in classes_still_needing.values()),
            )

            generation_kwargs = {
                "num_generations_per_class": request_size,
                "num_shot": num_shot,
            }

            try:
                generated_dataset = pipeline.generate(generation_kwargs)

                if generated_dataset is None:
                    log.warning("Failed to generate data on attempt %d", attempt + 1)
                    attempt += 1
                    continue

                generated_df = generated_dataset.to_pandas()
                if generated_df.empty:
                    attempt += 1
                    continue

                for label, data in classes_still_needing.items():
                    label_generated = generated_df[generated_df["label"] == label]
                    num_needed = data["num_needed"]
                    current_generated = all_generated_texts_per_class[label]

                    if not label_generated.empty:
                        new_texts = label_generated["text"].tolist()
                        for text in new_texts:
                            if text not in current_generated:
                                current_generated.append(text)
                                if len(current_generated) >= num_needed:
                                    break

                attempt += 1

            except Exception as e:
                log.warning("Error during generation on attempt %d: %s", attempt + 1, e)
                attempt += 1
                continue

        for label, data in llm_classes_data.items():
            generated_texts = all_generated_texts_per_class[label]
            num_needed = data["num_needed"]

            if len(generated_texts) < num_needed:
                log.warning(
                    "Generated only %d/%d examples for label '%s' after %d attempts",
                    len(generated_texts),
                    num_needed,
                    label,
                    attempt,
                )

            if generated_texts:
                generated_df_class = pd.DataFrame(
                    {"text": generated_texts[:num_needed], "label": label}
                )
                augmented_rows.append(generated_df_class)

    # Combine LLM-generated and fallback-generated rows
    all_augmented_rows = augmented_rows + fallback_rows

    if all_augmented_rows:
        dataset_df = pd.concat([dataset_df] + all_augmented_rows, ignore_index=True)
    return dataset_df


def _ensure_domain_context(
    dataset_df: pd.DataFrame,
    domain_desc: Optional[str],
    label_descriptions: Optional[Dict[str, str]],
    use_domain_analysis: bool,
    config_overrides: Optional[Dict[str, Any]],
    language: str = "en",
) -> tuple[Optional[str], Optional[Dict[str, str]]]:
    if domain_desc is not None and label_descriptions is not None:
        return domain_desc, label_descriptions

    if not use_domain_analysis:
        return domain_desc, label_descriptions

    try:
        label_counts = dataset_df.groupby("label").size()
        min_label_count = label_counts.min() if len(label_counts) > 0 else 0
        examples_per_label = min(7, min_label_count) if min_label_count > 0 else 5

        domain_desc, label_descriptions = analyze_domain_and_labels(
            reference_data=dataset_df,
            text_column="text",
            label_column="label",
            examples_per_label=examples_per_label,
            config_overrides=config_overrides,
            language=language,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Failed to perform domain analysis: %s", exc, exc_info=True)
    return domain_desc, label_descriptions


def count_labels(dataset: Dataset) -> tuple[pd.DataFrame, int]:
    """
    Counts the occurrence of each label in a given dataset.

    Args:
        dataset (Dataset): The input dataset.

    Returns:
        tuple[pd.DataFrame, int]: A DataFrame containing the count of each label and the minimum class size.
    """
    df: pd.DataFrame = dataset.to_pandas()  # type: ignore
    label_counts = df.groupby("label").agg("count")
    min_class_size = label_counts.agg("min")["text"]
    return label_counts, min_class_size


def validate_min_size(label_counts: pd.DataFrame, min_class_size: int):
    """
    Validates the minimum class size.

    Args:
        label_counts (pd.DataFrame): A DataFrame containing the count of each label.
        min_class_size (int): The minimum class size.

    Raises:
        ValueError: If the minimum class size is less than MIN_ANC_SET_FIT_SIZE for any class.
    """
    if min_class_size < MIN_ANC_SET_FIT_SIZE:
        not_sufficient_list = (
            label_counts.where(lambda x: x < MIN_ANC_SET_FIT_SIZE, pd.NA)
            .dropna()
            .index.tolist()
        )
        raise ValueError(
            f"There are not enough data for labels [{','.join(map(str, not_sufficient_list))}]"
        )


def resolve_method(min_class_size: int) -> str:
    """
    Resolves the method to use based on the minimum class size.

    Args:
        min_class_size (int): The minimum class size.

    Returns:
        str: The resolved method.
    """
    if min_class_size > MAX_SET_FIT_SIZE:
        method = FINETUNING_NAME
    elif min_class_size > MAX_ANC_SET_FIT_SIZE and min_class_size <= MAX_SET_FIT_SIZE:
        method = SETFIT_NAME
    elif (
        min_class_size >= MIN_ANC_SET_FIT_SIZE
        and min_class_size <= MAX_ANC_SET_FIT_SIZE
    ):
        method = ANCSETFIT_NAME
    else:
        raise ValueError(
            f"At least 2 samples per label required for pipeline. Found: {min_class_size} sample"
        )
    return method


def count_labels_ner(dataset: Dataset) -> tuple[pd.DataFrame, int]:  # TODO: IMPLEMENT
    """
    Counts the occurrence of each label in a given dataset for NER.

    Args:
        dataset (Dataset): The input dataset.

    Returns:
        tuple[pd.DataFrame, int]: A DataFrame containing the count of each label and the minimum class size.
    """
    main_counter = Counter()
    for record in dataset:
        entities = [
            label[2:]
            for label in record["labels"]
            if label != "O" and label[:2] == "B-"
        ]
        main_counter.update(entities)
    label_counts = pd.Series(
        list(main_counter.values()), index=list(main_counter.keys())
    )
    min_class_size = label_counts.min()
    return label_counts, min_class_size


def resolve_ner_method(max_class_size: int) -> str:
    """
    Resolves the method to use for NER based on the maximum class size.

    Args:
        max_class_size (int): The maximum class size.

    Returns:
        str: The resolved method.
    """

    return TOKEN_CLF_FINETUNING
