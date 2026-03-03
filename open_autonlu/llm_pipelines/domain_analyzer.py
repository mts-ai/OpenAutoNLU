from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

import pandas as pd

from .configs.llm_config import LlmClientConfig
from .model_endpoint import ModelEndpointWrapper
from .prompts import get_prompts

log = logging.getLogger(__name__)


def analyze_domain_and_labels(
    reference_data: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    examples_per_label: int = 5,
    config_overrides: Optional[Dict] = None,
    language: str = "en",
) -> Tuple[str, Dict[str, str]]:
    """
    Analyzes domain and labels using prompt analyzer.

    Args:
        reference_data: DataFrame with data for analysis
        text_column: Name of text column
        label_column: Name of label column
        examples_per_label: Number of examples per label for analysis
        config_overrides: Optional client configuration overrides

    Returns:
        Tuple[str, Dict[str, str]]: (domain description, dictionary of label descriptions)
    """
    prompts = get_prompts(language)

    examples_by_label = []
    for label in reference_data[label_column].unique():
        label_data = reference_data[reference_data[label_column] == label]
        num_examples = min(examples_per_label, len(label_data))
        if num_examples == 0:
            log.warning("Label '%s' has no examples. Skipping.", label)
            continue
        sample_texts = label_data.sample(n=num_examples)[text_column].tolist()

        examples_by_label.append(f"{prompts.label_prefix}: {label}")
        for text in sample_texts:
            examples_by_label.append(f"- {text}")
        examples_by_label.append("")

    examples_text = "\n".join(examples_by_label)

    labels_list = [str(label) for label in reference_data[label_column].unique()]
    label_names = ", ".join(labels_list)

    if config_overrides:
        llm_kwargs = config_overrides.get("LlmClientConfig", {})
        client_config = LlmClientConfig(**llm_kwargs)
    else:
        client_config = LlmClientConfig()

    client_config.default_system_prompt = prompts.analyzer_system_prompt
    client = ModelEndpointWrapper(config=client_config)

    log.info("Starting domain and labels analysis...")
    prompt_text = prompts.analyze_domain(
        examples_by_label=examples_text, label_names=label_names
    )

    try:
        response = client(prompt_text, system_prompt=prompts.analyzer_system_prompt)
        log.info("Received analyzer response (%s characters)", len(response))
    except Exception as exc:  # pylint: disable=broad-except
        log.error("Error calling analyzer: %s", exc, exc_info=True)
        return "", {}

    # Ensure labels are strings for regex matching
    labels = [str(label) for label in reference_data[label_column].unique().tolist()]
    domain_desc, label_descriptions = _parse_analysis_response(
        response=response,
        labels=labels,
        default_label_desc_template=prompts.default_label_desc_template,
        domain_description_header=prompts.domain_description_header,
        label_descriptions_header=prompts.label_descriptions_header,
    )
    log.info("Extracted domain description (%s characters)", len(domain_desc))
    log.info("Extracted label descriptions: %s", len(label_descriptions))

    # which labels were not found
    missing_labels = [label for label in labels if label not in label_descriptions]
    if missing_labels:
        log.warning("Could not extract descriptions for labels: %s", missing_labels)
        labels_header = get_prompts(language).label_descriptions_header
        if labels_header in response:
            labels_start = response.find(labels_header)
            labels_section = response[labels_start : labels_start + 3000]
            log.warning(
                "%s (first 3000 chars):\n%s",
                labels_header,
                labels_section,
            )
        else:
            log.warning("%s section not found in response", labels_header)

    return domain_desc, label_descriptions


def _parse_analysis_response(
    response: str,
    labels: list[str],
    default_label_desc_template: str = (
        "Texts of category '{label}' correspond to the general stylistic and "
        "thematic features of this text corpus, while having "
        "characteristic content related to the topic {label}."
    ),
    domain_description_header: str = "DOMAIN DESCRIPTION:",
    label_descriptions_header: str = "LABEL DESCRIPTIONS:",
) -> Tuple[str, Dict[str, str]]:
    domain_desc = ""
    label_descriptions: Dict[str, str] = {}

    if response:
        domain_pattern = (
            re.escape(domain_description_header)
            + r"(.*?)(?="
            + re.escape(label_descriptions_header)
            + r"|$)"
        )
        domain_match = re.search(domain_pattern, response, re.DOTALL)
        if domain_match:
            domain_desc = domain_match.group(1).strip()

        labels_pattern = re.escape(label_descriptions_header) + r"(.*?)$"
        labels_section = re.search(labels_pattern, response, re.DOTALL)
        if labels_section:
            labels_text = labels_section.group(1).strip()

            for label in labels:
                pattern = rf"{re.escape(label)}:\s*(.*?)(?=\n\n[a-zA-Z0-9_]+:|$)"
                label_match = re.search(pattern, labels_text, re.DOTALL)

                if label_match:
                    desc = label_match.group(1).strip()
                    if desc:
                        label_descriptions[label] = desc
                        log.info(
                            "Extracted description for label '%s': %s characters",
                            label,
                            len(desc),
                        )
                    else:
                        log.warning(
                            "WARNING: Description for label '%s' is empty",
                            label,
                        )
                else:
                    pattern2 = rf"{re.escape(label)}:(.*?)(?=\n[a-zA-Z0-9_]+:|$)"
                    label_match2 = re.search(pattern2, labels_text, re.DOTALL)
                    if label_match2:
                        desc = label_match2.group(1).strip()
                        if desc:
                            label_descriptions[label] = desc
                            log.info(
                                "Extracted description for label '%s' (alternative method): %s characters",
                                label,
                                len(desc),
                            )
                        else:
                            log.warning(
                                "WARNING: Description for label '%s' is empty (alternative method)",
                                label,
                            )
                    else:
                        log.warning(
                            "WARNING: Failed to extract description for label '%s'",
                            label,
                        )

        if not label_descriptions:
            log.error("Critical error: Failed to extract any label descriptions!")
            log.error("Full model response:\n%s", response)

            for label in labels:
                lines_list = response.split("\n")
                for i, line in enumerate(lines_list):
                    if line.strip().startswith(f"{label}:"):
                        desc_lines = []
                        j = i + 1
                        while j < len(lines_list) and not any(
                            lines_list[j].strip().startswith(f"{other_label}:")
                            for other_label in labels
                        ):
                            if lines_list[j].strip():
                                desc_lines.append(lines_list[j])
                            j += 1

                        if desc_lines:
                            label_descriptions[label] = "\n".join(desc_lines).strip()
                            log.info(
                                "Extracted description for label '%s' (line-by-line method): %s characters",
                                label,
                                len(label_descriptions[label]),
                            )

        log.info("Total extracted label descriptions: %s", len(label_descriptions))
        if len(label_descriptions) < len(labels):
            log.warning(
                "WARNING: Only extracted %s descriptions out of %s labels!",
                len(label_descriptions),
                len(labels),
            )

        for label in labels:
            if label not in label_descriptions or not label_descriptions[label]:
                default_desc = default_label_desc_template.format(label=label)
                label_descriptions[label] = default_desc
                log.warning("Created default description for label '%s'", label)

    return domain_desc, label_descriptions
