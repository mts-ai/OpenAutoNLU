"""
This module was taken from https://github.com/MantisAI/nervaluate/releases/tag/0.2.0
The purpose of positioning it here - custom behavior injection.
We need to return a confusion matrix for NER on entity level.
"""

from .evaluate import (
    Evaluator,
    compute_actual_possible,
    compute_metrics,
    compute_precision_recall,
    compute_precision_recall_wrapper,
    find_overlap,
    summary_report_ent,
    summary_report_overall,
    summary_report_ents_indices,
    summary_report_overall_indices,
)
from .utils import collect_named_entities, conll_to_spans, list_to_spans, split_list

__all__ = [
    "Evaluator",
    "compute_actual_possible",
    "compute_metrics",
    "compute_precision_recall",
    "compute_precision_recall_wrapper",
    "find_overlap",
    "summary_report_ent",
    "summary_report_overall",
    "summary_report_ents_indices",
    "summary_report_overall_indices",
    "collect_named_entities",
    "conll_to_spans",
    "list_to_spans",
    "split_list",
]

__version__ = "0.2.0"
