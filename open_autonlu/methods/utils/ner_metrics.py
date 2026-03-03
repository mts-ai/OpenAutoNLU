from typing import Any, Dict, List, Tuple
from ...nervaluate import Evaluator
import numpy as np


def evaluate_entity_level(
    true_labels: List[List[str]], pred_labels: List[List[str]], entity_names: List[str]
) -> Tuple[Dict[str, Any], np.ndarray]:
    evaluator = Evaluator(true_labels, pred_labels, tags=entity_names, loader="list")
    overall, entity_type, _, _, confusion_matrix = evaluator.evaluate()
    result = {}
    for name in entity_names:
        metrics = entity_type[name]["strict"]
        result[name] = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1-score": metrics["f1"],
            "support": metrics["possible"],
        }
    overal_metrics = overall["strict"]
    result["macro avg"] = {
        "precision": overal_metrics["precision"],
        "recall": overal_metrics["recall"],
        "f1-score": overal_metrics["f1"],
        "support": overal_metrics["possible"],
    }
    return result, confusion_matrix
