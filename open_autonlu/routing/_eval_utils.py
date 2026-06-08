"""Cheap, dependency-light evaluation helpers shared by the profilers.

Pure sklearn/numpy -- no torch, no transformers. Used for separability proxies
in both DatasetProfile (TF-IDF) and CapabilityProfile (encoder embeddings).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def stratified_cv_macro_f1(
    X,
    y: Sequence,
    estimator,
    seed: int = 42,
    max_folds: int = 3,
) -> Optional[float]:
    """Stratified-CV macro-F1, robust to tiny/degenerate label sets.

    Returns None when CV is not meaningful (fewer than 2 classes, or any class
    with fewer than 2 samples). ``X`` may be an array (embeddings) or a list of
    strings (when ``estimator`` is a text Pipeline).
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    min_count = int(counts.min())
    if min_count < 2:
        return None
    folds = min(max_folds, min_count)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y, cv=skf, scoring="f1_macro")
    return float(np.mean(scores))


def knn_macro_f1(
    X: np.ndarray,
    y: Sequence,
    seed: int = 42,
    max_folds: int = 3,
    n_neighbors: int = 5,
) -> Optional[float]:
    """kNN separability proxy on dense embeddings (stratified CV macro-F1)."""
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    min_count = int(counts.min())
    # Leave room for the held-out fold: neighbors must be < smallest train class.
    k = max(1, min(n_neighbors, min_count - 1))
    clf = KNeighborsClassifier(n_neighbors=k)
    return stratified_cv_macro_f1(X, y, clf, seed=seed, max_folds=max_folds)


def bucket(value: Optional[float], low_hi: float, med_hi: float) -> str:
    """Map a continuous value to a relative label.

    ``value < low_hi`` -> "low"; ``< med_hi`` -> "medium"; else "high".
    Returns "unknown" for None.
    """
    if value is None:
        return "unknown"
    if value < low_hi:
        return "low"
    if value < med_hi:
        return "medium"
    return "high"
