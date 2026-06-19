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


def few_shot_knn_macro_f1(
    X: np.ndarray,
    y: Sequence,
    *,
    n_shot: int = 5,
    seed: int = 42,
    max_folds: int = 3,
    n_neighbors: int = 1,
) -> Optional[float]:
    """Few-shot kNN: each train fold uses at most ``n_shot`` samples per class."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or int(counts.min()) < 2:
        return None
    min_count = int(counts.min())
    folds = min(max_folds, min_count)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    k = max(1, min(n_neighbors, n_shot))
    for train_idx, test_idx in skf.split(X, y):
        shot_global: list[int] = []
        y_train = y[train_idx]
        for cls in np.unique(y_train):
            cls_idx = train_idx[y_train == cls]
            if len(cls_idx) > n_shot:
                rng = np.random.RandomState(seed)
                cls_idx = rng.choice(cls_idx, size=n_shot, replace=False)
            shot_global.extend(cls_idx)
        shot_global = np.array(sorted(set(shot_global)))
        X_shot = X[shot_global]
        y_shot = y[shot_global]
        if len(np.unique(y_shot)) < 2:
            continue
        k_fit = max(1, min(k, len(y_shot) - 1))
        clf = KNeighborsClassifier(n_neighbors=k_fit)
        clf.fit(X_shot, y_shot)
        pred = clf.predict(X[test_idx])
        from sklearn.metrics import f1_score

        scores.append(float(f1_score(y[test_idx], pred, average="macro")))
    if not scores:
        return None
    return float(np.mean(scores))


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
