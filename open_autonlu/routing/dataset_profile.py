"""DatasetProfile: cheap, data-only signals (no training, no encoder).

Extracts relative, model-neutral features used for recipe soft-matching and as
input to the capability probes. Deterministic given a fixed seed; no
language-specific imports.

Design principle: signals are *relative* ("high imbalance", "low separability"),
not absolute routing thresholds. The router must not branch on raw sample counts
alone -- it combines these signals with the CapabilityProfile.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..methods.constants import OOS_LABEL
from ._eval_utils import bucket, stratified_cv_macro_f1

log = logging.getLogger(__name__)

DataLike = Union[Dataset, pd.DataFrame]


@dataclass
class DatasetProfile:
    """Data-only description of a classification dataset."""

    n_samples: int
    n_classes: int
    class_counts: Dict[str, int]
    min_class_size: int
    median_class_size: float
    max_class_size: int
    imbalance_ratio: float            # max_class / min_class
    label_entropy: float              # normalized to [0, 1]
    text_len_chars: Dict[str, float]  # min/median/mean/max/p95
    duplicate_rate: float
    has_oos_label: bool
    has_anc_label: bool
    has_hierarchy: bool
    tfidf_separability: Optional[float]  # stratified-CV macro-F1, or None
    # Relative buckets (for soft recipe matching; NOT router thresholds):
    size_bucket: str                  # scarce | moderate | ample
    imbalance_bucket: str             # low | medium | high
    separability_bucket: str          # low | medium | high | unknown

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_dataframe(data: DataLike, text_column: str, label_column: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, Dataset):
        df = data.to_pandas()
    else:
        raise TypeError(f"Unsupported data type: {type(data)!r}")
    for col in (text_column, label_column):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Have: {list(df.columns)}")
    out = df[[text_column, label_column]].copy()
    out.columns = ["text", "label"]
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(str)
    return out


def _normalized_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of the class distribution, normalized to [0, 1]."""
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(counts)))


def extract_dataset_profile(
    data: DataLike,
    task_spec: Optional[Any] = None,
    *,
    text_column: str = "text",
    label_column: str = "label",
    seed: int = 42,
    max_samples_for_separability: int = 2000,
) -> DatasetProfile:
    """Compute a :class:`DatasetProfile` from raw labeled text.

    Args:
        data: a ``datasets.Dataset`` or ``pandas.DataFrame``.
        task_spec: optional :class:`TaskSpec`; ``label_schema='hierarchical'``
            sets ``has_hierarchy``.
        seed: determinism for the TF-IDF separability CV and subsampling.
        max_samples_for_separability: cap for the separability probe (speed).
    """
    df = _to_dataframe(data, text_column, label_column)
    n_samples = len(df)

    counts_series = df.groupby("label").size().sort_values(ascending=False)
    counts = counts_series.to_numpy()
    n_classes = int(len(counts_series))
    min_class_size = int(counts.min())
    max_class_size = int(counts.max())
    median_class_size = float(np.median(counts))
    imbalance_ratio = float(max_class_size / max(min_class_size, 1))
    label_entropy = _normalized_entropy(counts)

    lengths = df["text"].str.len().to_numpy()
    text_len_chars = {
        "min": float(lengths.min()),
        "median": float(np.median(lengths)),
        "mean": float(lengths.mean()),
        "max": float(lengths.max()),
        "p95": float(np.percentile(lengths, 95)),
    }

    duplicate_rate = float(df["text"].duplicated().sum() / max(n_samples, 1))

    orig_cols = (
        set(data.columns)
        if isinstance(data, pd.DataFrame)
        else set(data.column_names)
    )
    has_anc_label = "anc_label" in orig_cols
    has_oos_label = bool(
        (df["label"].str.lower() == OOS_LABEL.lower()).any()
    )
    label_schema = getattr(task_spec, "label_schema", "flat") if task_spec else "flat"
    has_hierarchy = (
        has_anc_label
        or label_schema == "hierarchical"
        or bool({"scenario", "domain", "parent_label"} & orig_cols)
    )

    tfidf_separability = _tfidf_separability(
        df, seed=seed, max_samples=max_samples_for_separability
    )

    return DatasetProfile(
        n_samples=n_samples,
        n_classes=n_classes,
        class_counts={str(k): int(v) for k, v in counts_series.items()},
        min_class_size=min_class_size,
        median_class_size=median_class_size,
        max_class_size=max_class_size,
        imbalance_ratio=imbalance_ratio,
        label_entropy=label_entropy,
        text_len_chars=text_len_chars,
        duplicate_rate=duplicate_rate,
        has_oos_label=has_oos_label,
        has_anc_label=has_anc_label,
        has_hierarchy=has_hierarchy,
        tfidf_separability=tfidf_separability,
        size_bucket=_size_bucket(min_class_size),
        imbalance_bucket=bucket(imbalance_ratio, low_hi=2.0, med_hi=10.0),
        separability_bucket=bucket(tfidf_separability, low_hi=0.5, med_hi=0.8),
    )


def _size_bucket(min_class_size: int) -> str:
    """Coarse, descriptive size label (NOT the router's regime thresholds)."""
    if min_class_size < 10:
        return "scarce"
    if min_class_size < 100:
        return "moderate"
    return "ample"


def _tfidf_separability(
    df: pd.DataFrame, seed: int, max_samples: int
) -> Optional[float]:
    """TF-IDF + logistic-regression stratified-CV macro-F1 (no encoder needed)."""
    if df["label"].nunique() < 2:
        return None
    work = df
    if len(df) > max_samples:
        parts = []
        for _, g in df.groupby("label"):
            n = max(1, int(round(max_samples * len(g) / len(df))))
            parts.append(g.sample(n=min(n, len(g)), random_state=seed))
        work = pd.concat(parts, ignore_index=True)
    estimator = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    try:
        return stratified_cv_macro_f1(
            work["text"].tolist(), work["label"].tolist(), estimator, seed=seed
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("TF-IDF separability probe failed: %s", exc)
        return None
