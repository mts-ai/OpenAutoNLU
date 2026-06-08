"""CapabilityProfile: cheap, model-aware probes on the user's encoder.

This is the model-aware replacement for sample-count heuristics. The same
dataset with a different ``model_name_or_path`` can yield a different
``separability_score`` -- which is exactly what makes routing portable across
encoders.

Only a *frozen* forward pass + sklearn heads are used here (no fine-tuning), so
this is far cheaper than ``data_quality.dynamic_finetuner``. The HF/torch import
is lazy (inside ``HFEmbedder``) so importing this module stays light.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.linear_model import LogisticRegression

from ._eval_utils import bucket, knn_macro_f1, stratified_cv_macro_f1

log = logging.getLogger(__name__)

DataLike = Union[Dataset, pd.DataFrame]


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into a dense ``(n, d)`` embedding matrix."""

    def encode(self, texts: List[str]) -> np.ndarray: ...


@dataclass
class CapabilityProfile:
    """Encoder-aware description of how learnable the data is."""

    model_id: Optional[str]
    probed: bool
    n_samples_used: int
    embedding_dim: Optional[int]
    separability_score: Optional[float]   # kNN stratified-CV macro-F1
    linear_head_ceiling: Optional[float]  # logistic-regression CV macro-F1
    separability_bucket: str              # low | medium | high | unknown
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HFEmbedder:
    """Frozen mean-pooled embeddings from any HuggingFace encoder."""

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 256,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        import torch  # lazy
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(device)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        out_chunks: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                last = out.last_hidden_state  # [B, T, H]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (last * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                mean = summed / denom
                out_chunks.append(mean.cpu().numpy())
        return np.vstack(out_chunks)


def _to_dataframe(data: DataLike, text_column: str, label_column: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, Dataset):
        df = data.to_pandas()
    else:
        raise TypeError(f"Unsupported data type: {type(data)!r}")
    out = df[[text_column, label_column]].copy()
    out.columns = ["text", "label"]
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(str)
    return out


def _subsample(df: pd.DataFrame, max_per_class: int, max_total: int, seed: int) -> pd.DataFrame:
    parts = [
        g.sample(n=min(len(g), max_per_class), random_state=seed)
        for _, g in df.groupby("label")
    ]
    work = pd.concat(parts, ignore_index=True)
    if len(work) > max_total:
        work = work.sample(n=max_total, random_state=seed)
    return work.reset_index(drop=True)


def extract_capability_profile(
    data: DataLike,
    model_config: Optional[Any] = None,
    budget: Optional[Any] = None,
    *,
    embedder: Optional[Embedder] = None,
    text_column: str = "text",
    label_column: str = "label",
    max_samples_per_class: int = 50,
    max_total: int = 2000,
    seed: int = 42,
) -> CapabilityProfile:
    """Probe the user's encoder for class separability and linear-head ceiling.

    Args:
        model_config: a :class:`ModelConfig` (or anything with ``.encoder`` /
            ``.max_seq_length``). Used to build the default :class:`HFEmbedder`.
        budget: optional :class:`BudgetPolicy`; ``skip_probes=True`` returns an
            unprobed profile (no encoder load).
        embedder: inject a custom :class:`Embedder` (tests / cached pipelines).
            When None, an :class:`HFEmbedder` is built from ``model_config``.
    """
    model_id = getattr(model_config, "encoder", None)

    if budget is not None and getattr(budget, "skip_probes", False):
        return CapabilityProfile(
            model_id=model_id,
            probed=False,
            n_samples_used=0,
            embedding_dim=None,
            separability_score=None,
            linear_head_ceiling=None,
            separability_bucket="unknown",
            notes={"reason": "budget.skip_probes"},
        )

    df = _to_dataframe(data, text_column, label_column)
    work = _subsample(df, max_samples_per_class, max_total, seed)

    if embedder is None:
        if not model_id:
            raise ValueError(
                "No embedder and no model_config.encoder set. Provide a base "
                "model (model_config.encoder) or inject an `embedder`."
            )
        embedder = HFEmbedder(
            model_id,
            max_seq_length=getattr(model_config, "max_seq_length", 256),
        )

    X = embedder.encode(work["text"].tolist())
    y = work["label"].tolist()
    embedding_dim = int(X.shape[1]) if X.ndim == 2 else None

    separability = knn_macro_f1(X, y, seed=seed)
    linear_ceiling = stratified_cv_macro_f1(
        X, y, LogisticRegression(max_iter=1000, random_state=seed), seed=seed
    )

    return CapabilityProfile(
        model_id=model_id,
        probed=True,
        n_samples_used=len(work),
        embedding_dim=embedding_dim,
        separability_score=separability,
        linear_head_ceiling=linear_ceiling,
        separability_bucket=bucket(separability, low_hi=0.5, med_hi=0.8),
        notes={"n_classes": int(work["label"].nunique())},
    )
