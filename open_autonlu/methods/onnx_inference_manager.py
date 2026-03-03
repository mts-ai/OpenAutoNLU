import json
import os
from abc import ABC, abstractmethod
from bisect import bisect_left
from typing import List

import numpy as np
import onnxruntime
import torch
from accelerate import find_executable_batch_size
from transformers import AutoTokenizer

from ..methods.constants import BATCH_SIZE
from ..methods.data_types import (
    TextClassifierOutput,
    TextLabel,
    TokenClassifierOutput,
)
from ..methods.method import SequenceClassificationMethod
from ..methods.utils import batch as batcher

_PADDING_BUCKETS: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256, 512)


def _compute_bucket_pad_length(seq_length: int, model_max_length: int) -> int:
    index = bisect_left(_PADDING_BUCKETS, seq_length)
    if index < len(_PADDING_BUCKETS):
        return min(_PADDING_BUCKETS[index], model_max_length)
    return model_max_length


def _pad_inputs_to_bucket(
    inputs: dict[str, np.ndarray],
    model_max_length: int,
    pad_token_id: int,
) -> dict[str, np.ndarray]:
    seq_length = inputs["input_ids"].shape[1]
    bucket_length = _compute_bucket_pad_length(seq_length, model_max_length)
    if seq_length >= bucket_length:
        return inputs
    pad_width = bucket_length - seq_length
    padded: dict[str, np.ndarray] = {}
    for key, value in inputs.items():
        pad_value = pad_token_id if key == "input_ids" else 0
        padded[key] = np.pad(value, ((0, 0), (0, pad_width)), constant_values=pad_value)
    return padded


class AbstractONNXInferenceManager(ABC):
    """Abstract base class for ONNX model inference.

    Provides common infrastructure for loading and running ONNX models
    with automatic hardware acceleration detection. Subclasses implement
    task-specific prediction logic.

    Attributes:
        ort_provider: ONNX Runtime execution provider (CPU, CUDA, or CoreML).
    """

    @abstractmethod
    def __init__(self, path_to_package: str) -> None:
        """Initialize the inference manager.

        Automatically detects available hardware accelerators and selects
        the appropriate ONNX Runtime execution provider.

        Args:
            path_to_package: Path to directory containing model.onnx and
                associated files (tokenizer, label mapping).
        """
        self.ort_provider = "CPUExecutionProvider"
        if torch.cuda.is_available():
            self.ort_provider = "CUDAExecutionProvider"
        elif torch.backends.mps.is_available():
            self.ort_provider = "CoreMLExecutionProvider"

    @abstractmethod
    def predict(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        auto_find_batch_size: bool = False,
    ) -> list[TextClassifierOutput] | list[TokenClassifierOutput]:
        """Run inference on input texts.

        Args:
            texts: List of input texts to classify.
            batch_size: Batch size for inference.
            auto_find_batch_size: If True, automatically find the largest
                batch size that fits in memory.

        Returns:
            List of classification outputs.
        """
        ...


class GeneralSequenceClassifierInferenceManager(AbstractONNXInferenceManager):
    """ONNX inference manager for sequence classification models.

    Loads an ONNX-exported text classification model and provides batched
    inference with optional automatic batch size detection. Supports returning
    either the top prediction or all class hypotheses with scores.

    The model package directory must contain:
        - model.onnx: The exported ONNX model
        - tokenizer files: HuggingFace tokenizer configuration
        - label_mapping.json: Mapping of label IDs to label names

    Attributes:
        tokenizer: HuggingFace tokenizer for input preprocessing.
        session: ONNX Runtime inference session.
        id2label: Mapping from numeric IDs to label strings.

    Example:
        >>> manager = GeneralSequenceClassifierInferenceManager("./model_package")
        >>> outputs = manager.predict(["Hello world", "Test text"])
        >>> print(outputs[0].most_probable.label)
    """

    def __init__(self, path_to_package: str) -> None:
        super().__init__(path_to_package)
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_package)
        self.session = onnxruntime.InferenceSession(
            os.path.join(path_to_package, "model.onnx"),
            providers=[self.ort_provider, "CPUExecutionProvider"],
        )
        with open(os.path.join(path_to_package, "label_mapping.json"), "r") as f:
            config = json.load(f)
        self.id2label = {int(key): value for key, value in config["id2label"].items()}

    def predict(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        auto_find_batch_size: bool = False,
        return_all_hypotheses: bool = False,
    ) -> list[TextClassifierOutput]:
        """Run classification inference on input texts.

        Args:
            texts: List of input texts to classify.
            batch_size: Batch size for inference.
            auto_find_batch_size: If True, automatically find the largest
                batch size that fits in memory.
            return_all_hypotheses: If True, return scores for all classes.
                If False, return only the top prediction.

        Returns:
            List of TextClassifierOutput with predictions for each input.
        """
        if auto_find_batch_size:
            logits = find_executable_batch_size(
                self._get_logits, starting_batch_size=batch_size
            )(texts)  # type: ignore
        else:
            logits = self._get_logits(batch_size, texts)
        if return_all_hypotheses:
            hypotheses_batch = [[] for _ in range(logits.shape[0])]
            for text_id in range(logits.shape[0]):
                for label_id in range(logits.shape[1]):
                    hypotheses_batch[text_id].append(
                        {
                            "label": self.id2label[label_id],
                            "score": float(logits[text_id, label_id]),
                        }
                    )
            hypotheses_batch = [
                list(sorted(hypotheses, key=lambda x: x["score"], reverse=True))
                for hypotheses in hypotheses_batch
            ]
            packaged = []
            for text, output in zip(texts, hypotheses_batch):
                label_hypotheses = [
                    TextLabel(dict_label["label"], dict_label["score"])
                    for dict_label in output
                ]
                packaged.append(
                    TextClassifierOutput(text=text, hypotheses=label_hypotheses)
                )
            return packaged

        all_scores = SequenceClassificationMethod._extract_scores(logits)
        all_labels = SequenceClassificationMethod._extract_labels(logits, self.id2label)
        return [
            TextClassifierOutput(text=text, hypotheses=[TextLabel(label, score)])
            for label, score, text in zip(all_labels, all_scores, texts)
        ]

    def _get_logits(
        self,
        batch_size: int,
        texts: List[str],
    ) -> np.ndarray:
        total = len(texts)
        if total == 0:
            return np.empty((0, 0), dtype=np.float32)

        logits_buffer = None
        offset = 0
        for text_batch in batcher(texts, batch_size):
            inputs = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="np",
            ).data
            inputs = _pad_inputs_to_bucket(
                inputs, self.tokenizer.model_max_length, self.tokenizer.pad_token_id
            )
            logits = self.session.run(["logits"], inputs)[0]
            if logits_buffer is None:
                logits_buffer = np.empty((total, logits.shape[1]), dtype=logits.dtype)
            batch_size_actual = logits.shape[0]
            logits_buffer[offset : offset + batch_size_actual] = logits
            offset += batch_size_actual

        return logits_buffer if logits_buffer is not None else np.empty((0, 0))
