import enum
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from torch.utils.data import Dataset


@dataclass
class Entity:
    """Represents a named entity extracted from text.

    Attributes:
        label: Entity type/class label (e.g., "PERSON", "ORG").
        text: The actual text span of the entity.
        start: Character offset where the entity starts in the source text.
        end: Character offset where the entity ends in the source text.
        score: Confidence score for the entity prediction.
    """

    label: str
    text: str
    start: int
    end: int
    score: float


class NERFormat(enum.Enum):
    """Format for Named Entity Recognition annotations.

    Attributes:
        OFFSETS: Character offset format (start, end positions).
        BIO: BIO tagging format (Beginning-Inside-Outside).
    """

    OFFSETS = "offsets"
    BIO = "bio"


class SaveFormat(enum.Enum):
    """Model serialization format.

    Attributes:
        TORCH: PyTorch native format (.pt/.pth files).
        ONNX: ONNX format.
    """

    TORCH = "torch"
    ONNX = "onnx"


class OodMethod(enum.Enum):
    """Out-of-Distribution detection method.

    Attributes:
        NONE: No OOD detection.
        LOGIT: Logit-based OOD (adds an out-of-scope class during training).
        AUTO: Automatically select method based on the training approach.
        MARGINAL_MAHALANOBIS_OOD: Mahalanobis distance-based detection.
        MSP_OOD: Maximum Softmax Probability - uses classifier confidence
            as OOD score.
    """

    NONE = "none"
    LOGIT = "logit"
    AUTO = "auto"
    MARGINAL_MAHALANOBIS_OOD = "MarginalMahalanobisTrainer"
    MSP_OOD = "MaxSoftmaxProbabilityTrainer"


@dataclass
class TokenClassifierOutput:
    """Output from a token classification (NER) model.

    Attributes:
        text: The original input text.
        labels: Extracted entities as Entity objects (OFFSETS format) or
            BIO tags as strings (BIO format).
    """

    text: str
    labels: Union[List[Entity], List[str]]


@dataclass
class TextLabel:
    """A single classification hypothesis with confidence score.

    Attributes:
        label: The predicted class label.
        score: Confidence score for this prediction.
    """

    label: str
    score: float


@dataclass
class TextClassifierOutput:
    """Output from a text classification model.

    Contains the input text and ranked list of classification hypotheses.

    Attributes:
        text: The original input text.
        hypotheses: List of TextLabel predictions, sorted by score descending.
    """

    text: str
    hypotheses: List[TextLabel]

    @property
    def most_probable(self) -> TextLabel:
        """Returns the highest-scoring classification hypothesis."""
        return self.hypotheses[0]


class SimplePyTorchDataset(Dataset):
    """Simple PyTorch Dataset wrapper for dictionary-based data.

    Wraps a dictionary of column arrays into a PyTorch Dataset interface.
    Each item returns a dictionary with values at the given index from
    each column.

    Attributes:
        data: Dictionary mapping column names to iterable values.

    Example:
        >>> data = {"text": ["hello", "world"], "label": [0, 1]}
        >>> dataset = SimplePyTorchDataset(data)
        >>> dataset[0]  # {"text": "hello", "label": 0}
    """

    def __init__(self, columns_dict: Dict[str, Iterable]):
        self.data = columns_dict

    def __getitem__(self, index) -> Dict[str, Any]:
        item = {}
        for key, value in self.data.items():
            item[key] = value[index]
        return item

    def __len__(self):
        first_key = list(self.data.keys())[0]
        return len(self.data[first_key])


@dataclass
class ClassificationEvaluationResult:
    """Evaluation metrics for a classification model.

    Attributes:
        confusion_matrix: NxN confusion matrix as numpy array.
        classification_report: Per-class metrics (precision, recall, f1, support).
        f1: Macro-averaged F1 score.
    """

    confusion_matrix: np.ndarray
    classification_report: Dict[str, Dict[str, float]]
    f1: float


@dataclass
class RetrievalEvaluationResult:
    recall_at_1: float


@dataclass
class TrainingArtifactInfo:
    """Information about a trained model artifact.

    Captures hyperparameters, labels, and evaluation metrics from training.
    Returned by training methods to provide metadata about the trained model.

    Attributes:
        hyperparameters: Dictionary of hyperparameters used for training.
        labels: List of class labels the model was trained on.
        dev_metrics: Evaluation results on development/validation set.
        test_metrics: Evaluation results on test set (all samples).
        test_metrics_inscope: Evaluation results on in-scope test samples only
            (excludes OOD samples).
    """

    hyperparameters: Dict[str, Any]
    labels: List[str]
    dev_metrics: Optional[ClassificationEvaluationResult] = None
    test_metrics: Optional[
        Union[ClassificationEvaluationResult, RetrievalEvaluationResult]
    ] = None
    test_metrics_inscope: Optional[ClassificationEvaluationResult] = None
