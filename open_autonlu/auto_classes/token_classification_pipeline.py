import logging
from typing import List, Optional, Type

from datasets import Dataset

from .. import methods
from .abstract_pipeline import (
    AbstractInferencePipeline,
    AbstractTrainingPipeline,
    get_method_from_string,
)
from ..constants import OOD_METHOD_MAP, TRAIN_KEY
from ..data.ner_data_provider import SimpleNerDataProvider
from ..data_quality import LabelAggregation, Retag, Uncertainty
from ..data_quality.data_types import DatasetEvaluatorOutput
from ..data_quality.dynamic_finetuner import TokenClassDynamicTuner
from ..methods.data_types import TokenClassifierOutput
from ..method_resolver import (
    count_labels_ner,
    resolve_ner_method,
)

log = logging.getLogger(__name__)


class TokenClassificationTrainingPipeline(AbstractTrainingPipeline):
    """Training pipeline for token classification (NER) tasks.

    Supports standard token classification finetuning approach.

    Example:
        >>> from open_autonlu.auto_classes import TokenClassificationTrainingPipeline
        >>> from open_autonlu.methods.data_types import SaveFormat
        >>>
        >>> pipeline = TokenClassificationTrainingPipeline(
        ...     train_path="train.json",
        ...     test_path="test.json"
        ... )
        >>> result = pipeline.train()
        >>> pipeline.save("./model", SaveFormat.ONNX)
    """

    data_provider_cls = SimpleNerDataProvider
    data_quality_tuner_cls = TokenClassDynamicTuner
    data_quality_evaluators_classes = [Retag, Uncertainty, LabelAggregation]
    training_method_cls: Type[methods.Method]
    task_type = "token_classification"

    def diagnose(self) -> Optional[DatasetEvaluatorOutput]:
        """Run data quality diagnostics on the NER training dataset.

        Evaluates the training data using Retag, Uncertainty, and
        LabelAggregation evaluators to identify problematic samples.

        Returns:
            DatasetEvaluatorOutput with evaluation results.
        """
        evaluation_result: DatasetEvaluatorOutput = self.evaluate_training_data()
        return evaluation_result

    def _synergize_data_and_method(self) -> tuple[Type[methods.Method], Dataset]:
        """Analyze NER data and select the appropriate training method.

        Counts entity label distribution and selects the training method
        based on the maximum label count in the dataset.

        Returns:
            Tuple of (selected method class, training dataset).
        """
        training_data = self.splits[TRAIN_KEY]
        label_counts, _ = count_labels_ner(training_data)
        method_name = resolve_ner_method(label_counts.max())
        return get_method_from_string(OOD_METHOD_MAP[method_name]), training_data


class TokenClassificationInferenceManager(AbstractInferencePipeline):
    """Inference pipeline for token classification (NER) models.

    Loads a trained NER model and provides entity extraction
    capabilities for new text inputs.

    Example:
        >>> from open_autonlu.auto_classes import TokenClassificationInferenceManager
        >>>
        >>> inferer = TokenClassificationInferenceManager("./model")
        >>> results = inferer.predict(["John works at Google"], batch_size=1)
        >>> for r in results:
        ...     print(r.entities)
    """

    def predict(
        self, texts: List[str], batch_size: int, return_all_hypotheses: bool = False
    ) -> List[TokenClassifierOutput]:
        """Run NER inference on a list of texts.

        Args:
            texts: List of text strings to extract entities from.
            batch_size: Number of texts to process in each batch.
            return_all_hypotheses: Not supported for token classification.
                Will be ignored with a debug warning.

        Returns:
            List of TokenClassifierOutput, one per input text, containing
            extracted entities with their spans and labels.
        """
        if return_all_hypotheses:
            log.debug(
                "return_all_hypotheses is not supported for Token Classification and will be ignored."
            )
        return self.inferer.predict(
            texts,
            batch_size=batch_size,
        )
