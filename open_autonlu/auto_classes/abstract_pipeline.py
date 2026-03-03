import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from datasets import Dataset

from .. import methods
from ..constants import (
    METHOD_HUMAN_READABLE_NAME_MAP,
    OOD_METHOD_MAP,
    TRAIN_KEY,
)
from ..data.simple_data_provider import AbstractDataProvider
from ..data_quality.data_types import DatasetEvaluatorOutput
from ..data_quality.dynamic_finetuner import DynamicTuner
from ..data_quality.evaluator import DatasetEvaluator
from ..data_quality.sample_eval import AbstractSampleEvalMethod
from ..data_types import ProgressCallbackPayload
from ..llm_pipelines import analyze_domain_and_labels
from ..methods.data_types import OodMethod
from ..methods.method import Method, SequenceClassificationMethod
from ..progress_reporter import ProgressReporter
from .constants import LEGACY_META_FILE_NAME, META_FILE_NAME


def get_method_from_string(method_name: str):
    class_ = getattr(methods, method_name)
    return class_


class AbstractTrainingPipeline(ABC):
    """Abstract base class for all training pipelines.

    This class defines the interface and common functionality for training
    pipelines that handle data loading, preprocessing, method selection,
    training, and model serialization.

    Attributes:
        data_provider_cls: Class used to load and parse training data.
        data_quality_tuner_cls: Class for dynamic tuning during data quality evaluation.
        data_quality_evaluators_classes: List of sample evaluation method classes.
        training_method_cls: The training method class to use.
        task_type: String identifier for the task type (e.g., "multiclass_classification").
    """

    data_provider_cls: Type[AbstractDataProvider]
    data_quality_tuner_cls: Type[DynamicTuner]
    data_quality_evaluators_classes: List[Type[AbstractSampleEvalMethod]]
    training_method_cls: Type[methods.Method]
    task_type: str

    def __init__(
        self,
        train_path: str,
        test_path: Optional[str] = None,
        on_progress_changed: Callable[[ProgressCallbackPayload], None] = lambda x: None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the training pipeline.

        Args:
            train_path: Path to the training data file (CSV format).
            test_path: Optional path to the test data file. If not provided,
                test metrics will not be computed after training.
            on_progress_changed: Callback function invoked with progress updates
                during training. Receives a ProgressCallbackPayload object.
            config_overrides: Dictionary to override default configurations.
                Supports multiple resolution strategies:

                1. **By method class name**: Use the method class name as key.
                   ``{"SetFitMethod": {"num_iterations": 30}}``

                2. **By config class name**: Use the config class name as key.
                   ``{"SetFitMethodConfig": {"num_iterations": 30}}``

                3. **Direct field names**: Specify config fields directly.
                   ``{"num_iterations": 30, "batch_size": 32}``

                **Common options:**

                - ``ood_method``: OOD detection method. Values: ``OodMethod.AUTO``,
                  ``OodMethod.NONE``, ``OodMethod.LOGIT``, ``OodMethod.MAHALANOBIS``,
                  ``OodMethod.MSP``.

                - ``llm_augmentation``: LLM-based data augmentation settings::

                    {
                        "llm_augmentation": {
                            "enabled": True,
                            "use_domain_analysis": True,
                            "config_overrides": {
                                "LlmClientConfig": {
                                    "api_key": "your-key",
                                    "base_url": "https://api.example.com/v1/"
                                }
                            }
                        }
                    }

                - ``llm_test_generation``: Synthetic test data generation::

                    {
                        "llm_test_generation": {
                            "enabled": True,
                            "num_samples_per_class": 100,
                            "use_domain_analysis": True,
                            "synthetic_test_path": "./synthetic_test.csv"
                        }
                    }
        """
        language = "en"
        if config_overrides is not None and "language" in config_overrides:
            language = config_overrides["language"]
        self.language = language
        self.splits = self.data_provider_cls(
            train_path=train_path, test_path=test_path, language=language
        ).load()
        self.progress_callback = on_progress_changed
        self.config_overrides = config_overrides
        self._domain_desc: Optional[str] = None
        self._label_descriptions: Optional[Dict[str, str]] = None
        self._log = logging.getLogger(self.__class__.__name__)

    def evaluate_training_data(self) -> DatasetEvaluatorOutput:
        """Evaluate training data quality using configured evaluators.

        Runs the data quality pipeline on the training split, using sample
        evaluators to identify problematic samples (e.g., mislabeled, ambiguous,
        or low-confidence samples).

        Returns:
            DatasetEvaluatorOutput containing evaluation results, including
            filtered dataset and per-sample quality scores.
        """
        method_kwargs = self._resolve_config_overrides(self.data_quality_tuner_cls)
        sample_evaluators = [cls() for cls in self.data_quality_evaluators_classes]  # type: ignore
        progress_reporter = ProgressReporter(self.progress_callback)
        progress_reporter.stages = [e.name for e in sample_evaluators]
        tuner = self.data_quality_tuner_cls(
            self.data_quality_tuner_cls.config_cls(**method_kwargs),
        )
        dataset_evaluator = DatasetEvaluator(
            task_type=self.task_type,
            tuner=tuner,
            sample_evaluators=sample_evaluators,
        )
        # TODO: refactor DQ to directly take in only a Dataset, not a DatasetDict
        # As for now it processes only "train" split, but semantically code looks
        # like it processes all splits.
        evaluation_result: DatasetEvaluatorOutput = dataset_evaluator.filter_data(
            self.splits,
            progress_reporter=progress_reporter,
        )
        return evaluation_result

    # TODO: refactor to use from Mixin
    def _resolve_config_overrides(self, method_cls):
        """Resolve configuration overrides for a given method class.

        Extracts relevant configuration parameters from config_overrides
        using multiple resolution strategies:

        1. If method_cls has ``parent_finetuner_cls_name``, look up by that name
        2. Otherwise, look up by ``method_cls.__name__``
        3. If not found, extract fields that match ``method_cls.config_cls`` attributes

        Args:
            method_cls: The method class to resolve configuration for.

        Returns:
            Dictionary of configuration parameters for the method.
        """
        if self.config_overrides is None:
            return {}

        if hasattr(method_cls, "parent_finetuner_cls_name"):
            method_kwargs = self.config_overrides.get(
                method_cls.parent_finetuner_cls_name
            )
        else:
            method_kwargs = self.config_overrides.get(method_cls.__name__)
        if method_kwargs is None:
            method_kwargs = {
                k: v
                for k, v in self.config_overrides.items()
                if hasattr(method_cls.config_cls, k)
            }
        if method_kwargs.get("ood_method") == OodMethod.AUTO:
            method_kwargs.pop("ood_method")
        return method_kwargs

    @abstractmethod
    def diagnose(self) -> Optional[DatasetEvaluatorOutput]:
        """
        Run the DataQuality untility on splits["train"] and return the result.
        """
        ...

    @abstractmethod
    def _synergize_data_and_method(self) -> tuple[Type[Method], Dataset]:
        """
        Analyze the incoming train dataset, verify its validity
        in terms of data quantity. Upsample, or downsample it,
        and finally resolve the final training method.
        """
        ...

    @property
    def frontend_training_method_name(self) -> str:
        """Get the human-readable name of the training method.

        Returns:
            Human-readable method name for display in frontend/UI.
        """
        return METHOD_HUMAN_READABLE_NAME_MAP[self.training_method_cls.__name__]

    def train(self) -> methods.data_types.TrainingArtifactInfo:
        """Train the model using the configured method and data.

        This method:
        1. Calls _synergize_data_and_method to select the appropriate method
           and preprocess data (upsample/downsample as needed)
        2. Resolves configuration overrides for the selected method
        3. Trains the model on the prepared data
        4. Optionally evaluates on test data if available

        Returns:
            TrainingArtifactInfo containing training results, metrics,
            and model artifacts.
        """
        self.training_method_cls, self.training_data = self._synergize_data_and_method()
        print(self.training_method_cls)
        method_kwargs = self._resolve_config_overrides(self.training_method_cls)
        progress_reporter = ProgressReporter(self.progress_callback)
        self.method_manager = self.training_method_cls(
            self.training_method_cls.config_cls(**method_kwargs),
            progress_reporter=progress_reporter,
        )
        training_artifact_info = self.method_manager.train(self.training_data)
        if test_data := self.splits.get("test"):
            if isinstance(self.method_manager, SequenceClassificationMethod):
                training_artifact_info.test_metrics = self.method_manager.test(
                    test_data, inscope_only=False
                )
            else:
                training_artifact_info.test_metrics = self.method_manager.test(
                    test_data
                )
        return training_artifact_info

    def _maybe_analyze_domain(
        self,
        use_domain_analysis: bool,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not use_domain_analysis:
            return
        if self._domain_desc is not None and self._label_descriptions is not None:
            return

        train_df = self.splits[TRAIN_KEY].to_pandas()
        try:
            label_counts = train_df.groupby("label").size()
            min_label_count = label_counts.min() if len(label_counts) > 0 else 0
            examples_per_label = min(7, min_label_count) if min_label_count > 0 else 5

            self._domain_desc, self._label_descriptions = analyze_domain_and_labels(
                reference_data=train_df,
                text_column="text",
                label_column="label",
                examples_per_label=examples_per_label,
                config_overrides=config_overrides,
                language=self.language,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._log.warning(
                "Failed to perform domain analysis: %s", exc, exc_info=True
            )

    def save(self, path_to_package: str, save_format: methods.data_types.SaveFormat):
        """Save the trained model to disk.

        Serializes the model and writes metadata to the specified directory.
        The metadata file contains information about the method class and
        save format for later reconstruction during inference.

        Args:
            path_to_package: Directory path where the model will be saved.
            save_format: Format for saving (SaveFormat.TORCH or SaveFormat.ONNX).
        """
        self.method_manager.save(path_to_package, save_format)
        with open(path_to_package + "/meta.json", "w") as f:
            json.dump(
                {
                    "method_class": self.training_method_cls.__name__,
                    "type": save_format.value,
                },
                f,
            )


class AbstractInferencePipeline(ABC):
    """Abstract base class for inference pipelines.

    Loads a trained model from disk and provides inference capabilities.
    Automatically detects the model format (Torch or ONNX) and loads
    the appropriate inference manager.
    """

    def __init__(self, path_to_package: str) -> None:
        """Initialize the inference pipeline from a saved model.

        Args:
            path_to_package: Path to the directory containing the saved model
                and metadata file (meta.json).
        """
        path_to_package_obj = Path(path_to_package)
        if (path_to_package_obj / LEGACY_META_FILE_NAME).exists():
            with (path_to_package_obj / LEGACY_META_FILE_NAME).open() as f:
                meta = {
                    "method_class": OOD_METHOD_MAP[f.read().strip()],
                    "type": "torch",
                }
        else:
            with open(path_to_package_obj / META_FILE_NAME) as f:
                meta = json.load(f)
        producer_cls = get_method_from_string(meta["method_class"])
        if (
            methods.data_types.SaveFormat(meta["type"])
            == methods.data_types.SaveFormat.TORCH
        ):
            self.inferer = producer_cls.load(path_to_package)
        else:
            self.inferer = producer_cls.onnx_inference_manager(path_to_package)
