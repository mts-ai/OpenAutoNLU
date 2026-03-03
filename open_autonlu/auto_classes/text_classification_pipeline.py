import logging
from typing import List, Optional, Type
from pathlib import Path
from datasets import Dataset

from .. import methods
from .abstract_pipeline import (
    AbstractInferencePipeline,
    AbstractTrainingPipeline,
    get_method_from_string,
)
from ..constants import (
    ANCSETFIT_NAME,
    FINETUNING_NAME,
    LOW_RES_CLS_FRACT_THRESHOLD,
    MAX_ANC_SET_FIT_SIZE,
    MAX_SET_FIT_SIZE,
    OOD_METHOD_MAP,
    NO_OOD_METHOD_MAP,
    SETFIT_NAME,
    TRAIN_KEY,
    UPSAMPLE_AUGMENTEX,
    UPSAMPLE_LLM,
    TEST_KEY,
)
from ..data.simple_data_provider import SimpleDataProvider
from ..data.utils import anc_label_is_present
from ..data_quality.data_types import DatasetEvaluatorOutput
from ..data_quality.dynamic_finetuner import SeqClassDynamicTuner
from ..data_quality.sample_eval import Retag, Uncertainty, VInfo
from ..method_resolver import (
    count_labels,
    downsample,
    resolve_method,
    upsample,
    validate_min_size,
)
from ..methods.data_types import TextClassifierOutput, OodMethod
from ..llm_pipelines.pipelines import DataGenerationPipeline
from ..llm_pipelines.prompts import get_prompts
from typing import Dict, Set

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TextClassificationTrainingPipeline(AbstractTrainingPipeline):
    """Training pipeline for text classification tasks.

    Automatically selects the optimal training method based on dataset size:
    - AncSetFit for small datasets with anchor labels (few-shot learning)
    - SetFit for medium datasets
    - Finetuning for larger datasets

    Supports LLM-based data augmentation and synthetic test generation.

    Example:
        >>> from open_autonlu.auto_classes import TextClassificationTrainingPipeline
        >>> from open_autonlu.methods.data_types import OodMethod, SaveFormat
        >>>
        >>> pipeline = TextClassificationTrainingPipeline(
        ...     train_path="train.csv",
        ...     test_path="test.csv",
        ...     config_overrides={"ood_method": OodMethod.LOGIT}
        ... )
        >>> result = pipeline.train()
        >>> pipeline.save("./model", SaveFormat.ONNX)
    """

    data_provider_cls = SimpleDataProvider
    data_quality_tuner_cls = SeqClassDynamicTuner
    data_quality_evaluators_classes = [Retag, Uncertainty, VInfo]
    training_method_cls: Type[methods.Method]
    task_type = "multiclass_classification"

    def _synergize_data_and_method(self) -> tuple[Type[methods.Method], Dataset]:
        """Analyze data and select the appropriate training method.

        This method:
        1. Counts label distribution in training data
        2. Upsamples underrepresented classes if needed (using LLM or Augmentex)
        3. Downsamples overrepresented classes for few-shot methods
        4. Selects the appropriate training method based on data size
        5. Chooses OOD vs non-OOD variant based on config

        Returns:
            Tuple of (selected method class, processed training dataset).
        """
        training_data = self.splits[TRAIN_KEY]
        label_counts, min_class_size = count_labels(training_data)
        log.info(label_counts)

        # Upsample if needed
        low_res_cls_fraction = (label_counts <= MAX_SET_FIT_SIZE).values.sum() / len(
            label_counts
        )
        llm_aug_config = (
            self.config_overrides.get("llm_augmentation", {})
            if self.config_overrides
            else {}
        )
        llm_aug_enabled = llm_aug_config.get("enabled", False)
        should_upsample = (llm_aug_enabled and low_res_cls_fraction > 0.0) or (
            low_res_cls_fraction > 0.0
            and low_res_cls_fraction <= LOW_RES_CLS_FRACT_THRESHOLD
        )
        if should_upsample:
            upsample_method = UPSAMPLE_LLM if llm_aug_enabled else UPSAMPLE_AUGMENTEX
            if llm_aug_enabled:
                self._maybe_analyze_domain(
                    use_domain_analysis=llm_aug_config.get(
                        "use_domain_analysis", False
                    ),
                    config_overrides=llm_aug_config.get("config_overrides"),
                )
            training_data = upsample(
                training_data,
                label_counts,
                MAX_SET_FIT_SIZE + 1,
                upsample_method=upsample_method,
                llm_config_overrides=llm_aug_config if llm_aug_enabled else None,
                domain_desc=self._domain_desc,
                label_descriptions=self._label_descriptions,
                language=self.language,
            )
            label_counts, min_class_size = count_labels(training_data)
            log.info(label_counts)
            method_name = FINETUNING_NAME
        else:
            method_name = resolve_method(min_class_size)

        validate_min_size(label_counts, min_class_size)

        # Downsample if needed
        is_ancsetfit_and_no_anc_label = (
            method_name == ANCSETFIT_NAME
        ) and not anc_label_is_present(self.splits[TRAIN_KEY])
        if method_name == SETFIT_NAME or is_ancsetfit_and_no_anc_label:
            if is_ancsetfit_and_no_anc_label:
                method_name = SETFIT_NAME
                log.warning(
                    "anc_label column is absent. Can't use AncSetFit. Falling back to SetFit method"
                )
            training_data = downsample(training_data, label_counts, MAX_SET_FIT_SIZE)
        elif method_name == ANCSETFIT_NAME and anc_label_is_present(training_data):
            training_data = downsample(
                training_data, label_counts, MAX_ANC_SET_FIT_SIZE
            )

        # Choose OOD/no-OOD method
        try:
            # ood_method in config_overrides
            ood_method = (
                self.config_overrides["ood_method"]
                if self.config_overrides is not None
                else OodMethod.AUTO
            )
        except KeyError:
            # ood_method is not specified
            ood_method = OodMethod.AUTO

        if ood_method in (OodMethod.NONE, OodMethod.LOGIT):
            return get_method_from_string(NO_OOD_METHOD_MAP[method_name]), training_data
        else:
            return get_method_from_string(OOD_METHOD_MAP[method_name]), training_data

    def train(self) -> methods.data_types.TrainingArtifactInfo:
        """Train the text classification model.

        Extends the base train method to optionally generate a synthetic
        test dataset using LLM before training if configured.

        Returns:
            TrainingArtifactInfo containing training results and metrics.
        """
        self._generate_test_dataset_if_needed()
        return super().train()

    def _generate_test_dataset_if_needed(self) -> None:
        """Generate synthetic test.

        Generation logic:
        - If llm_test_generation.enabled = False -> skip generation
        - If test data already exists (test_path was provided) -> skip generation
        - If llm_test_generation.enabled = True and no test data exists -> generate
        """

        llm_config = (
            self.config_overrides.get("llm_test_generation", {})
            if self.config_overrides
            else {}
        )

        if not llm_config.get("enabled", False):
            self._log.info("LLM test generation is disabled or not configured.")
            return

        if TEST_KEY in self.splits:
            self._log.info(
                "Test data already exists (test_path was provided). Skipping generation."
            )
            return

        self._maybe_analyze_domain(
            use_domain_analysis=llm_config.get("use_domain_analysis", False),
            config_overrides=llm_config.get("config_overrides"),
        )

        train_df = self.splits[TRAIN_KEY].to_pandas()
        num_samples_per_class = llm_config.get("num_samples_per_class", 100)
        config_overrides = llm_config.get("config_overrides", {})

        pipeline = DataGenerationPipeline(
            reference_data=train_df,
            text_column="text",
            label_column="label",
            prompt=get_prompts(self.language).generate_texts,
            config_overrides=config_overrides,
            domain_desc=self._domain_desc,
            label_descriptions=self._label_descriptions,
            language=self.language,
        )

        all_generated_texts_per_class: Dict[str, Set[str]] = {}
        labels = [str(x) for x in train_df["label"].unique().tolist()]
        for label in labels:
            all_generated_texts_per_class[label] = set()

        max_attempts = config_overrides.get("max_attempts", 10)
        min_request_size = config_overrides.get("min_request_size", 50)
        attempt = 0

        while attempt < max_attempts:
            classes_still_needing = {
                label
                for label in labels
                if len(all_generated_texts_per_class[label]) < num_samples_per_class
            }

            if not classes_still_needing:
                break

            max_remaining = max(
                num_samples_per_class - len(all_generated_texts_per_class[label])
                for label in classes_still_needing
            )
            request_size = max(int(max_remaining * 1.2) + 1, min_request_size)

            try:
                generated_dataset = pipeline.generate(
                    {"num_generations_per_class": request_size, "num_shot": 5}
                )

                if generated_dataset is None:
                    attempt += 1
                    continue

                generated_df = generated_dataset.to_pandas()
                for label in classes_still_needing:
                    label_generated = generated_df[generated_df["label"] == label]
                    if not label_generated.empty:
                        new_texts = label_generated["text"].astype(str).tolist()
                        for text in new_texts:
                            if text not in all_generated_texts_per_class[label]:
                                all_generated_texts_per_class[label].add(text)
                                if (
                                    len(all_generated_texts_per_class[label])
                                    >= num_samples_per_class
                                ):
                                    break

                attempt += 1
            except Exception as e:
                self._log.warning(
                    "Error during generation on attempt %d: %s", attempt + 1, e
                )
                attempt += 1
                continue

        out = {"text": [], "label": []}
        for label in labels:
            generated_texts = all_generated_texts_per_class[label]
            texts = list(generated_texts)[:num_samples_per_class]
            out["text"].extend(texts)
            out["label"].extend([label] * len(texts))

        test_dataset = Dataset.from_dict(out) if out["text"] else None

        if test_dataset is not None:
            self.splits[TEST_KEY] = test_dataset
            self._log.info("Generated test dataset with %s samples", len(test_dataset))

            try:
                synthetic_test_path = llm_config.get("synthetic_test_path")
                if synthetic_test_path is not None:
                    synthetic_test_path = Path(synthetic_test_path)
                    synthetic_test_path.parent.mkdir(parents=True, exist_ok=True)
                    test_dataset.to_pandas().to_csv(synthetic_test_path, index=False)
                    self._log.info(
                        "Saved synthetic test dataset to %s", synthetic_test_path
                    )
            except Exception as e:
                self._log.warning("Failed to save synthetic test dataset: %s", e)

    def diagnose(self) -> Optional[DatasetEvaluatorOutput]:
        """Run data quality diagnostics on the training dataset.

        Evaluates the training data using configured quality evaluators
        (Retag, Uncertainty, VInfo) to identify problematic samples.
        Skips evaluation for very small datasets that would use AncSetFit.

        Returns:
            DatasetEvaluatorOutput with evaluation results, or None if
            the dataset is too small for meaningful quality evaluation.
        """
        _, min_class_size = count_labels(self.splits[TRAIN_KEY])
        # Check original data
        method_name = resolve_method(min_class_size)

        if method_name != ANCSETFIT_NAME:
            # Create a filtered version of original dataset
            evaluation_result: Optional[DatasetEvaluatorOutput] = (
                self.evaluate_training_data()
            )
        else:
            log.warning(
                "There are not enough data to employ Data Quality Technology. Skipping Data Quality Stage."
            )
            evaluation_result = None
        return evaluation_result


class TextClassificationInferenceManager(AbstractInferencePipeline):
    """Inference pipeline for text classification models.

    Loads a trained text classification model and provides prediction
    capabilities for new text inputs.

    Example:
        >>> from open_autonlu.auto_classes import TextClassificationInferenceManager
        >>>
        >>> inferer = TextClassificationInferenceManager("./model")
        >>> results = inferer.predict(["Hello world", "Goodbye"], batch_size=32)
        >>> for r in results:
        ...     print(f"{r.label}: {r.score:.3f}")
    """

    def predict(
        self,
        texts: List[str],
        batch_size: int,
        return_all_hypotheses: bool = False,
        auto_find_batch_size: bool = False,
    ) -> list[TextClassifierOutput]:
        """Run inference on a list of texts.

        Args:
            texts: List of text strings to classify.
            batch_size: Number of texts to process in each batch.
            return_all_hypotheses: If True, return scores for all classes.
                Currently only supported for ONNX-based inference.
            auto_find_batch_size: If True, automatically reduce batch size
                on OOM errors until a working size is found.

        Returns:
            List of TextClassifierOutput, one per input text, containing
            predicted label and confidence score.
        """
        return self.inferer.predict(
            texts,
            batch_size=batch_size,
            return_all_hypotheses=return_all_hypotheses,
            auto_find_batch_size=auto_find_batch_size,
        )
