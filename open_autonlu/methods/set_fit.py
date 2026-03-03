import logging
from collections import Counter
from dataclasses import replace
from typing import Any, List, Optional

from deprecated import deprecated

import joblib
import onnxruntime
from datasets import Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
import numpy as np
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, set_seed
from transformers.integrations.integration_utils import ClearMLCallback
from accelerate.utils import find_executable_batch_size
from open_autonlu.progress_reporter import (
    ProgressReporter,
    TrainerProgressReporterCallback,
)

from ..constants import (
    TRAINING_STAGE,
)
from .configs import SetFitMethodConfig
from .data_types import (
    ClassificationEvaluationResult,
    SaveFormat,
    TextClassifierOutput,
    TextLabel,
    TrainingArtifactInfo,
)
from .method import SequenceClassificationMethod
from ..methods.constants import OOS_LABEL
from ..methods.data_types import OodMethod
from .models.onnx_wrappers import SetFitWrapper
from .onnx_inference_manager import (
    AbstractONNXInferenceManager,
    GeneralSequenceClassifierInferenceManager,
)
from .utils import GenericSequenceClassifierONNXExportMixin
from .constants import BATCH_SIZE
from .utils import batch as batcher, GibberishDatasetGenerator

log = logging.getLogger(__name__)


class SetFitMethodBase(
    SequenceClassificationMethod, GenericSequenceClassifierONNXExportMixin
):
    """Base class for SetFit few-shot text classification.

    SetFit is a few-shot learning method that finetunes a sentence transformer and trains a classification head
    on the resulting embeddings.

    Attributes:
        config_cls: Configuration class for this method.
        stages: List of training stages for progress reporting.
    """

    config_cls = SetFitMethodConfig
    stages = [TRAINING_STAGE]

    def __init__(
        self,
        config: Optional[SetFitMethodConfig] = None,
        progress_reporter: Optional[ProgressReporter] = None,
    ) -> None:
        """Initialize the SetFit method.

        Args:
            config: Configuration object with model and training parameters.
                If None, uses default SetFitMethodConfig.
            progress_reporter: Reporter for tracking training progress.
                If None, creates a default reporter.
        """
        super().__init__(config)
        if config is None:
            config = SetFitMethodConfig()
        if progress_reporter is None:
            progress_reporter = ProgressReporter()
        self.progress_reporter = progress_reporter

        if config.model_name_or_path is not None:
            self.model = SetFitModel.from_pretrained(config.model_name_or_path)
        else:
            self.model = None
        self.ood_method = config.ood_method
        self.body_lr = config.body_lr
        self.num_iterations = config.num_iterations
        self.model.model_body.tokenizer.model_max_length = config.max_seq_length  # hardcoded for russian encoders, that lack their own model_max_length parameter
        self.seed = config.seed
        self.batch_size = config.batch_size
        set_seed(config.seed)

    @property
    def id2label(self):
        if getattr(self, "model"):
            return self.model.config.id2label
        else:
            return self._id2label

    @id2label.setter
    def id2label(self, value):
        self._id2label = value

    def _generate_synthetic_oos(self, num_train_rows: int) -> Dataset:
        generator = GibberishDatasetGenerator(self.seed)
        synthetic_texts = generator(num_rows=num_train_rows)
        synthetic_dataset = Dataset.from_dict(
            {"text": synthetic_texts, "label": [OOS_LABEL] * num_train_rows}
        )
        return synthetic_dataset

    def train(self, dataset: Dataset) -> TrainingArtifactInfo:
        """Train the SetFit model on the provided dataset.

        Args:
            dataset: Training dataset with 'text' and 'label' columns.

        Returns:
            TrainingArtifactInfo with training arguments and label list.
        """

        self.progress_reporter.stages = self.stages
        self.progress_reporter(TRAINING_STAGE, 0.0)
        fit_with_auto_batch_size = find_executable_batch_size(
            self._fit_with_auto_batch_size, starting_batch_size=self.batch_size
        )
        label_counts = Counter(dataset["label"])
        ood_count = label_counts.get(OOS_LABEL, 0)
        p95 = int(np.percentile(list(label_counts.values()), 95))
        if self.ood_method == OodMethod.NONE:
            if ood_count > 0:
                synthetic_texts_ds = self._generate_synthetic_oos(p95 - ood_count)
                dataset = concatenate_datasets([dataset, synthetic_texts_ds])
        elif self.ood_method == OodMethod.LOGIT:
            if ood_count == 0:
                synthetic_texts_ds = self._generate_synthetic_oos(p95)
                dataset = concatenate_datasets([dataset, synthetic_texts_ds])
            elif ood_count < p95:
                synthetic_texts_ds = self._generate_synthetic_oos(p95 - ood_count)
                dataset = concatenate_datasets([dataset, synthetic_texts_ds])
        trainer = fit_with_auto_batch_size(dataset=dataset)
        # Model swapping
        self.base_setfit_instance = self.model
        model_pooler = self.model.model_body._modules["1"]

        self.model = SetFitWrapper(
            self.model.model_body._modules["0"].auto_model,
            model_pooler,
            "2" in self.model.model_body._modules,
            self.model.model_head,
        ).eval()
        return TrainingArtifactInfo(trainer.args.to_dict(), self.label_list)

    def _fit_with_auto_batch_size(self, batch_size: int, dataset: Dataset) -> Trainer:
        trainer = Trainer(
            model=self.model,
            train_dataset=dataset,
            args=TrainingArguments(
                batch_size=batch_size,
                body_learning_rate=self.body_lr,
                report_to="none",
                seed=self.seed,
            ),
        )
        # argument `report_to="none"` is ignored during instantiation of BCSentenceTransformersTrainer
        # inside of Trainer, so we have to remove all integration callbacks manually
        if hasattr(trainer, "st_trainer"):
            if "clearml" not in trainer.args.report_to:
                trainer.st_trainer.remove_callback(ClearMLCallback)
            trainer.st_trainer.add_callback(
                TrainerProgressReporterCallback(self.progress_reporter, TRAINING_STAGE)
            )
        if self.num_iterations is not None:
            trainer.args = replace(trainer.args, num_iterations=self.num_iterations)
        trainer.train()
        return trainer

    def test(
        self, dataset: Dataset, inscope_only: bool = False
    ) -> ClassificationEvaluationResult:
        """Evaluate the model on a test dataset.

        Args:
            dataset: Test dataset with 'text' and 'label' columns.
            inscope_only: If True, exclude OOS predictions and labels from
                evaluation, replacing OOS predictions with the most probable
                in-scope label.

        Returns:
            ClassificationEvaluationResult with confusion matrix,
            classification report, and macro F1 score.
        """
        predictions = self.predict(dataset["text"], batch_size=BATCH_SIZE)
        labels_hat = [pred.most_probable.label for pred in predictions]

        if inscope_only:
            for i, pred in enumerate(predictions):
                if pred.most_probable.label == OOS_LABEL:
                    inscope_hypotheses = [
                        h for h in pred.hypotheses if h.label != OOS_LABEL
                    ]
                    if inscope_hypotheses:
                        inscope_hypotheses.sort(key=lambda x: x.score, reverse=True)
                        labels_hat[i] = inscope_hypotheses[0].label

            original_labels = dataset["label"]
            mask = [label != OOS_LABEL for label in original_labels]
            dataset = dataset.filter(lambda example, idx: mask[idx], with_indices=True)
            labels_hat = [label for label, keep in zip(labels_hat, mask) if keep]
            inscope_labels = [label for label in self.label_list if label != OOS_LABEL]
        else:
            inscope_labels = None

        labels_for_conf_matrix = inscope_labels if inscope_only else self.label_list
        ds_label_set = set(dataset.unique("label"))
        if inscope_only and OOS_LABEL in ds_label_set:
            ds_label_set.remove(OOS_LABEL)
        model_label_set = set(labels_for_conf_matrix)
        extra_in_test = ds_label_set - model_label_set
        if extra_in_test:
            log.warning(
                "Test dataset contains labels not seen during training: %s",
                sorted(extra_in_test),
            )

        clf_report = classification_report(
            dataset["label"],
            labels_hat,
            labels=sorted(ds_label_set),
            output_dict=True,
            zero_division=0,
        )
        # Mitigating the issue with missing labels in model label set
        # while retaining the order of labels in the confusion matrix
        if ds_label_set != model_label_set:
            labels_for_conf_matrix = labels_for_conf_matrix + list(
                ds_label_set - model_label_set
            )

        conf_matrix = confusion_matrix(
            dataset["label"], labels_hat, labels=labels_for_conf_matrix
        )
        return ClassificationEvaluationResult(
            conf_matrix, clf_report, clf_report["macro avg"]["f1-score"]
        )

    def _predict(
        self,
        batch_size: int,
        texts: List[str],
    ) -> list[TextClassifierOutput]:
        if isinstance(texts, str):
            texts = [texts]

        logits = self.base_setfit_instance.predict_proba(
            texts, batch_size=batch_size, as_numpy=True
        )
        labels = self._extract_labels(logits, self.id2label)
        scores = self._extract_scores(logits)
        output = [
            TextClassifierOutput(text=text, hypotheses=[TextLabel(label, score)])
            for label, score, text in zip(labels, scores, texts)
        ]
        return output

    def save(
        self, path_to_package: str, checkpoint_format: SaveFormat = SaveFormat.TORCH
    ) -> None:
        """Save the trained model to disk.

        Args:
            path_to_package: Directory path for saving the model.
            checkpoint_format: Format to save in. TORCH saves the full
                SetFit model; ONNX exports an optimized inference model.
        """
        if checkpoint_format == SaveFormat.TORCH:
            self.base_setfit_instance.save_pretrained(path_to_package)
        else:
            self.tokenizer = self.base_setfit_instance.model_body.tokenizer
            self.export_onnx(path_to_package)
            self.tokenizer.save_pretrained(path_to_package)
        log.info("Model is saved at " + path_to_package)

    @classmethod
    def load(cls, path_to_package: str):
        """Load a trained SetFit model from disk.

        Args:
            path_to_package: Directory path containing the saved model.

        Returns:
            Initialized SetFitMethodBase instance with loaded model.
        """
        self = cls()  # TODO: add config loading
        model_body = SentenceTransformer(path_to_package)
        model_head = joblib.load(path_to_package + "/model_head.pkl")
        self.model = SetFitModel(model_body=model_body, model_head=model_head)
        self.base_setfit_instance = self.model
        self.model = SetFitWrapper(
            model_body._modules["0"].auto_model,
            model_body._modules["1"],
            "2" in model_body._modules,
            model_head,
        ).eval()
        return self


@deprecated(
    version="0.3.5",
    reason="This class is outdated, use GeneralSequenceClassifierInferenceManager instead.",
)
class SetFitInferenceManager(AbstractONNXInferenceManager):
    def __init__(self, path_to_package: str) -> None:
        super().__init__(path_to_package)
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_package)
        self.session = onnxruntime.InferenceSession(
            path_to_package + "/model.onnx",
            providers=[self.ort_provider, "CPUExecutionProvider"],
        )

    def predict(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        auto_find_batch_size: bool = False,
        return_all_hypotheses: bool = False,
    ):
        if return_all_hypotheses:
            log.debug(
                "return_all_hypotheses is not supported for torch-based inference for now and will be ignored."
            )
        if auto_find_batch_size:
            return find_executable_batch_size(
                self._predict, starting_batch_size=batch_size
            )(texts)  # type: ignore
        return self._predict(batch_size, texts)

    def _predict(self, batch_size: int, texts: List[str]) -> List[Any]:
        all_labels, all_scores = [], []
        for text_batch in batcher(texts, batch_size):
            inputs = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="np",
            )
            labels, logits = self.session.run(None, dict(inputs))
            scores = SetFitMethodBase._extract_scores(logits)
            all_labels.extend(labels)
            all_scores.extend(scores)
        return [
            {"label": label, "score": score}
            for label, score in zip(all_labels, all_scores)
        ]


class SetFitMethod(SetFitMethodBase):
    """SetFit method for few-shot text classification.

    Concrete implementation of SetFitMethodBase that uses the standard
    GeneralSequenceClassifierInferenceManager for ONNX inference.

    This is the recommended SetFit variant for production use.
    """

    onnx_inference_manager = GeneralSequenceClassifierInferenceManager
