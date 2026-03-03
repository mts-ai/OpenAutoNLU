import logging
import os
from collections import Counter
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from accelerate.utils import find_executable_batch_size
from datasets import Dataset, DatasetDict, concatenate_datasets
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline as ort_pipeline
from optuna import Trial
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    set_seed,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import EarlyStoppingCallback

from ..constants import (
    DATA_PROCESSING_STAGE,
    DEV_EVALUATION_STAGE,
    HPO_STAGE,
    TRAINING_STAGE,
)
from ..methods.data_types import OodMethod
from ..methods.constants import OOS_LABEL
from ..methods.utils.misc import initialize_transformers_tokenizer
from ..progress_reporter import (
    ProgressReporter,
    TrainerProgressReporterCallback,
)
from .configs import FinetunerConfig
from .constants import BATCH_SIZE
from .data_types import (
    ClassificationEvaluationResult,
    SaveFormat,
    SimplePyTorchDataset,
    TextClassifierOutput,
    TextLabel,
    TrainingArtifactInfo,
)
from .hyp_search_mixin import FinetuningHyperparameterSearchMixin
from .method import SequenceClassificationMethod
from .onnx_inference_manager import AbstractONNXInferenceManager
from .utils import GibberishDatasetGenerator

log = logging.getLogger(__name__)


class FinetunerBase(SequenceClassificationMethod, FinetuningHyperparameterSearchMixin):
    """Base class for finetuning sequence classification models.

    Finetunes a pretrained transformer model for text classification.
    Supports hyperparameter optimization via Optuna and early stopping.

    Attributes:
        config_cls: Configuration class for this method.
        stages: List of training stages for progress reporting.
    """

    config_cls = FinetunerConfig
    stages = [DATA_PROCESSING_STAGE, HPO_STAGE, TRAINING_STAGE, DEV_EVALUATION_STAGE]

    def __init__(
        self,
        config: Optional[FinetunerConfig] = None,
        progress_reporter: Optional[ProgressReporter] = None,
    ):
        """Initialize the Finetuner.

        Args:
            config: Configuration object with model and training parameters.
                If None, uses default FinetunerConfig.
            progress_reporter: Reporter for tracking training progress.
                If None, creates a default reporter.
        """
        if config is None:
            config = FinetunerConfig()
        if progress_reporter is None:
            progress_reporter = ProgressReporter()
        self.progress_reporter = progress_reporter
        self.model_name = config.model_name_or_path
        self.num_hpo_trials = config.num_hpo_trials
        self.training_args = config.training_arguments
        self.seed = self.training_args.seed
        self.ood_method = config.ood_method

        self.training_args = replace(
            self.training_args,
            report_to="none",
            overwrite_output_dir=True,
        )
        self.dev_fraction = config.dev_fraction
        self.tokenizer = initialize_transformers_tokenizer(
            self.model_name, config.max_seq_length, config.tokenizer_kwargs
        )
        self.trainer = None
        self.model = None
        self.id2label = None
        self.label2id = None
        self.hyp_search_done = False
        self.progress_reporter = progress_reporter

    @property
    def id2label(self):
        if getattr(self, "model"):
            return self.model.config.id2label
        else:
            return self._id2label

    @id2label.setter
    def id2label(self, value):
        self._id2label = value

    def preprocess_function(self, examples):
        """Takes a batch of examples and tokenizes it. Adds labels to the batch if present"""
        tokenized_examples = self.tokenizer(examples["text"], truncation=True)
        labels = examples.get("label")
        if labels is not None:
            if self.model:
                label2id = self.model.config.label2id
            else:
                label2id = self.label2id
            tokenized_examples["labels"] = [label2id[label] for label in labels]
        return tokenized_examples

    @staticmethod
    def _postprocess_logits(logits: np.ndarray) -> np.ndarray:
        return softmax(logits, axis=-1)

    def compute_metrics_train(self, eval_preds):
        """Computes macro f1 metric during training"""
        logits, labels = eval_preds
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, predictions, average="macro")}

    def compute_metrics_eval(self, eval_preds, id2label=None):
        logits, labels = eval_preds
        labels = [id2label[label] for label in labels]
        if isinstance(logits, tuple):
            logits = logits[0]
        labels_hat = self._extract_labels(logits, id2label)
        clf_report = classification_report(
            labels,
            labels_hat,
            output_dict=True,
        )
        result = {
            "classification_report": clf_report,
            "confusion_matrix": confusion_matrix(
                y_true=labels, y_pred=labels_hat, labels=self.label_list
            ),
            "f1": clf_report["macro avg"]["f1-score"],
        }
        return result

    def _split_data(self, train_dataset: Dataset) -> DatasetDict:
        train_ids, eval_ids = train_test_split(
            range(train_dataset.num_rows),
            test_size=self.dev_fraction,
            random_state=self.seed,
            stratify=train_dataset["label"],
        )
        return DatasetDict(
            {
                "train": train_dataset.select(train_ids),
                "eval": train_dataset.select(eval_ids),
            }
        )

    def _generate_synthetic_oos(self, num_train_rows: int) -> Dataset:
        generator = GibberishDatasetGenerator(self.training_args.seed)
        synthetic_texts = generator(num_rows=num_train_rows)
        synthetic_dataset = Dataset.from_dict(
            {"text": synthetic_texts, "label": [OOS_LABEL] * num_train_rows}
        )
        return synthetic_dataset

    def train(self, train_dataset: Dataset) -> TrainingArtifactInfo:
        """
        Trains the model using the provided training dataset.

        Implementation Details:
            - The method first splits the training dataset into training and evaluation subsets.
            - It then initializes the model, data collator, and trainer.
            - If hyperparameter optimization (HPO) is enabled, it performs HPO using the specified search space and number of trials.
            - After HPO, it reconfigures the trainer with the best hyperparameters found.
            - Finally, it trains the model using the trainer and updates the model attribute with the trained model.

        Args:
            train_dataset (Dataset): The training dataset containing the labeled data.

        Returns:
            TrainingArtifactInfo: Information about the trained model and its performance.
        """
        self.progress_reporter.stages = self.stages
        self.progress_reporter(DATA_PROCESSING_STAGE, 0.0)
        set_seed(self.seed)

        label_counts = Counter(train_dataset["label"])
        ood_count = label_counts.get(OOS_LABEL, 0)
        p95 = int(np.percentile(list(label_counts.values()), 95))
        if self.ood_method == OodMethod.NONE:
            if ood_count > 0:
                synthetic_texts_ds = self._generate_synthetic_oos(p95 - ood_count)
                train_dataset = concatenate_datasets(
                    [train_dataset, synthetic_texts_ds]
                )
        elif self.ood_method == OodMethod.LOGIT:
            if ood_count == 0:
                synthetic_texts_ds = self._generate_synthetic_oos(p95)
                train_dataset = concatenate_datasets(
                    [train_dataset, synthetic_texts_ds]
                )
            elif ood_count < p95:
                synthetic_texts_ds = self._generate_synthetic_oos(p95 - ood_count)
                train_dataset = concatenate_datasets(
                    [train_dataset, synthetic_texts_ds]
                )

        labels = train_dataset.unique("label")
        self.num_labels = len(labels)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.label2id, self.id2label = self._get_label_mappings(labels)
        data = self._split_data(train_dataset).map(
            self.preprocess_function,
            remove_columns=train_dataset.column_names,
            batched=True,
        )
        self.progress_reporter(DATA_PROCESSING_STAGE, 1.0)
        self.training_args.configure_for_early_stopping()
        self.trainer = Trainer(
            model_init=self._prepare_model_init_function(),
            args=self.training_args,
            train_dataset=data["train"],
            eval_dataset=data["eval"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_train,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        if self.training_args.hyp_search_space and not self.hyp_search_done:
            log.info("Running hyperparameter optimization")
            self.progress_reporter(HPO_STAGE, 0.0)
            best_params = self.optimize_hyperparameters(
                self.trainer,
                self.training_args.hyp_search_space,
                self.num_hpo_trials,
                progress_reporter=self.progress_reporter,
            )
            self.progress_reporter(HPO_STAGE, 1.0)
            self.training_args = replace(self.training_args, **best_params)
            self.training_args.configure_for_early_stopping()
            self.trainer = Trainer(
                model_init=self._prepare_model_init_function(),
                args=self.training_args,
                train_dataset=data["train"],
                eval_dataset=data["eval"],
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics_train,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=5),
                    TrainerProgressReporterCallback(
                        self.progress_reporter, TRAINING_STAGE
                    ),
                ],
            )

            self.hyp_search_done = True
        self.trainer.train()
        self.model = self.trainer.model
        self.trainer.compute_metrics = partial(
            self.compute_metrics_eval, id2label=self.id2label
        )
        self.progress_reporter(DEV_EVALUATION_STAGE, 0.0)
        metrics: Dict[str, Any] = self.trainer.evaluate(metric_key_prefix="")
        self.progress_reporter(DEV_EVALUATION_STAGE, 1.0)

        # Set actual train batch size before reporting
        if (
            hasattr(self.training_args, "auto_find_batch_size")
            and self.training_args.auto_find_batch_size is True
        ):
            self.training_args.per_device_train_batch_size = (
                self.trainer.state.train_batch_size
            )

        return TrainingArtifactInfo(
            self.training_args.to_dict(),
            self.label_list,
            ClassificationEvaluationResult(
                classification_report=metrics["_classification_report"],
                confusion_matrix=metrics["_confusion_matrix"],
                f1=metrics["_f1"],
            ),
        )

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
        # Mitigating the issue with missing labels in model label set
        # by removing labels from the dataset before preprocessing
        # so that preprocess_finction will not try to map unknown labels to ids
        columns_to_remove = dataset.column_names
        columns_to_remove.remove("label")
        preprocessed_inputs = dataset.remove_columns("label").map(
            self.preprocess_function, remove_columns=columns_to_remove, batched=True
        )
        prediction_output = self.trainer.predict(preprocessed_inputs).predictions
        labels_hat = self._extract_labels(prediction_output, self.id2label)

        if inscope_only:
            oos_label_id = self.label2id.get(OOS_LABEL)
            if oos_label_id is not None:
                inscope_label_ids = [
                    label_id
                    for label_id, label in self.id2label.items()
                    if label != OOS_LABEL
                ]
                if inscope_label_ids:
                    for i, pred_label in enumerate(labels_hat):
                        if pred_label == OOS_LABEL:
                            logits = prediction_output[i]
                            masked_logits = logits.copy()
                            masked_logits[oos_label_id] = -np.inf
                            top_inscope_label_id = inscope_label_ids[
                                np.argmax(
                                    [
                                        masked_logits[label_id]
                                        for label_id in inscope_label_ids
                                    ]
                                )
                            ]
                            labels_hat[i] = self.id2label[top_inscope_label_id]

            original_labels = dataset["label"]
            mask = [label != OOS_LABEL for label in original_labels]
            dataset = dataset.filter(lambda example, idx: mask[idx], with_indices=True)
            labels_hat = [label for label, keep in zip(labels_hat, mask) if keep]

        ds_label_set = set(dataset.unique("label"))
        if inscope_only and OOS_LABEL in ds_label_set:
            ds_label_set.remove(OOS_LABEL)

        model_label_set = set(self.label_list)
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
        labels_for_conf_matrix = self.label_list
        if ds_label_set != model_label_set:
            labels_for_conf_matrix = self.label_list + list(
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
        preprocessed_inputs = SimplePyTorchDataset(
            self.tokenizer(texts, truncation=True).data
        )
        if batch_size != self.training_args.per_device_eval_batch_size:
            self.training_args.per_device_eval_batch_size = batch_size
        prediction_output = self.trainer.predict(preprocessed_inputs).predictions
        prediction_output = self._postprocess_logits(prediction_output)
        labels = self._extract_labels(prediction_output, self.id2label)
        scores = self._extract_scores(prediction_output)
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
                model; ONNX exports an optimized inference model.
        """
        self.trainer.save_model(path_to_package)
        if checkpoint_format == SaveFormat.ONNX:
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                path_to_package, export=True
            )
            ort_model.save_pretrained(path_to_package)
            os.unlink(path_to_package + "/model.safetensors")
        log.info("Model is saved at " + path_to_package)

    @classmethod
    def load(cls, path_to_package: str):
        """Load a trained Finetuner model from disk.

        Args:
            path_to_package: Directory path containing the saved model.

        Returns:
            Initialized FinetunerBase instance with loaded model.
        """
        self = cls()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path_to_package, return_dict=True
        )
        self.trainer = Trainer(
            self.model,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            args=self.training_args,
        )
        log.info("Model is loaded and ready to use")
        return self

    def _prepare_model_init_function(
        self, **model_kwargs
    ) -> Callable[[Trial], PreTrainedModel]:
        default_model_kwargs = {
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if not model_kwargs:
            model_kwargs = default_model_kwargs
        else:
            default_model_kwargs.update(model_kwargs)
            model_kwargs = default_model_kwargs

        def model_init(trial: Trial) -> PreTrainedModel:
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
                **model_kwargs,
            )

        return model_init


class FinetunerInferenceManager(AbstractONNXInferenceManager):
    """ONNX-based inference manager for finetuned models.

    Uses ONNX Runtime for optimized inference on finetuned
    sequence classification models.
    """

    def __init__(self, path_to_package: str) -> None:
        """Initialize the inference manager.

        Args:
            path_to_package: Path to directory with saved ONNX model.
        """
        super().__init__(path_to_package)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            path_to_package, provider=self.ort_provider
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_package)
        self._pipeline = ort_pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,  # type: ignore
            accelerator="ort",
        )

    def predict(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        auto_find_batch_size: bool = False,
        return_all_hypotheses: bool = False,
    ) -> list[TextClassifierOutput]:
        """Run inference on a list of texts.

        Args:
            texts: List of text strings to classify.
            batch_size: Number of texts to process in each batch.
            auto_find_batch_size: If True, automatically reduce batch size
                on OOM errors.
            return_all_hypotheses: Not supported for ONNX inference.

        Returns:
            List of TextClassifierOutput with predicted labels and scores.
        """
        if return_all_hypotheses:
            log.debug(
                "return_all_hypotheses is not supported for ONNX-based inference and will be ignored."
            )
        if auto_find_batch_size:
            return find_executable_batch_size(
                self._predict, starting_batch_size=batch_size
            )(texts)  # type: ignore
        return self._predict(batch_size, texts)

    def _predict(self, batch_size: int, texts: List[str]) -> list[TextClassifierOutput]:
        return [
            TextClassifierOutput(
                text=text, hypotheses=[TextLabel(result["label"], result["score"])]
            )
            for result, text in zip(
                self._pipeline(
                    texts, batch_size=batch_size, truncation=True, padding=True
                ),
                texts,
            )  # type: ignore
        ]


class Finetuner(FinetunerBase):
    """Finetuner method for text classification.

    Concrete implementation of FinetunerBase with ONNX inference support.
    This is the recommended finetuning variant for production use.
    """

    onnx_inference_manager = FinetunerInferenceManager
