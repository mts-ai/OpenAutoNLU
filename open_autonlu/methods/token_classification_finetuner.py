import json
import logging
import os
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from accelerate.utils import find_executable_batch_size
from datasets import Dataset, DatasetDict
from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.pipelines import pipeline as ort_pipeline
from optuna import Trial
from seqeval.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    pipeline,
    set_seed,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import AggregationStrategy
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_callback import EarlyStoppingCallback

from ..constants import (
    DATA_PROCESSING_STAGE,
    DEV_EVALUATION_STAGE,
    HPO_STAGE,
    TRAINING_STAGE,
)
from ..data.utils import convert_offsets_to_bio
from ..methods.configs import TokenClassificationFinetunerConfig
from ..methods.data_types import (
    ClassificationEvaluationResult,
    Entity,
    NERFormat,
    SaveFormat,
    TokenClassifierOutput,
    TrainingArtifactInfo,
)
from ..methods.hyp_search_mixin import (
    FinetuningHyperparameterSearchMixin,
)
from ..methods.method import TokenClassificationMethod
from ..methods.onnx_inference_manager import AbstractONNXInferenceManager
from ..methods.utils import evaluate_entity_level
from ..methods.utils.misc import initialize_transformers_tokenizer
from ..progress_reporter import (
    ProgressReporter,
    TrainerProgressReporterCallback,
)
from .constants import BATCH_SIZE
from .utils import resolve_device

log = logging.getLogger(__name__)


class TokenClassificationFinetunerBase(
    TokenClassificationMethod, FinetuningHyperparameterSearchMixin
):
    """Base class for finetuning token classification (NER) models.

    Finetunes a pretrained transformer model for named entity recognition.
    Supports hyperparameter optimization via Optuna and early stopping.
    Uses BIO tagging scheme for entity labels.

    Attributes:
        config_cls: Configuration class for this method.
        stages: List of training stages for progress reporting.
    """

    config_cls = TokenClassificationFinetunerConfig
    stages = [DATA_PROCESSING_STAGE, HPO_STAGE, TRAINING_STAGE, DEV_EVALUATION_STAGE]

    def __init__(
        self,
        config: Optional[TokenClassificationFinetunerConfig] = None,
        progress_reporter: Optional[ProgressReporter] = None,
    ):
        """Initialize the Token Classification Finetuner.

        Args:
            config: Configuration object with model and training parameters.
                If None, uses default TokenClassificationFinetunerConfig.
            progress_reporter: Reporter for tracking training progress.
                If None, creates a default reporter.
        """
        if config is None:
            config = TokenClassificationFinetunerConfig()
        if progress_reporter is None:
            progress_reporter = ProgressReporter()
        self.device = resolve_device()
        self.model_name = config.model_name_or_path
        self.language = config.language
        self.num_hpo_trials = config.num_hpo_trials
        self.training_args = config.training_arguments
        self.seed = self.training_args.seed

        self.dev_fraction = config.dev_fraction
        self.training_args = replace(
            self.training_args,
            report_to="none",
            overwrite_output_dir=True,
        )
        self.tokenizer = initialize_transformers_tokenizer(
            self.model_name, config.max_seq_length, config.tokenizer_kwargs
        )
        self.hyp_search_done = False
        self.label_names = None
        self.progress_reporter = progress_reporter

    def align_target(
        self, label_ids: List[int], word_ids: List[Optional[int]]
    ) -> List[int]:
        # Initialize an empty list to store aligned labels and a variable to track the last word
        aligned_labels = []
        last_word = None

        # Iterate through the word_ids
        for word_id in word_ids:
            if word_id is None:
                label_id = -100  # Set label to -100 for None word_ids
            elif word_id != last_word:
                label_id = label_ids[
                    word_id
                ]  # Use the label corresponding to the current word_id
            else:
                label_id = label_ids[word_id]
                # Change B- to I- if the previous word is the same
                if label_id in self.begin2inside:
                    label_id = self.begin2inside[label_id]  # Map B- to I-

            # Append the label to the aligned_labels list and update last_word
            aligned_labels.append(label_id)
            last_word = word_id

        return aligned_labels

    @staticmethod
    def get_entity_names(dataset: Dataset) -> List[str]:
        entity_names = []
        for bio_tags in dataset["labels"]:
            entity_names.extend({tag[2:] for tag in bio_tags if tag != "O"})
        return list(sorted(set(entity_names)))

    @staticmethod
    def get_label_names(entity_names: List[str]) -> List[str]:
        label_names = []
        for entity_name in entity_names:
            label_names.extend((f"B-{entity_name}", f"I-{entity_name}"))
        label_names.append("O")
        return label_names

    def tokenize_fn(self, batch: Dict[str, List[Any]]) -> BatchEncoding:
        # Tokenize the input batch
        tokenized_inputs = self.tokenizer(
            batch["tokens"], truncation=True, is_split_into_words=True
        )

        # Extract the labels batch from the input batch
        labels_batch: Sequence[List[int]] = batch["label_ids"]
        span_masks_batch: Sequence[List[int]] = batch.get("span_mask")
        if span_masks_batch:
            aligned_span_masks_batch = []

        # Initialize a list to store aligned targets for each example in the batch
        aligned_targets_batch = []

        # Iterate through each example and align the labels
        for i, label_ids in enumerate(labels_batch):
            # Extract the word_ids for the current example
            word_ids = tokenized_inputs.word_ids(i)

            # Use the align_target function to align the labels
            aligned_targets_batch.append(self.align_target(label_ids, word_ids))

            if span_masks_batch:
                aligned_span_masks_batch.append(
                    self.align_target(span_masks_batch[i], word_ids)
                )

        # Saving BIO tags for stratification
        tokenized_inputs["original_labels"] = batch["labels"]
        # Add the aligned labels to the tokenized inputs under the key "labels"
        tokenized_inputs["labels"] = aligned_targets_batch
        if span_masks_batch:
            tokenized_inputs["span_mask"] = aligned_span_masks_batch

        # Return the tokenized inputs, including aligned labels
        return tokenized_inputs

    def compute_metrics_train(self, logits_and_labels):
        """Function to compute evaluation metrics from model logits and true labels"""

        # Unpack the logits and labels
        logits, labels = logits_and_labels

        # Get predictions from the logits
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens)
        str_labels = [
            [self.label_names[t] for t in label if t != -100] for label in labels
        ]

        str_preds = [
            [self.label_names[p] for (p, t) in zip(prediction, label) if t != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Compute metrics
        results = classification_report(
            str_labels, str_preds, output_dict=True, mode="strict", digits=2
        )

        stats = results["macro avg"]  # type: ignore
        # Extract key metrics
        return {
            "precision": stats["precision"],  # type: ignore
            "recall": stats["recall"],  # type: ignore
            "f1": stats["f1-score"],  # type: ignore
        }

    def train(self, train_dataset: Dataset):
        """Train the token classification model.

        Preprocesses the dataset, runs optional hyperparameter optimization,
        trains the model, and evaluates on a held-out dev set.

        Args:
            train_dataset: Dataset with 'tokens' and 'labels' columns.
                Labels should be in BIO format.

        Returns:
            TrainingArtifactInfo with training args, entity names, and metrics.
        """
        set_seed(self.training_args.seed)
        self.progress_reporter.stages = self.stages
        self.progress_reporter(DATA_PROCESSING_STAGE, 0.0)
        splits, features = self._preprocess_dataset(train_dataset)

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, padding=True
        )

        self.training_args.num_train_epochs = 10
        self.training_args.configure_for_early_stopping()
        # TODO: add progress reporter callback
        self.trainer = Trainer(
            model_init=self._prepare_model_init_function(),
            args=self.training_args,
            train_dataset=features["train"],
            eval_dataset=features["test"],
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics_train,
        )

        if self.training_args.hyp_search_space and not self.hyp_search_done:
            self.progress_reporter(HPO_STAGE, 0.0)
            log.info("Running hyperparameter optimization")
            best_params = self.optimize_hyperparameters(
                self.trainer,
                self.training_args.hyp_search_space,
                self.num_hpo_trials,
                self.progress_reporter,
            )
            self.training_args = replace(self.training_args, **best_params)
            self.training_args.configure_for_early_stopping()
            self.trainer = Trainer(
                model_init=self._prepare_model_init_function(),
                args=self.training_args,
                train_dataset=features["train"],
                eval_dataset=features["test"],
                data_collator=data_collator,
                processing_class=self.tokenizer,
                compute_metrics=self.compute_metrics_train,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=5),
                    TrainerProgressReporterCallback(
                        self.progress_reporter, TRAINING_STAGE
                    ),
                ],
            )
            self.progress_reporter(HPO_STAGE, 1.0)
            self.hyp_search_done = True

        self.progress_reporter(TRAINING_STAGE, 0.0)
        self.trainer.train()
        self.progress_reporter(TRAINING_STAGE, 1.0)
        self.model = self.trainer.model
        self.progress_reporter(DEV_EVALUATION_STAGE, 0.0)
        dev_evaluation_result = self.test(splits["test"])
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
            self.entity_names,
            dev_metrics=dev_evaluation_result,
        )

    def _preprocess_dataset(
        self, train_dataset: Dataset
    ) -> Tuple[DatasetDict, DatasetDict]:
        # Get entity names from the dataset and create label names
        self.entity_names = self.get_entity_names(train_dataset)
        self.label_names = self.get_label_names(self.entity_names)
        self.label2id, self.id2label = self._get_label_mappings(self.label_names)
        # Convert labels to label ids
        label_ids = []
        for line in train_dataset["labels"]:
            label_ids.append([self.label2id[label] for label in line])
        train_dataset = train_dataset.add_column("label_ids", label_ids)  # type: ignore
        # Define a mapping from beginning (B-) labels to inside (I-) labels
        self.begin2inside = {}
        for label in self.label_names:
            if "B-" in label:
                self.begin2inside[self.label2id[label]] = self.label2id[
                    "I-" + label[2:]
                ]
        # Split dataset into train and test
        splits = self.train_test_split(
            train_dataset,
            self.entity_names,
            test_size=self.dev_fraction,
            seed=self.training_args.seed,
        )
        # Tokenize dataset and align labels
        processed_data = splits.map(
            self.tokenize_fn, batched=True, remove_columns=train_dataset.column_names
        )

        return splits, processed_data

    def test(self, dataset: Dataset) -> ClassificationEvaluationResult:
        """
        Takes in a dataset, that contains test split
        with texts and labels. Runs inference on texts and counts needed metrics.
        Returns a classification_report and a confusion_matrix
        """
        true_labels = dataset["labels"]
        predictions = self.predict(dataset["text"], output_format=NERFormat.BIO)
        pred_labels = []
        for pred in predictions:
            pred_labels.append(pred.labels)
        clf_report, confusion_matrix = evaluate_entity_level(
            true_labels, pred_labels, self.entity_names
        )
        log.info(clf_report)
        log.info(confusion_matrix)
        return ClassificationEvaluationResult(
            confusion_matrix, clf_report, clf_report["macro avg"]["f1-score"]
        )

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
            ort_model = ORTModelForTokenClassification.from_pretrained(
                path_to_package, export=True
            )
            ort_model.save_pretrained(path_to_package)
            os.unlink(path_to_package + "/model.safetensors")
        with open(path_to_package + "/entity_names.json", "w") as f:
            json.dump({"entity_names": self.entity_names, "language": self.language}, f)
        log.info("Model is saved at " + path_to_package)

    @classmethod
    def load(cls, path_to_package: str):
        """Load a trained token classification model from disk.

        Args:
            path_to_package: Directory path containing the saved model.

        Returns:
            Initialized TokenClassificationFinetunerBase with loaded model.
        """
        self = cls(cls.config_cls(model_name_or_path=path_to_package))
        self.model = AutoModelForTokenClassification.from_pretrained(
            path_to_package, return_dict=True
        )
        self.trainer = Trainer(
            self.model,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            args=self.training_args,
        )
        with open(path_to_package + "/entity_names.json") as f:
            entity_data = json.load(f)
        if isinstance(entity_data, list):
            self.entity_names = entity_data
            self.language = "en"
        else:
            self.entity_names = entity_data["entity_names"]
            self.language = entity_data.get("language", "en")
        log.info("Model is loaded and ready to use")
        return self

    def predict(
        self,
        texts: Union[List[str], str],
        batch_size: int = BATCH_SIZE,
        output_format: NERFormat = NERFormat.OFFSETS,
    ) -> List[TokenClassifierOutput]:
        """Run NER inference on texts.

        Args:
            texts: Single text or list of texts to process.
            batch_size: Number of texts per batch.
            output_format: NERFormat.OFFSETS for entity spans or
                NERFormat.BIO for BIO tags.

        Returns:
            List of TokenClassifierOutput with extracted entities.
        """
        return find_executable_batch_size(
            self._predict, starting_batch_size=batch_size
        )(texts, output_format=output_format)

    def _predict(
        self,
        batch_size: int,
        texts: List[str],
        output_format: NERFormat = NERFormat.OFFSETS,
    ) -> List[TokenClassifierOutput]:
        """
        Takes in a collection of texts and runs inference on it.
        Returns a collection of objects of type TokenClassifierOutput.
        """
        ner = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=AggregationStrategy.AVERAGE,
            batch_size=batch_size,
            device=self.device,
        )
        pipeline_outputs = []
        results = ner(texts)
        if output_format == NERFormat.OFFSETS:
            for i in range(len(results)):
                result = results[i]
                labels = []
                for ent in result:
                    labels.append(
                        Entity(
                            label=ent["entity_group"],
                            text=texts[i][ent["start"] : ent["end"]],
                            start=ent["start"],
                            end=ent["end"],
                            score=ent["score"],
                        )
                    )
                pipeline_outputs.append(
                    TokenClassifierOutput(text=texts[i], labels=labels)
                )
        else:
            bio_results = [
                convert_offsets_to_bio(text, entities, label_key="entity_group")[1]
                for text, entities in zip(texts, results)
            ]
            for i in range(len(texts)):
                pipeline_outputs.append(
                    TokenClassifierOutput(text=texts[i], labels=bio_results[i])
                )

        return pipeline_outputs

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
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                id2label=self.id2label,
                label2id=self.label2id,
                **model_kwargs,
            )
            for param in model.parameters():
                param.data = param.data.contiguous()
            return model

        return model_init


class TokenClassificationInferenceManager(AbstractONNXInferenceManager):
    """ONNX-based inference manager for token classification models.

    Uses ONNX Runtime for optimized NER inference.
    """

    def __init__(self, path_to_package: str) -> None:
        """Initialize the inference manager.

        Args:
            path_to_package: Path to directory with saved ONNX model.
        """
        super().__init__(path_to_package)
        self.model = ORTModelForTokenClassification.from_pretrained(
            path_to_package, provider=self.ort_provider
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_package)
        self._pipeline = ort_pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            accelerator="ort",
            aggregation_strategy="average",
        )

    def predict(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        auto_find_batch_size: bool = False,
    ) -> List[Any]:
        """Run NER inference on texts.

        Args:
            texts: List of texts to process.
            batch_size: Number of texts per batch.
            auto_find_batch_size: If True, reduce batch size on OOM.

        Returns:
            List of TokenClassifierOutput with extracted entities.
        """
        if auto_find_batch_size:
            return find_executable_batch_size(
                self._predict, starting_batch_size=batch_size
            )(texts)  # type: ignore
        return self._predict(batch_size, texts)

    def _predict(
        self, batch_size: int, texts: List[str]
    ) -> List[TokenClassifierOutput]:
        results = [result for result in self._pipeline(texts, batch_size=batch_size)]  # type: ignore
        pipeline_outputs = []
        for i in range(len(results)):
            result = results[i]
            labels = []
            for ent in result:
                labels.append(
                    Entity(
                        label=ent["entity_group"],
                        text=texts[i][ent["start"] : ent["end"]],
                        start=ent["start"],
                        end=ent["end"],
                        score=float(ent["score"]),
                    )
                )
            pipeline_outputs.append(TokenClassifierOutput(text=texts[i], labels=labels))
        return pipeline_outputs


class TokenClassificationFinetuner(TokenClassificationFinetunerBase):
    """Token classification finetuner for NER tasks.

    Concrete implementation with ONNX inference support.
    This is the recommended NER finetuning method for production use.
    """

    onnx_inference_manager = TokenClassificationInferenceManager
