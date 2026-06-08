import json
import logging
import os
from collections import Counter
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import safetensors.torch
import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoConfig, DataCollatorWithPadding, Trainer, set_seed
from transformers.trainer_callback import EarlyStoppingCallback

from ..constants import (
    DATA_PROCESSING_STAGE,
    DEV_EVALUATION_STAGE,
    HPO_STAGE,
    TRAINING_STAGE,
)
from ..methods.constants import OOS_LABEL
from ..progress_reporter import (
    ProgressReporter,
    TrainerProgressReporterCallback,
)
from .configs import FinetunerOODConfig
from .data_types import (
    ClassificationEvaluationResult,
    SaveFormat,
    TrainingArtifactInfo,
)
from .finetuner import FinetunerBase
from .models.classifiers_with_pooled_outputs import (
    remap_auto_classes,
)
from .models.mahalanobis import MarginalMahalanobisTrainer
from .models.msp import MaxSoftmaxProbabilityTrainer
from .models.onnx_wrappers import FinetunerOODWrapper
from .onnx_inference_manager import (
    GeneralSequenceClassifierInferenceManager,
)
from .utils import GenericSequenceClassifierONNXExportMixin

logging.basicConfig()
log = logging.getLogger(__name__)


class FinetunerWithOOD(FinetunerBase, GenericSequenceClassifierONNXExportMixin):
    """Finetuner with out-of-distribution (OOD) detection.

    Extends the standard Finetuner with OOD detection capabilities
    using either Mahalanobis distance or Maximum Softmax Probability.
    Samples classified as OOD are labeled as "outOfScope".

    Attributes:
        onnx_inference_manager: Inference manager class for ONNX export.
        config_cls: Configuration class for this method.
        stages: Training stages for progress reporting.
    """

    onnx_inference_manager = GeneralSequenceClassifierInferenceManager
    config_cls = FinetunerOODConfig
    stages = [DATA_PROCESSING_STAGE, HPO_STAGE, TRAINING_STAGE, DEV_EVALUATION_STAGE]

    def __init__(
        self,
        config: Optional[FinetunerOODConfig] = None,
        progress_reporter: Optional[ProgressReporter] = None,
    ):
        """Initialize Finetuner with OOD detection.

        Args:
            config: Configuration with OOD detection parameters.
            progress_reporter: Reporter for tracking training progress.
        """
        if config is None:
            config = FinetunerOODConfig()
        if progress_reporter is None:
            progress_reporter = ProgressReporter()
        super().__init__(config)
        self.threshold_factor = config.threshold_factor
        self.ood_trainer_cls = globals()[config.ood_method.value]
        self.progress_reporter = progress_reporter
        remap_auto_classes()

    @staticmethod
    def _postprocess_logits(logits: np.ndarray) -> np.ndarray:
        """
        This method is overloaded to skip the softmax
        application as it is done in the model itself.
        """
        return logits

    def _generate_synthetic_oos(self, num_train_rows: int) -> Dataset:
        from ..plugins.ood_sampling import resolve_ood_sampler

        sampler = resolve_ood_sampler(self.ood_sampler, self.training_args.seed)
        synthetic_texts = sampler.sample(num_train_rows)
        synthetic_dataset = Dataset.from_dict(
            {"text": synthetic_texts, "label": [OOS_LABEL] * num_train_rows}
        )
        return synthetic_dataset

    def train(self, train_dataset: Dataset) -> TrainingArtifactInfo:
        set_seed(self.seed)
        self.progress_reporter.stages = self.stages
        self.progress_reporter(DATA_PROCESSING_STAGE, 0.0)
        # Data preprocessing
        label_counts = Counter(train_dataset["label"])
        # OOD augmentation
        ood_count = label_counts.get(OOS_LABEL, 0)
        if ood_count == 0:
            synthetic_texts_ds = self._generate_synthetic_oos(train_dataset.num_rows)
            train_dataset = concatenate_datasets([train_dataset, synthetic_texts_ds])
        elif ood_count < train_dataset.num_rows:
            synthetic_texts_ds = self._generate_synthetic_oos(
                train_dataset.num_rows - ood_count
            )
            train_dataset = concatenate_datasets([train_dataset, synthetic_texts_ds])
        labels = train_dataset.unique("label")
        if OOS_LABEL in labels:
            labels_for_data = [label for label in labels if label != OOS_LABEL] + [
                OOS_LABEL
            ]
        else:
            labels_for_data = labels
        data_label2id, data_id2label = self._get_label_mappings(labels_for_data)

        inscope_labels = [label for label in labels if label != OOS_LABEL]
        self.num_labels = len(inscope_labels)
        self.label2id, self._id2label = self._get_label_mappings(inscope_labels)

        original_label2id = self.label2id
        self.label2id = data_label2id
        data = self._split_data(train_dataset).map(
            self.preprocess_function,
            remove_columns=train_dataset.column_names,
            batched=True,
        )
        self.label2id = original_label2id

        inscope_data = data.filter(
            lambda example: example["labels"] != data_label2id.get(OOS_LABEL)
        )
        oos_data = data.filter(
            lambda example: example["labels"] == data_label2id.get(OOS_LABEL)
        )

        self.progress_reporter(DATA_PROCESSING_STAGE, 1.0)
        # Training set-up
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.training_args.configure_for_early_stopping()
        self.trainer = Trainer(
            model_init=self._prepare_model_init_function(),
            args=self.training_args,
            train_dataset=inscope_data["train"],
            eval_dataset=inscope_data["eval"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_train,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        # HPO
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
                train_dataset=inscope_data["train"],
                eval_dataset=inscope_data["eval"],
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

        # In-scope training
        self.trainer.train()

        # Out-of-Scope training
        oos_trainer = self.ood_trainer_cls(self.training_args.seed)
        features_only = inscope_data["train"].remove_columns("labels")
        oos_features_only = oos_data["train"].remove_columns("labels")

        if isinstance(oos_trainer, MarginalMahalanobisTrainer):
            oos_embeddings = self.trainer.predict(oos_features_only).predictions[1]  # type: ignore
            inscope_embeddings = self.trainer.predict(features_only).predictions[1]  # type: ignore
            oos_detector = oos_trainer.train(
                inscope_embeddings,
                oos_embeddings=oos_embeddings,
                threshold_magnifier=self.threshold_factor,
            ).to(self.trainer.model.device)
        elif isinstance(oos_trainer, MaxSoftmaxProbabilityTrainer):
            oos_scores = self.trainer.predict(oos_features_only).predictions[0]  # type: ignore
            inscope_scores = self.trainer.predict(features_only).predictions[0]  # type: ignore
            oos_detector = oos_trainer.train(
                inscope_logits=inscope_scores,
                oos_logits=oos_scores,
            ).to(self.trainer.model.device)
        self.classifier_config = self.trainer.model.config
        # Model swapping
        self.model = FinetunerOODWrapper(
            self.trainer.model, oos_detector, enable_softmax=False
        ).eval()
        self.trainer.model = self.model
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        # Evaluation
        self.trainer.compute_metrics = partial(
            self.compute_metrics_eval, id2label=self.id2label
        )
        self.progress_reporter(DEV_EVALUATION_STAGE, 0.0)
        metrics: Dict[str, Any] = self.trainer.evaluate(
            eval_dataset=data["eval"], metric_key_prefix=""
        )
        self.progress_reporter(DEV_EVALUATION_STAGE, 1.0)
        self.model.enable_softmax = True
        return TrainingArtifactInfo(
            self.training_args.to_dict(),
            self.label_list,
            ClassificationEvaluationResult(
                classification_report=metrics["_classification_report"],
                confusion_matrix=metrics["_confusion_matrix"],
                f1=metrics["_f1"],
            ),
        )

    def save(
        self, path_to_package: str, checkpoint_format: SaveFormat = SaveFormat.TORCH
    ) -> None:
        self.tokenizer.save_pretrained(path_to_package)
        if checkpoint_format == SaveFormat.TORCH:
            torch.save(
                self.training_args, os.path.join(path_to_package, "training_args.bin")
            )
            config = self.classifier_config.to_dict()
            config["oos_classifier_cls"] = self.model.oos_classifier.__class__.__name__
            with open(os.path.join(path_to_package, "config.json"), "w") as f:
                json.dump(config, f)
            for parameter in self.model.inscope_classifier.parameters():
                parameter.data = parameter.data.contiguous()
            safetensors.torch.save_file(
                self.model.state_dict(),
                os.path.join(path_to_package, "model.safetensors"),
            )
        elif checkpoint_format == SaveFormat.ONNX:
            # Ensuring softmax is on to get the correct ONNX model
            self.model.enable_softmax = True
            self.export_onnx(path_to_package)
        log.info("Model is saved at " + path_to_package)

    @classmethod
    def load(cls, path_to_package: str):
        remap_auto_classes()
        self = cls(cls.config_cls(model_name_or_path=path_to_package))
        config = AutoConfig.from_pretrained(path_to_package)
        self.model = FinetunerOODWrapper(classifier_config=config, enable_softmax=True)
        safetensors.torch.load_model(
            self.model, os.path.join(path_to_package, "model.safetensors")
        )
        self.model.eval()
        self.trainer = Trainer(
            self.model,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            args=self.training_args,
        )
        log.info("Model is loaded and ready to use")
        return self
