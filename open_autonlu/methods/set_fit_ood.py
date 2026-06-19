import logging
from collections import Counter
from dataclasses import replace
from typing import List, Optional

import joblib
import numpy as np
import torch
from accelerate.utils import find_executable_batch_size
from datasets import Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers.integrations import ClearMLCallback

from ..constants import PRETRAINING_STAGE, TRAINING_STAGE
from ..methods.constants import OOS_LABEL
from ..progress_reporter import (
    ProgressReporter,
)
from .configs import SetFitOODConfig
from .data_types import (
    SaveFormat,
    TextClassifierOutput,
    TextLabel,
    TrainingArtifactInfo,
)
from .models.mahalanobis import MarginalMahalanobisDetector, MarginalMahalanobisTrainer
from .models.msp import MaxSoftmaxProbabilityDetector, MaxSoftmaxProbabilityTrainer
from .models.onnx_wrappers import SetFitOODWrapper
from .onnx_inference_manager import (
    GeneralSequenceClassifierInferenceManager,
)
from .set_fit import SetFitMethodBase

log = logging.getLogger(__name__)


class SetFitOOD(SetFitMethodBase):
    """SetFit method with out-of-distribution (OOD) detection.

    Attributes:
        onnx_inference_manager: Inference manager class for ONNX export.
        config_cls: Configuration class for this method.
        stages: Training stages for progress reporting.
    """

    onnx_inference_manager = GeneralSequenceClassifierInferenceManager
    config_cls = SetFitOODConfig
    stages = [PRETRAINING_STAGE, TRAINING_STAGE]

    def __init__(
        self,
        config: Optional[SetFitOODConfig] = None,
        progress_reporter: Optional[ProgressReporter] = None,
    ) -> None:
        """Initialize SetFit with OOD detection.

        Args:
            config: Configuration with OOD detection parameters.
            progress_reporter: Reporter for tracking training progress.
        """
        if config is None:
            config = SetFitOODConfig()
        super().__init__(config)
        self.threshold_factor = config.threshold_factor
        self.ood_trainer_cls = globals()[config.ood_method.value]
        if progress_reporter is None:
            progress_reporter = ProgressReporter()
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

    def _generate_synthetic_oos(self, num_train_rows: int) -> Dataset:
        from ..plugins.ood_sampling import resolve_ood_sampler

        sampler = resolve_ood_sampler(self.ood_sampler, self.seed)
        synthetic_texts = sampler.sample(num_train_rows)
        synthetic_dataset = Dataset.from_dict(
            {"text": synthetic_texts, "label": [OOS_LABEL] * num_train_rows}
        )
        return synthetic_dataset

    def train(self, dataset: Dataset) -> TrainingArtifactInfo:
        self.progress_reporter.stages = self.stages
        self.progress_reporter(PRETRAINING_STAGE, 0.0)
        label_counts = Counter(dataset["label"])
        # OOD augmentation
        ood_count = label_counts.get(OOS_LABEL, 0)
        if ood_count == 0:
            synthetic_texts_ds = self._generate_synthetic_oos(dataset.num_rows)
            dataset = concatenate_datasets([dataset, synthetic_texts_ds])
        elif ood_count < dataset.num_rows:
            synthetic_texts_ds = self._generate_synthetic_oos(
                dataset.num_rows - ood_count
            )
            dataset = concatenate_datasets([dataset, synthetic_texts_ds])

        # Use OOS_LABEL as a label in trainer or as OOD head?
        inscope_data = dataset.filter(lambda example: example["label"] != OOS_LABEL)
        oos_data = dataset.filter(lambda example: example["label"] == OOS_LABEL)

        fit_with_auto_batch_size = find_executable_batch_size(
            self._fit_with_auto_batch_size, starting_batch_size=self.batch_size
        )
        trainer = fit_with_auto_batch_size(dataset=inscope_data)
        self.progress_reporter(PRETRAINING_STAGE, 1.0)

        # Out-of-Scope training
        oos_trainer = self.ood_trainer_cls(self.seed)
        if isinstance(oos_trainer, MarginalMahalanobisTrainer):
            inscope_embeddings = self.model.encode(inscope_data["text"]).astype(
                np.float64
            )
            oos_embeddings = self.model.encode(oos_data["text"]).astype(np.float64)
            self.oos_detector = oos_trainer.train(
                inscope_embeddings,
                oos_embeddings=oos_embeddings,
                threshold_magnifier=self.threshold_factor,
            ).to(self.model.device)
        elif isinstance(oos_trainer, MaxSoftmaxProbabilityTrainer):
            inscope_scores = self.model.predict_proba(inscope_data["text"])
            oos_scores = self.model.predict_proba(oos_data["text"])
            self.oos_detector = oos_trainer.train(inscope_scores, oos_scores).to(
                self.model.device
            )
        self.progress_reporter(TRAINING_STAGE, 1.0)
        # Model swapping
        self.base_setfit_instance = self.model
        model_pooler = self.model.model_body._modules["1"]

        self.model = SetFitOODWrapper(
            self.model.model_body._modules["0"].auto_model,
            model_pooler,
            "2" in self.model.model_body._modules,
            self.model.model_head,
            self.oos_detector,
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
        if self.num_iterations is not None:
            trainer.args = replace(trainer.args, num_iterations=self.num_iterations)
        trainer.train()
        return trainer

    def _predict(self, batch_size: int, texts: List[str]) -> list[TextClassifierOutput]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.base_setfit_instance.encode(texts, batch_size=batch_size)
        inscope_probs = self.base_setfit_instance.model_head.predict_proba(embeddings)
        if isinstance(self.model.oos_classifier, MarginalMahalanobisDetector):
            oos_input = embeddings
        elif isinstance(self.model.oos_classifier, MaxSoftmaxProbabilityDetector):
            oos_input = inscope_probs
        oos_scores = (
            self.model.oos_classifier(
                torch.from_numpy(oos_input)
                .to(torch.float32)
                .to(self.base_setfit_instance.device)
            )
            .unsqueeze(1)
            .cpu()
            .numpy()
        )
        logits = np.hstack((inscope_probs, oos_scores))
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
        # TODO: add config saving
        if checkpoint_format == SaveFormat.TORCH:
            self.base_setfit_instance.save_pretrained(path_to_package)
            torch.save(self.oos_detector, path_to_package + "/oos_detector.pt")
        else:
            self.tokenizer = self.base_setfit_instance.model_body.tokenizer
            self.export_onnx(path_to_package)
            self.tokenizer.save_pretrained(path_to_package)
        log.info("Model is saved at " + path_to_package)

    @classmethod
    def load(cls, path_to_package: str):
        self = cls()  # TODO: add config loading
        model_body = SentenceTransformer(path_to_package)
        model_head = joblib.load(path_to_package + "/model_head.pkl")
        self.base_setfit_instance = SetFitModel(
            model_body=model_body, model_head=model_head
        )
        with open(path_to_package + "/oos_detector.pt", "rb") as f:
            self.oos_detector = torch.load(
                f, map_location=model_body.device, weights_only=False
            )
        self.model = SetFitOODWrapper(
            model_body._modules["0"].auto_model,
            model_body._modules["1"],
            "2" in model_body._modules,
            model_head,
            self.oos_detector,
        ).eval()
        return self
