import math
from typing import Optional

import numpy as np
from accelerate.utils import find_executable_batch_size
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, models
from setfit import SetFitModel
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from transformers import set_seed

from ..constants import (
    DATA_PROCESSING_STAGE,
    PRETRAINING_STAGE,
    TRAINING_STAGE,
)
from ..progress_reporter import ProgressReporter
from .anc_set_fit import AncSetFitDataProcessingMixin
from .configs import AncSetFitOODConfig
from .data_types import TrainingArtifactInfo
from .models.mahalanobis import MarginalMahalanobisTrainer
from .models.msp import MaxSoftmaxProbabilityTrainer
from .models.onnx_wrappers import SetFitOODWrapper
from .onnx_inference_manager import GeneralSequenceClassifierInferenceManager
from .set_fit_ood import SetFitOOD
from .utils import fit_encoder


class AncSetFitOOD(SetFitOOD, AncSetFitDataProcessingMixin):
    """Anchor-based SetFit with Out-of-Distribution detection.
    Combines AncSetFit's anchor-based contrastive learning with OOD detection

    Training pipeline:
        1. Data processing: resolve anchor labels from the dataset
        2. Pretraining: contrastive learning with triplet loss
        3. Training: fit logistic regression classifier + OOD detector

    Attributes:
        onnx_inference_manager: Manager class for ONNX inference.
        config_cls: Configuration class (AncSetFitOODConfig).
        stages: Training stages for progress reporting.
        oos_detector: Trained OOD detection model.
        threshold_factor: Multiplier for OOD detection threshold.
    """

    onnx_inference_manager = GeneralSequenceClassifierInferenceManager
    config_cls = AncSetFitOODConfig
    stages = [DATA_PROCESSING_STAGE, PRETRAINING_STAGE, TRAINING_STAGE]

    def __init__(
        self,
        config: Optional[AncSetFitOODConfig] = None,
        progress_reporter: Optional[ProgressReporter] = None,
    ) -> None:
        if config is None:
            config = AncSetFitOODConfig()
        if progress_reporter is None:
            progress_reporter = ProgressReporter()
        self.progress_reporter = progress_reporter
        self.margin = config.margin
        self.max_seq_length = config.max_seq_length
        self.num_iterations = config.num_iterations
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.encoder = SentenceTransformer(config.model_name_or_path)
        self.encoder.max_seq_length = config.max_seq_length
        self.encoder.tokenizer.model_max_length = config.max_seq_length
        self.body_lr = config.body_lr
        self.template = config.template
        self.seed = config.seed
        set_seed(self.seed)
        if config.add_normalization_layer:
            self.encoder._modules["2"] = models.Normalize()
        self.threshold_factor = config.threshold_factor
        self.ood_trainer_cls = globals()[config.ood_method.value]

    def train(self, data: Dataset) -> TrainingArtifactInfo:
        self.progress_reporter.stages = self.stages
        self.progress_reporter(DATA_PROCESSING_STAGE, 0.0)
        self._label2id, self._id2label = self._get_label_mappings(data.unique("label"))
        x_train = data["text"]
        y_train = [self._label2id[label] for label in data["label"]]

        # Add TripetLoss
        train_loss = losses.TripletLoss(
            model=self.encoder,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=self.margin,
        )
        train_examples = []

        self.anc_map = self._resolve_anchor_labels(data.to_pandas())
        for _ in range(self.num_iterations):
            # changes how to generate input pairs to fit achor
            dict_templates = self._create_anchors(self.template, self.anc_map)
            train_examples = self._generate_input_triplets(
                np.array(x_train), y_train, dict_templates, train_examples
            )

        fit_with_auto_batch_size = find_executable_batch_size(
            self._fit_with_auto_batch_size, starting_batch_size=self.batch_size
        )
        self.progress_reporter(PRETRAINING_STAGE, 0.0)
        fit_with_auto_batch_size(train_examples=train_examples, train_loss=train_loss)  # type: ignore
        self.progress_reporter(TRAINING_STAGE, 0.0)
        # Train the inscope classifier

        self.classifier = LogisticRegression()
        x_train_encoded = self.encoder.encode(x_train)
        self.classifier.fit(x_train_encoded, data["label"])
        self.model = SetFitModel(model_body=self.encoder, model_head=self.classifier)

        # Train the oos classifier
        oos_trainer = self.ood_trainer_cls()
        if isinstance(oos_trainer, MarginalMahalanobisTrainer):
            self.oos_detector = oos_trainer.train(
                x_train_encoded.astype(np.float64),
                threshold_magnifier=self.threshold_factor,
            ).to(self.model.device)
        elif isinstance(oos_trainer, MaxSoftmaxProbabilityTrainer):
            inscope_scores = self.model.predict_proba(x_train)
            synthetic_texts_ds = self._generate_synthetic_oos(len(x_train))
            oos_scores = self.model.predict_proba(synthetic_texts_ds["text"])
            self.oos_detector = oos_trainer.train(inscope_scores, oos_scores).to(
                self.model.device
            )

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
        hyper_params = {
            "margin": self.margin,
            "max_seq_length": self.max_seq_length,
            "num_iterations": self.num_iterations,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.body_lr,
        }
        self.progress_reporter(TRAINING_STAGE, 1.0)
        return TrainingArtifactInfo(hyper_params, self.label_list)

    def _fit_with_auto_batch_size(
        self,
        batch_size,
        train_examples,
        train_loss,
    ):
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,  # type: ignore
        )
        train_steps = len(train_dataloader)

        warmup_steps = math.ceil(train_steps * 0.1)
        self.encoder = fit_encoder(
            encoder=self.encoder,
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            steps_per_epoch=train_steps,
            warmup_steps=warmup_steps,
            show_progress_bar=False,
            optimizer_params={"lr": self.body_lr},
            progress_reporter=self.progress_reporter,
        )
