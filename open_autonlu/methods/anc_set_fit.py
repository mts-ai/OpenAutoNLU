import logging
import math
from typing import Iterable, Optional

import numpy as np
from accelerate.utils import find_executable_batch_size
from datasets import Dataset
from pandas import DataFrame
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from setfit import SetFitModel
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from transformers import set_seed

from ..constants import (
    DATA_PROCESSING_STAGE,
    PRETRAINING_STAGE,
    TRAINING_STAGE,
)
from ..methods.data_types import TrainingArtifactInfo
from ..progress_reporter import ProgressReporter
from .configs import AncSetFitConfig
from .models.onnx_wrappers import SetFitWrapper
from .set_fit import SetFitMethod
from .utils import fit_encoder

logger = logging.getLogger(__name__)


class AncSetFitDataProcessingMixin:
    """Mixin providing data processing utilities for anchor-based SetFit methods.
    Contains methods for generating triplet training examples and managing
    anchor templates. Used by AncSetFitMethod and AncSetFitOOD.
    """

    @staticmethod
    def _generate_input_triplets(sentences, label_ids, template_dict, input_pairs):
        for positive_idx in range(len(sentences)):
            positive_sentence = sentences[positive_idx]
            current_label = label_ids[positive_idx]
            # get the achor template
            anchor = template_dict[current_label]
            negative_sentence = np.random.choice(
                np.array(sentences)[np.array(label_ids) != current_label]
            )

            input_pairs.append(
                InputExample(texts=[anchor, positive_sentence, negative_sentence])
            )

        return input_pairs

    @staticmethod
    def _create_anchors(template: str, labels: Iterable[str]) -> dict:
        anchors = {}
        for label_id, label in enumerate(labels):
            anchors[label_id] = template + label
        return anchors

    def _resolve_anchor_labels(self, df: DataFrame):
        assert (
            "anc_label" in df.keys()
        ), "You have to provide anchor label names in column anc_label"
        assert hasattr(self, "_label2id"), "Instance should have _label2id assigned"
        anc_map = np.empty(len(self._label2id), dtype=object)  # type: ignore
        for _, row in df.iterrows():
            if row.anc_label not in anc_map:
                anc_map[self._label2id[row.label]] = row.anc_label  # type: ignore
        return anc_map


class AncSetFitMethod(SetFitMethod, AncSetFitDataProcessingMixin):
    """Anchor-based SetFit method for few-shot text classification.

    Extends SetFit with triplet loss training using anchor templates. Each class
    is represented by an anchor (template + human-readable class description),
    and the model learns to place positive examples close to their anchors while
    pushing negative examples away.

    Training pipeline:
        1. Data processing: resolve anchor labels from the dataset
        2. Pretraining: contrastive learning with triplet loss (anchor, positive, negative)
        3. Training: fit a logistic regression classifier on encoded embeddings

    Requires the dataset to contain an 'anc_label' column with human-readable
    class descriptions. If absent, use SetFitMethod instead.

    Attributes:
        config_cls: Configuration class (AncSetFitConfig).
        stages: Training stages for progress reporting.
        margin: Triplet loss margin.
        template: Prompt template for anchor construction.
        anc_map: Mapping from label IDs to anchor descriptions.

    Example:
        >>> config = AncSetFitConfig(template="This text is about: ")
        >>> method = AncSetFitMethod(config)
        >>> artifact = method.train(dataset)  # dataset must have 'anc_label' column
        >>> predictions = method.predict(["new text"])
    """

    config_cls = AncSetFitConfig
    stages = [DATA_PROCESSING_STAGE, PRETRAINING_STAGE, TRAINING_STAGE]

    def __init__(
        self,
        config: Optional[AncSetFitConfig] = None,
        progress_reporter: Optional[ProgressReporter] = None,
    ) -> None:
        if config is None:
            config = AncSetFitConfig()
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
        set_seed(config.seed)
        if config.add_normalization_layer:
            self.encoder._modules["2"] = models.Normalize()

    def train(self, data: Dataset):
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

        self.anc_map = self._resolve_anchor_labels(data.to_pandas())  # type: ignore
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
        fit_with_auto_batch_size(train_examples=train_examples, train_loss=train_loss)

        # Train the final classifier
        self.progress_reporter(TRAINING_STAGE, 0.0)
        self.classifier = LogisticRegression()
        x_train_encoded = self.encoder.encode(x_train)
        self.classifier.fit(x_train_encoded, data["label"])
        self.model = SetFitModel(model_body=self.encoder, model_head=self.classifier)

        # Model swapping
        self.base_setfit_instance = self.model
        model_pooler = self.model.model_body._modules["1"]  # type: ignore

        self.model = SetFitWrapper(
            self.model.model_body._modules["0"].auto_model,  # type: ignore
            model_pooler,
            "2" in self.model.model_body._modules,  # type: ignore
            self.model.model_head,  # type: ignore
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
        )
