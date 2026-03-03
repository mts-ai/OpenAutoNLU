from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from setfit.exporters.utils import mean_pooling
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from ...methods.constants import OOS_LABEL
from ...methods.models.mahalanobis import MarginalMahalanobisDetector
from ...methods.models.msp import MaxSoftmaxProbabilityDetector

import logging

log = logging.getLogger(__name__)


@dataclass
class OOSSequenceClassifierConfig:
    id2label: dict
    label2id: dict

    def _add_oos_label_to_labels_and_ids(self):
        oos_label_id = max(self.id2label.keys()) + 1
        self.id2label = {**self.id2label, oos_label_id: OOS_LABEL}
        self.label2id = {**self.label2id, OOS_LABEL: oos_label_id}


class SetFitOODWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        pooler: nn.Module,
        normalization: bool,
        indomain_clf: LogisticRegression,
        oos_classifier: Union[
            MarginalMahalanobisDetector, MaxSoftmaxProbabilityDetector
        ],
    ):
        super().__init__()
        self.encoder = encoder
        self.do_norm = normalization
        self.indomain_classifier = self.linear_from_logreg(indomain_clf).to(
            self.encoder.device
        )
        self.oos_classifier = oos_classifier.to(self.encoder.device)
        if pooler is None:
            log.warning("No pooler was set so defaulting to mean pooling.")
            self.pooler = mean_pooling
        else:
            self.pooler = pooler
        id2label = {idx: label for idx, label in enumerate(indomain_clf.classes_)}
        label2id = {label: idx for idx, label in enumerate(indomain_clf.classes_)}
        self.config = OOSSequenceClassifierConfig(id2label=id2label, label2id=label2id)
        self.config._add_oos_label_to_labels_and_ids()

    @staticmethod
    def linear_from_logreg(model: LogisticRegression) -> nn.Linear:
        weights = torch.from_numpy(model.coef_).to(torch.float32)
        biases = torch.from_numpy(np.array(model.intercept_)).to(torch.float32)
        linear = nn.Linear(weights.shape[1], weights.shape[0], bias=True)
        linear.weight = nn.Parameter(weights, requires_grad=False)
        linear.bias = nn.Parameter(biases, requires_grad=False)
        return linear

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = {
            "token_embeddings": encoder_output[0],
            "attention_mask": attention_mask,
        }
        if self.pooler is mean_pooling:
            embeddings = self.pooler(**hidden_states)
        else:
            embeddings = self.pooler(hidden_states)["sentence_embedding"]

        if self.do_norm:
            embeddings = F.normalize(embeddings, dim=-1)
        if self.indomain_classifier.out_features != 1:
            indomain_logits = F.softmax(self.indomain_classifier(embeddings), dim=-1)
        else:
            class_1_scores = F.sigmoid(self.indomain_classifier(embeddings))
            class_0_scores = 1 - class_1_scores
            indomain_logits = torch.hstack([class_0_scores, class_1_scores])
        if isinstance(self.oos_classifier, MaxSoftmaxProbabilityDetector):
            oos_input = indomain_logits
        elif isinstance(self.oos_classifier, MarginalMahalanobisDetector):
            oos_input = embeddings
        oos_scores = self.oos_classifier(oos_input).unsqueeze(dim=1)
        logits = torch.hstack([indomain_logits, oos_scores])
        return logits


class SetFitWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        pooler: nn.Module,
        normalization: bool,
        indomain_clf: LogisticRegression,
    ):
        super().__init__()
        self.encoder = encoder
        self.do_norm = normalization
        self.indomain_classifier = self.linear_from_logreg(indomain_clf).to(
            self.encoder.device
        )
        if pooler is None:
            log.warning("No pooler was set so defaulting to mean pooling.")
            self.pooler = mean_pooling
        else:
            self.pooler = pooler
        id2label = {idx: label for idx, label in enumerate(indomain_clf.classes_)}
        label2id = {label: idx for idx, label in enumerate(indomain_clf.classes_)}
        self.config = OOSSequenceClassifierConfig(id2label=id2label, label2id=label2id)

    @staticmethod
    def linear_from_logreg(model: LogisticRegression) -> nn.Linear:
        weights = torch.from_numpy(model.coef_).to(torch.float32)
        biases = torch.from_numpy(np.array(model.intercept_)).to(torch.float32)
        linear = nn.Linear(weights.shape[1], weights.shape[0], bias=True)
        linear.weight = nn.Parameter(weights, requires_grad=False)
        linear.bias = nn.Parameter(biases, requires_grad=False)
        return linear

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = {
            "token_embeddings": encoder_output[0],
            "attention_mask": attention_mask,
        }
        if self.pooler is mean_pooling:
            embeddings = self.pooler(**hidden_states)
        else:
            embeddings = self.pooler(hidden_states)["sentence_embedding"]

        if self.do_norm:
            embeddings = F.normalize(embeddings, dim=-1)
        if self.indomain_classifier.out_features != 1:
            logits = F.softmax(self.indomain_classifier(embeddings), dim=-1)
        else:
            class_1_scores = F.sigmoid(self.indomain_classifier(embeddings))
            class_0_scores = 1 - class_1_scores
            logits = torch.hstack([class_0_scores, class_1_scores])
        return logits


class FinetunerOODWrapper(nn.Module):
    def __init__(
        self,
        inscope_classifier: Optional[torch.nn.Module] = None,
        detector: Optional[
            Union[MarginalMahalanobisDetector, MaxSoftmaxProbabilityDetector]
        ] = None,
        classifier_config: Optional[PretrainedConfig] = None,
        enable_softmax: Optional[bool] = True,
    ) -> None:
        super().__init__()
        if inscope_classifier is not None and detector is not None:
            self.inscope_classifier = inscope_classifier
            self.oos_classifier = detector
            self.config = OOSSequenceClassifierConfig(
                id2label=inscope_classifier.config.id2label,
                label2id=inscope_classifier.config.label2id,
            )
        elif classifier_config is not None:
            self.inscope_classifier = AutoModelForSequenceClassification.from_config(
                classifier_config
            )
            if classifier_config.oos_classifier_cls == "MarginalMahalanobisDetector":
                self.oos_classifier = MarginalMahalanobisDetector(
                    np.zeros(classifier_config.hidden_size, dtype=np.float32),
                    np.zeros(
                        (classifier_config.hidden_size, classifier_config.hidden_size),
                        dtype=np.float32,
                    ),
                    threshold=1.0,
                )
            elif (
                classifier_config.oos_classifier_cls == "MaxSoftmaxProbabilityDetector"
            ):
                self.oos_classifier = MaxSoftmaxProbabilityDetector(threshold=1.0)

            self.config = OOSSequenceClassifierConfig(
                id2label=classifier_config.id2label,
                label2id=classifier_config.label2id,
            )
        else:
            raise ValueError(
                "Either inscope_classifier and detector or classifier_config must be provided"
            )
        self.enable_softmax = enable_softmax
        self.config._add_oos_label_to_labels_and_ids()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        SequenceClassifierOutput,
    ]:
        classifier_output = self.inscope_classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # type: ignore
        if isinstance(self.oos_classifier, MaxSoftmaxProbabilityDetector):
            oos_input = classifier_output.logits
        elif isinstance(self.oos_classifier, MarginalMahalanobisDetector):
            oos_input = classifier_output.pooled_output
        oos_output = self.oos_classifier(oos_input).unsqueeze(dim=1)
        if self.enable_softmax:
            logits = torch.nn.functional.softmax(classifier_output.logits, dim=-1)
        else:
            logits = classifier_output.logits
        logits = torch.hstack((logits, oos_output))

        loss = None
        if labels is not None:
            loss = torch.Tensor([0]).to(
                logits.device
            )  # for transformers.Trainer compatibility

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=classifier_output.hidden_states,
            attentions=classifier_output.attentions,
        )
