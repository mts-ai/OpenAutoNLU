import gc
import logging
import types
from abc import ABC, abstractmethod
from typing import Optional

import torch
from datasets import Dataset
from torch import nn
from transformers import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    set_seed,
)

from ..constants import DQ_TRAIN
from ..methods import Finetuner, TokenClassificationFinetuner
from ..progress_reporter import (
    ProgressReporter,
    TrainerProgressReporterCallback,
)
from .dynamic_trainer import DynamicTrainer

log = logging.getLogger(__name__)


class NotEnoughDataException(Exception):
    pass


class AbstractDynamicTuner(ABC):
    @abstractmethod
    def train(
        self, train_dataset: Dataset, progress_reporter: Optional[ProgressReporter]
    ):
        raise NotImplementedError()

    @abstractmethod
    def mc_dropout_regime_on(self):
        raise NotImplementedError()

    @abstractmethod
    def mc_dropout_regime_off(self):
        raise NotImplementedError()

    @abstractmethod
    def mc_droupout(self, dataset: Dataset, num_repetitions: int = 15):
        raise NotImplementedError()


class DynamicTuner(AbstractDynamicTuner):
    def mc_dropout_regime_on(self):
        """
        This decision was made not to copy
        a bunch of code from transformers and replace
        one line in trainer.model.eval().
        """

        def eval_func(self, mode: bool = True):
            def apply_dropout(module):
                if type(module) is nn.Dropout:
                    module.train()

            self.apply(apply_dropout)

        log.info("MC dropout regime on")

        self.true_eval_func = self.trainer.model.eval()
        setattr(
            self.trainer.model, "eval", types.MethodType(eval_func, self.trainer.model)
        )

    def mc_dropout_regime_off(self):
        if self.true_eval_func is None:
            raise ValueError("MC dropout regime is not on")
        setattr(
            self.trainer.model,
            "eval",
            types.MethodType(self.true_eval_func, self.trainer.model),
        )

        log.info("MC dropout regime off")

    def mc_droupout(self, dataset: Dataset, num_repetitions: int = 15):
        dataset = dataset.remove_columns("id")

        label2id = {v: k for k, v in enumerate(self.label_names)}

        label_ids = []
        for line in dataset["labels"]:
            label_ids.append([label2id[label] for label in line])
        dataset = dataset.add_column("label_ids", label_ids)
        # Define a mapping from beginning (B-) labels to inside (I-) labels

        remove_columns = dataset.column_names
        # remove_columns.remove('id')
        dataset = dataset.map(
            self.tokenize_fn, batched=True, remove_columns=remove_columns
        )
        self.mc_dropout_regime_on()
        raw_data = []
        labels = []
        log.info("Starting MC dropout predictions...")
        for _ in range(num_repetitions):
            res = self.trainer.predict(dataset)
            raw_data.append(res.predictions)
            labels = res.label_ids

        self.mc_dropout_regime_off()

        return raw_data, labels


class SeqClassDynamicTuner(DynamicTuner, Finetuner):
    parent_finetuner_cls_name = Finetuner.__name__
    config_cls = Finetuner.config_cls

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_args.output_dir = "./models"
        self.training_args.logging_strategy = "epoch"
        self.training_args.save_strategy = "no"
        self.num_hpo_trials = 10

    def train(
        self,
        train_dataset: Dataset,
        progress_reporter: Optional[ProgressReporter] = None,
    ):
        if progress_reporter is None:
            progress_reporter = ProgressReporter()
        set_seed(self.training_args.seed)
        labels = train_dataset.unique("label")
        self.num_labels = len(labels)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.label2id, self.id2label = self._get_label_mappings(labels)

        tokenized_train = train_dataset.map(
            self.preprocess_function,
            remove_columns=["text", "label"],
            batched=True,
        )

        self.trainer = DynamicTrainer(
            model_init=self._prepare_model_init_function(output_hidden_states=True),
            args=self.training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            callbacks=[TrainerProgressReporterCallback(progress_reporter, DQ_TRAIN)],
        )

        self.trainer.train()
        self.model = self.trainer.model

        gc.collect()
        torch.cuda.empty_cache()


class TokenClassDynamicTuner(DynamicTuner, TokenClassificationFinetuner):
    parent_finetuner_cls_name = TokenClassificationFinetuner.__name__
    config_cls = TokenClassificationFinetuner.config_cls

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_args.output_dir = "./models"
        self.training_args.logging_strategy = "epoch"
        self.training_args.save_strategy = "no"
        self.true_eval_func = None
        self.training_args.num_train_epochs = 5
        self.num_hpo_trials = 10

    def align_target(self, labels, word_ids):
        # Initialize an empty list to store aligned labels and a variable to track the last word
        align_labels = []
        last_word = None

        # Iterate through the word_ids
        for word in word_ids:
            if word is None:
                label = -100  # Set label to -100 for None word_ids
            elif word != last_word:
                label = labels[
                    word
                ]  # Use the label corresponding to the current word_id
            else:
                label = -100
                # Change B- to I- if the previous word is the same
                if label in self.begin2inside:
                    label = self.begin2inside[label]  # Map B- to I-

            # Append the label to the align_labels list and update last_word
            align_labels.append(label)
            last_word = word

        return align_labels

    def train(
        self,
        train_dataset: Dataset,
        progress_reporter: Optional[ProgressReporter] = None,
    ):
        if progress_reporter is None:
            progress_reporter = ProgressReporter()
        set_seed(self.training_args.seed)
        entity_names = self.get_entity_names(train_dataset)
        self.label_names = self.get_label_names(entity_names)
        self.label2id, self.id2label = self._get_label_mappings(self.label_names)

        label_ids = []
        for line in train_dataset["labels"]:
            label_ids.append([self.label2id[label] for label in line])
        train_dataset = train_dataset.add_column("label_ids", label_ids)
        # Define a mapping from beginning (B-) labels to inside (I-) labels
        self.begin2inside = {}
        for label in self.label_names:
            if "B-" in label:
                self.begin2inside[self.label2id[label]] = self.label2id[
                    "I-" + label[2:]
                ]

        remove_columns = train_dataset.column_names
        # =====injection=====
        remove_columns.remove("id")
        # =====/injection=====

        tokenized_train = train_dataset.map(
            self.tokenize_fn, batched=True, remove_columns=remove_columns
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, padding=True
        )

        tokenized_train = tokenized_train.remove_columns("original_labels")
        self.trainer = DynamicTrainer(
            model_init=self._prepare_model_init_function(),
            args=self.training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            callbacks=[TrainerProgressReporterCallback(progress_reporter, DQ_TRAIN)],
        )

        self.trainer.train()
        self.model = self.trainer.model
