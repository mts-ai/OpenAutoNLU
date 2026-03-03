import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import transformers
from datasets import Dataset, DatasetDict
from packaging import version
from sentence_transformers import SentenceTransformer
from sentence_transformers.datasets import NoDuplicatesDataLoader, SentenceLabelDataset
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.fit_mixin import (
    EvaluatorCallback,
    OriginalCallback,
    SaveModelCallback,
)
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import is_datasets_available
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ...constants import PRETRAINING_STAGE
from ...progress_reporter import (
    ProgressReporter,
    TrainerProgressReporterCallback,
)

logger = logging.getLogger(__name__)


# re-implementation of fit mixin from https://github.com/UKPLab/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/fit_mixin.py#L162
# for better control
def fit_encoder(
    encoder: SentenceTransformer,
    train_objectives: Iterable[tuple[DataLoader, nn.Module]],
    evaluator: SentenceEvaluator = None,
    epochs: int = 1,
    steps_per_epoch=None,
    scheduler: str = "WarmupLinear",
    warmup_steps: int = 10000,
    optimizer_class: type[Optimizer] = torch.optim.AdamW,
    optimizer_params: dict[str, object] = {"lr": 2e-5},
    weight_decay: float = 0.01,
    evaluation_steps: int = 0,
    output_path: str = None,
    save_best_model: bool = True,
    max_grad_norm: float = 1,
    use_amp: bool = False,
    callback: Callable[[float, int, int], None] = None,
    show_progress_bar: bool = True,
    checkpoint_path: str = None,
    checkpoint_save_steps: int = 500,
    checkpoint_save_total_limit: int = 0,
    progress_reporter: Optional[ProgressReporter] = None,
) -> SentenceTransformer:
    """
    Deprecated training method from before Sentence Transformers v3.0, it is recommended to use
    :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` instead. This method uses
    :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` behind the scenes, but does
    not provide as much flexibility as the Trainer itself.

    This training approach uses a list of DataLoaders and Loss functions to train the model. Each DataLoader
    is sampled in turn for one batch. We sample only as many batches from each DataLoader as there are in the
    smallest one to make sure of equal training with each dataset, i.e. round robin sampling.

    This method should produce equivalent results in v3.0+ as before v3.0, but if you encounter any issues
    with your existing training scripts, then you may wish to use
    :meth:`SentenceTransformer.old_fit <sentence_transformers.SentenceTransformer.old_fit>` instead.
    That uses the old training method from before v3.0.

    Args:
        encoder: SentenceTransformer model to train.
        train_objectives: Tuples of (DataLoader, LossFunction). Pass
            more than one for multi-task learning
        evaluator: An evaluator (sentence_transformers.evaluation)
            evaluates the model performance during training on held-
            out dev data. It is used to determine the best model
            that is saved to disc.
        epochs: Number of epochs for training
        steps_per_epoch: Number of training steps per epoch. If set
            to None (default), one epoch is equal the DataLoader
            size from train_objectives.
        scheduler: Learning rate scheduler. Available schedulers:
            constantlr, warmupconstant, warmuplinear, warmupcosine,
            warmupcosinewithhardrestarts
        warmup_steps: Behavior depends on the scheduler. For
            WarmupLinear (default), the learning rate is increased
            from o up to the maximal learning rate. After these many
            training steps, the learning rate is decreased linearly
            back to zero.
        optimizer_class: Optimizer
        optimizer_params: Optimizer parameters
        weight_decay: Weight decay for model parameters
        evaluation_steps: If > 0, evaluate the model using evaluator
            after each number of training steps
        output_path: Storage path for the model and evaluation files
        save_best_model: If true, the best model (according to
            evaluator) is stored at output_path
        max_grad_norm: Used for gradient normalization.
        use_amp: Use Automatic Mixed Precision (AMP). Only for
            Pytorch >= 1.6.0
        callback: Callback function that is invoked after each
            evaluation. It must accept the following three
            parameters in this order: `score`, `epoch`, `steps`
        show_progress_bar: If True, output a tqdm progress bar
        checkpoint_path: Folder to save checkpoints during training
        checkpoint_save_steps: Will save a checkpoint after so many
            steps
        checkpoint_save_total_limit: Total number of checkpoints to
            store
    """
    if not is_datasets_available():
        raise ImportError(
            "Please install `datasets` to use this function: `pip install datasets`."
        )

    # Delayed import to counter the SentenceTransformers -> FitMixin -> SentenceTransformerTrainer -> SentenceTransformers circular import
    from sentence_transformers.trainer import SentenceTransformerTrainer

    data_loaders, loss_fns = zip(*train_objectives)

    # Clear the dataloaders from collate functions as we just want raw InputExamples
    def identity(batch):
        return batch

    for data_loader in data_loaders:
        data_loader.collate_fn = identity

    batch_size = 8
    batch_sampler = BatchSamplers.BATCH_SAMPLER
    # Convert dataloaders into a DatasetDict
    # TODO: This is rather inefficient, as we load all data into memory. We might benefit from a more efficient solution
    train_dataset_dict = {}
    for loader_idx, data_loader in enumerate(data_loaders, start=1):
        if isinstance(data_loader, NoDuplicatesDataLoader):
            batch_sampler = BatchSamplers.NO_DUPLICATES
        elif hasattr(data_loader, "dataset") and isinstance(
            data_loader.dataset, SentenceLabelDataset
        ):
            batch_sampler = BatchSamplers.GROUP_BY_LABEL

        batch_size = getattr(data_loader, "batch_size", batch_size)
        texts = []
        labels = []
        for batch in data_loader:
            batch_texts, batch_labels = zip(
                *[(example.texts, example.label) for example in batch]
            )
            texts += batch_texts
            labels += batch_labels
        dataset = Dataset.from_dict(
            {f"sentence_{idx}": text for idx, text in enumerate(zip(*texts))}
        )
        # Add label column, unless all labels are 0 (the default value for `labels` in InputExample)
        add_label_column = True
        try:
            if set(labels) == {0}:
                add_label_column = False
        except TypeError:
            pass
        if add_label_column:
            dataset = dataset.add_column("label", labels)
        train_dataset_dict[f"_dataset_{loader_idx}"] = dataset

    train_dataset_dict = DatasetDict(train_dataset_dict)

    def _default_checkpoint_dir() -> str:
        dir_name = "checkpoints/model"
        idx = 1
        while Path(dir_name).exists() and len(list(Path(dir_name).iterdir())) != 0:
            dir_name = f"checkpoints/model_{idx}"
            idx += 1
        return dir_name

    # Convert loss_fns into a dict with `dataset_{idx}` keys
    loss_fn_dict = {
        f"_dataset_{idx}": loss_fn for idx, loss_fn in enumerate(loss_fns, start=1)
    }

    # Use steps_per_epoch to perhaps set max_steps
    max_steps = -1
    if steps_per_epoch is not None and steps_per_epoch > 0:
        if epochs == 1:
            max_steps = steps_per_epoch
        else:
            logger.warning(
                "Setting `steps_per_epoch` alongside `epochs` > 1 no longer works. "
                "We will train with the full datasets per epoch."
            )
            steps_per_epoch = None

    # Transformers renamed `evaluation_strategy` to `eval_strategy` in v4.41.0
    eval_strategy_key = (
        "eval_strategy"
        if version.parse(transformers.__version__) >= version.parse("4.41.0")
        else "evaluation_strategy"
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=checkpoint_path or _default_checkpoint_dir(),
        batch_sampler=batch_sampler,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.ROUND_ROBIN,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        **{
            eval_strategy_key: "steps"
            if evaluation_steps is not None and evaluation_steps > 0
            else "no",
        },
        eval_steps=evaluation_steps,
        # load_best_model_at_end=save_best_model, # <- TODO: Look into a good solution for save_best_model
        max_grad_norm=max_grad_norm,
        fp16=use_amp,
        disable_tqdm=not show_progress_bar,
        save_strategy="steps" if checkpoint_path is not None else "no",
        save_steps=checkpoint_save_steps,
        save_total_limit=checkpoint_save_total_limit,
        report_to=["none"],
    )

    if steps_per_epoch is None or steps_per_epoch == 0:
        steps_per_epoch = min(
            [
                len(train_dataset) // batch_size
                for train_dataset in train_dataset_dict.values()
            ]
        )
    num_train_steps = int(steps_per_epoch * epochs)

    # Prepare optimizer & scheduler
    param_optimizer = list(encoder.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler_obj = encoder._get_scheduler(
        optimizer,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        t_total=num_train_steps,
    )

    # Create callbacks
    callbacks = []
    if evaluator is not None:
        callbacks.append(EvaluatorCallback(evaluator, output_path))
        if callback is not None:
            callbacks.append(OriginalCallback(callback, evaluator))
    if progress_reporter is not None:
        callbacks.append(
            TrainerProgressReporterCallback(progress_reporter, PRETRAINING_STAGE)
        )

    trainer = SentenceTransformerTrainer(
        model=encoder,
        args=args,
        train_dataset=train_dataset_dict,
        eval_dataset=None,
        loss=loss_fn_dict,
        evaluator=evaluator,
        optimizers=(optimizer, scheduler_obj),
        callbacks=callbacks,
    )
    # Set the trainer on the EvaluatorCallback, required for logging the metrics
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, EvaluatorCallback):
            callback.trainer = trainer

    if output_path is not None:
        trainer.add_callback(SaveModelCallback(output_path, evaluator, save_best_model))

    trainer.train()
    return trainer.model
