from __future__ import annotations

import glob
import json
import logging
import math
import os
import warnings
from collections import Counter
from collections.abc import Iterable
from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict

from ..constants import DQ_TRAIN
from ..methods.constants import OOS_LABEL
from ..progress_reporter import ProgressReporter
from .data_types import DatasetEvaluatorOutput, TaskType
from .dynamic_finetuner import DynamicTuner
from .sample_eval import AbstractSampleEvalMethod

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class DatasetEvaluator:
    """Evaluates dataset quality and identifies problematic samples.

    Runs multiple sample evaluation methods (Retag, Uncertainty, VInfo,
    LabelAggregation) to identify samples that may be mislabeled,
    ambiguous, or otherwise problematic for training.
    """

    def __init__(
        self,
        task_type: Union[str, TaskType],
        tuner: DynamicTuner,
        sample_evaluators: Iterable[AbstractSampleEvalMethod] | None,
        workdir: str = "./data_quality_output",
        save_results: bool = True,
        aggregate: str = "once",
    ):
        """Initialize the dataset evaluator.

        Args:
            task_type: Type of task ("multiclass_classification" or
                "token_classification").
            tuner: DynamicTuner instance for model-based evaluation.
            sample_evaluators: Iterable of evaluation methods to apply.
            workdir: Directory for saving evaluation results.
            save_results: Whether to save results to disk.
            aggregate: Aggregation strategy for multiple evaluators:
                - 'once': Flag if any method marks as problematic
                - 'all': Flag only if all methods agree
                - 'vote': Flag if majority of methods agree
        """
        self.task_type = (
            TaskType(task_type) if not isinstance(task_type, TaskType) else task_type
        )
        self.sample_evaluators: Iterable[AbstractSampleEvalMethod] = (
            self._validate_evaluators(sample_evaluators)
        )
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self.save_results = save_results
        self.aggregate = aggregate
        self.tuner = tuner
        self.id2label = None
        self.aggregated_report_filename = "aggregated_report.csv"

    # TODO: need more advanced validation, pydantic fields or smth like this
    def _validate_evaluators(
        self, sample_evaluators: Any
    ) -> Iterable[AbstractSampleEvalMethod]:
        """Validate and transform format to list of methods, or empty list

        Args:
            sample_evaluators (Iterable[AbstractSampleEvalMethod] | None): sample_evaluators

        Returns:
            list: sample_evaluators
        """
        if sample_evaluators is None:
            return []

        if isinstance(sample_evaluators, Iterable):
            for evaluator in sample_evaluators:
                if self.task_type not in evaluator.supported_tasks:
                    raise ValueError(
                        f"{evaluator.name} is not supported for {self.task_type} task"
                    )

            return sample_evaluators

        raise TypeError(
            f"sample_evaluators should be None, Iterable, not {type(sample_evaluators)}"
        )

    def filter_data(
        self, dataset: DatasetDict | Dataset, progress_reporter: ProgressReporter
    ) -> DatasetEvaluatorOutput:
        """The main method to filter dataset

        Args:
            dataset (DatasetDict | Dataset): dataset

        Returns:
            DatasetEvaluatorOutput
        """
        assert progress_reporter.stages is not None

        # no ood samples for data quality
        if isinstance(dataset, DatasetDict):
            filtered_splits = DatasetDict()
            for split_name, split_ds in dataset.items():
                if "label" in split_ds.column_names:
                    filtered_splits[split_name] = split_ds.filter(
                        lambda example: example["label"] != OOS_LABEL
                    )
                else:
                    filtered_splits[split_name] = split_ds
            dataset = filtered_splits
        elif isinstance(dataset, Dataset):
            if "label" in dataset.column_names:
                dataset = dataset.filter(lambda example: example["label"] != OOS_LABEL)

        # all evaluators, methods need to identify sample
        dataset = dataset.map(lambda _, idx: {"id": idx}, with_indices=True)

        if self._need_tuned_model():
            progress_reporter.stages = [DQ_TRAIN, *progress_reporter.stages]
            self._tune_model(dataset, progress_reporter)

        counter = Counter()
        method_candidates = {}
        for evaluator in self.sample_evaluators:
            progress_reporter(evaluator.name, 0.0)
            log.info(f"{evaluator.name:=^20}")
            tuner = self.tuner if evaluator.need_tuner_instance else None
            # TODO: передавать только нужные параметры
            indices_to_remove: List[int] = evaluator.candidates_to_remove(
                dataset,
                workdir=self.workdir,
                tuner=tuner,
                id2label=self.id2label,
                task_type=self.task_type,
            )
            method_candidates[evaluator.name] = indices_to_remove
            log.info(f"Candidates: {len(indices_to_remove)}")
            counter.update(indices_to_remove)
            progress_reporter(evaluator.name, 1.0)

        log.info(f"{'method to aggregate: ' + self.aggregate:=^10}")
        if self.aggregate == "all":
            counter = dict(
                filter(lambda x: x[1] == len(self.sample_evaluators), counter.items())
            )
        if self.aggregate == "vote":
            counter = dict(
                filter(
                    lambda x: x[1] >= math.ceil(len(self.sample_evaluators) / 2),
                    counter.items(),
                )
            )

        aggregate_indices = counter.keys()

        log.info(f"Final length of indices to remove: {len(aggregate_indices)}")

        dataset = dataset.filter(
            lambda _, idx: idx not in aggregate_indices, with_indices=True
        )
        dataset = dataset.remove_columns("id")
        if self.save_results:
            self._save(aggregate_indices, method_candidates)
            if self.aggregated_report_filename is None:
                aggregated_report_path = None
            else:
                aggregated_report_path = os.path.join(
                    self.workdir, self.aggregated_report_filename
                )
            return DatasetEvaluatorOutput(
                splits_filtered=dataset,
                indices_to_remove=aggregate_indices,
                aggregated_report_path=aggregated_report_path,
                reports_by_method_folder=self.workdir,
            )

        return DatasetEvaluatorOutput(
            splits_filtered=dataset,
            indices_to_remove=aggregate_indices,
            aggregated_report_path=None,
            reports_by_method_folder=None,
        )

    def _save(
        self, aggregate_indices: Iterable[int], method_candidates: Dict[str, List[int]]
    ):
        files_to_merge = None
        common_columns = ["id_sample", "text", "label"]
        if self.task_type == TaskType.NER:
            common_columns.append("id_token")

        for evaluator in self.sample_evaluators:
            evaluator.save(self.workdir)
            if evaluator.prepared_data.shape[0] > 0:
                evaluator.prepared_data[f"method_{evaluator.name}"] = evaluator.name
                if files_to_merge is None:
                    files_to_merge = evaluator.prepared_data
                    files_to_merge.columns = [
                        col if col in common_columns else col + "_" + evaluator.name
                        for col in files_to_merge.columns
                    ]
                else:
                    evaluator.prepared_data.columns = [
                        col if col in common_columns else col + "_" + evaluator.name
                        for col in evaluator.prepared_data.columns
                    ]
                    files_to_merge = files_to_merge.merge(
                        evaluator.prepared_data,
                        how="outer",
                        on=common_columns,
                        suffixes=("", "_" + evaluator.name),
                    )

        if files_to_merge is not None:
            method_columns = [
                c for c in files_to_merge.columns if c.startswith("method")
            ]
            files_to_merge[method_columns] = files_to_merge[method_columns].fillna("")
            files_to_merge["methods"] = files_to_merge[method_columns].agg(
                " ".join, axis=1
            )
            files_to_merge["methods"] = (
                files_to_merge["methods"].replace(r"\s+", " ", regex=True).str.strip()
            )
            files_to_merge = files_to_merge.drop(method_columns, axis=1)

            files_to_merge.to_csv(
                os.path.join(self.workdir, self.aggregated_report_filename), index=False
            )
            with open(
                os.path.join(self.workdir, "candidates_to_remove.json"), "w"
            ) as file:
                json.dump(method_candidates, file)
            with open(
                os.path.join(self.workdir, "aggregated_candidates.json"), "w"
            ) as file:
                json.dump(list(aggregate_indices), file)
        else:
            self.aggregated_report_filename = None

    def _need_tuned_model(self) -> bool:
        return any([eval.need_tuned_model for eval in self.sample_evaluators])

    def _tune_model(
        self, dataset: DatasetDict | Dataset, progress_reporter: ProgressReporter
    ):
        if progress_reporter is None:
            progress_reporter = ProgressReporter()

        filepaths = glob.glob(os.path.join(f"{self.workdir}", "**"), recursive=True)
        for filepath in filepaths:
            if os.path.isfile(filepath):
                os.remove(filepath)
                log.info(f"removed previous calculations: {filepath}")

        self.tuner.training_args.output_dir = self.workdir
        self.tuner.train(dataset["train"], progress_reporter)
        # token classification naming
        if hasattr(self.tuner, "label_names"):
            self.id2label = {k: v for k, v in enumerate(self.tuner.label_names)}
        # sequence classification naming
        if hasattr(self.tuner, "id2label"):
            self.id2label = self.tuner.id2label
