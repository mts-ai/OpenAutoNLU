from __future__ import annotations

import glob
import json
import logging
import os
import sys
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from crowdkit.aggregation import DawidSkene
from datasets import Dataset, DatasetDict
from matplotlib.figure import Figure

from .data_types import TaskType
from .dynamic_finetuner import DynamicTuner
from .utils import staticproperty

SOURCE_PATH = os.path.split(os.getcwd())[0]
sys.path.append(SOURCE_PATH)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HYDRA_FULL_ERROR"] = "1"


log = logging.getLogger(__name__)

Report = Union[List[Dict[str, Union[int, List[float]]]], pd.DataFrame]


class AbstractSampleEvalMethod(ABC):
    def __init__(self, criteria: dict):
        self.criteria = criteria
        self.prepared_data: pd.DataFrame = None

    @abstractmethod
    def candidates_to_remove(
        self, dataset: DatasetDict, workdir: str, task_type: TaskType
    ) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def save(self, savedir: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _preprocess_data(
        self, dataset: DatasetDict, workdir: str
    ) -> Dict[str, Dict[str, int | List[float]]]:
        raise NotImplementedError()

    @abstractmethod
    def _finalize_report(self, dataset: DatasetDict, report: Report) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _remove_prev_exps(self, workdir: str) -> None:
        raise NotImplementedError()

    @property
    def name(self):
        return re.sub("(?!^)([A-Z]+)", r"_\1", self.__class__.__name__).lower()

    @staticproperty
    @abstractmethod
    def need_tuned_model():
        raise NotImplementedError()

    @staticproperty
    @abstractmethod
    def need_tuner_instance():
        raise NotImplementedError()

    @staticproperty
    @abstractmethod
    def supported_tasks() -> List[TaskType]:
        raise NotImplementedError()


class BaseSampleEvalMethod(AbstractSampleEvalMethod):
    def save(self, savedir: str) -> None:
        savedir_postfix = os.path.join(savedir, self.name)
        os.makedirs(savedir_postfix, exist_ok=True)
        self._remove_prev_exps(savedir_postfix)

        path_file = os.path.join(savedir_postfix, f"{self.name}_by_sample_stat.csv")
        if "id" in self.prepared_data.columns:
            self.prepared_data = self.prepared_data.drop(columns=["id"])
        self.prepared_data.to_csv(path_file, index=False)
        log.info(f"{self.name} stats saved to {path_file}")

    def _remove_prev_exps(self, workdir: str) -> None:
        """removing previous data quality results"""
        filepaths = glob.glob(os.path.join(f"{workdir}", "**"), recursive=True)
        for filepath in filepaths:
            if os.path.isfile(filepath):
                os.remove(filepath)
                log.info(f"removed previous calculations: {filepath}")

    def _finalize_report(
        self,
        dataset: Dataset,
        report: Report,
        id2label: Dict[int, str],
    ) -> None:
        if isinstance(report, list):
            self.prepared_data = pd.DataFrame(report)
        elif isinstance(report, pd.DataFrame):
            self.prepared_data = report
        else:
            raise ValueError("report must be a list")

        if self.prepared_data.shape[0] == 0:
            return

        if "id_sample" not in self.prepared_data.columns:
            raise ValueError("id_sample column is missing in report")

        dataset_df = dataset.to_pandas()
        if "id" not in dataset_df.columns:
            dataset_df["id"] = dataset_df.index

        columns_from_dataset = ["id", "text"]
        if "text" not in dataset_df.columns:
            raise ValueError("text column is missing in dataset")

        if "pred_label" in self.prepared_data.columns:
            if self.prepared_data["pred_label"].dtype == int:
                self.prepared_data["pred_label"] = self.prepared_data["pred_label"].map(
                    id2label
                )
            elif self.prepared_data["pred_label"].dtype == list:
                self.prepared_data["pred_label"] = self.prepared_data[
                    "pred_label"
                ].apply(lambda x: list(map(id2label.get, x)))

        if "pred_label_by_epoch" in self.prepared_data.columns:
            self.prepared_data["pred_label_by_epoch"] = self.prepared_data[
                "pred_label_by_epoch"
            ].apply(lambda x: list(map(id2label.get, x)))

        if "tokens" in dataset_df.columns:
            columns_from_dataset.append("tokens")
        # token classification
        if "labels" in dataset_df.columns:
            columns_from_dataset.append("labels")
        # sequence classification
        elif "label" in dataset_df.columns:
            columns_from_dataset.append("label")

        dataset_df = dataset_df[columns_from_dataset]
        self.prepared_data = self.prepared_data.merge(
            dataset_df, left_on="id_sample", right_on="id", how="left"
        )
        if (
            "tokens" in dataset_df.columns
            and "id_token" in self.prepared_data
            and "labels" in dataset_df.columns
        ):

            def get_pred_tokens(row):
                row["tokens"] = row["tokens"][row["id_token"]]
                row["labels"] = row["labels"][row["id_token"]]
                return row

            self.prepared_data = self.prepared_data.apply(get_pred_tokens, axis=1)
            self.prepared_data.rename(
                columns={"tokens": "candidate_token"}, inplace=True
            )
            self.prepared_data.rename(columns={"labels": "label"}, inplace=True)
            self.prepared_data.drop(["id"], axis=1, inplace=True)

    def _task_resolver(self, task_type: TaskType):
        if task_type == TaskType.MULTICLASS:
            return self._candidates_multiclass
        if task_type == TaskType.NER:
            return self._candidates_ner

        raise ValueError(f"Unknown task type: {task_type}")

    def _candidates_multiclass(self, data: Dict[str, Dict[str, int | List[float]]]):
        raise NotImplementedError()

    def _candidates_ner(self, data: Dict[str, Dict[str, int | List[float]]]):
        raise NotImplementedError()


class ModelBasedMethod(BaseSampleEvalMethod):
    def candidates_to_remove(
        self,
        dataset: DatasetDict,
        workdir: str,
        task_type: TaskType,
        id2label: Optional[Dict[int, str]],
        tuner: DynamicTuner = None,
    ) -> List[int]:
        data = self._preprocess_data(dataset, workdir, tuner=tuner)

        task_method = self._task_resolver(task_type)
        prepared_data = task_method(data)

        self._finalize_report(dataset["train"], prepared_data, id2label)

        if self.prepared_data.shape[0] == 0:
            return []

        return self.prepared_data["id_sample"].tolist()

    @staticproperty
    def need_tuned_model():
        return True

    @staticproperty
    def need_tuner_instance():
        return False

    @staticproperty
    def supported_tasks() -> List[TaskType]:
        return [TaskType.MULTICLASS, TaskType.NER]


class LogitBasedMethod(ModelBasedMethod):
    need_last_epoch = True

    def __init__(
        self,
        criteria: dict,
    ):
        super().__init__(criteria)

    def _preprocess_data(
        self, dataset: DatasetDict, workdir: str, tuner: DynamicTuner = None
    ) -> Dict[str, Dict[str, int | List[float]]]:
        return self._load_data_from_tuner(workdir)

    def _load_data_from_tuner(
        self, path: str
    ) -> Dict[str, Dict[str, int | List[float]]]:
        """Load logits from saved result.jsonl. This file
        can be located in self.workdir as well as in subfolder

        Args:
            path (str): where result.jsonl located

        Returns:
            Dict[str, Dict[str, int | List[float]]]: reformated file for further processing
        """
        dirpattern = os.path.join(path, "result.jsonl")

        reformat_data = defaultdict(dict)

        with open(dirpattern, "r") as file:
            for line in file:
                record = json.loads(line)
                id_sample = record["id_sample"]
                epoch = record["epoch"]
                if id_sample not in reformat_data:
                    assert epoch == 0
                    reformat_data[id_sample]["gold"] = record["gold"]
                    reformat_data[id_sample]["logits"] = []
                if self.__class__.need_last_epoch:
                    reformat_data[id_sample]["logits"] = record["logits"]
                else:
                    reformat_data[id_sample]["logits"].append(record["logits"])

        return reformat_data


class Cartography(LogitBasedMethod):
    need_last_epoch = False

    def __init__(
        self,
        criteria: Dict[str, float] = {"confidence": 0.1, "variability": 0.1},
        include_ci: bool = False,
    ):
        """Sample Eval class for Dataset Cartography

        Args:
            workdir (List[str]|str): json files dirs
            criteria (dict): criterias to filter data
            include_ci (bool): Based on prior work on active bias (https://arxiv.org/abs/1704.07433).
                Default to False

        """
        super().__init__(criteria)
        self.include_ci = include_ci
        self.plot = None

    def _candidates_multiclass(self, data: dict) -> List[Dict[str, int | float]]:
        prepared_data = []

        log.info("Metrics computed: confidence, variability, correctness")

        for i, (id_sample, record) in enumerate(data.items()):
            row = {"id_sample": id_sample}

            true_probs_trend = []
            correctness_trend = []
            preds = []
            for epoch_logits in record["logits"]:
                probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
                true_class_prob = float(probs[record["gold"]])
                true_probs_trend.append(true_class_prob)

                prediction = np.argmax(epoch_logits)
                is_correct = (prediction == record["gold"]).item()
                correctness_trend.append(is_correct)
                preds.append(prediction)

            row["correctness"] = sum(correctness_trend)
            row["confidence"] = np.mean(true_probs_trend)
            row["variability"] = self._variability(true_probs_trend)
            row["pred_label"] = preds[-1]
            row["pred_label_by_epoch"] = preds

            prepared_data.append(row)

        self.plot = self.__class__._plot_data_map(prepared_data)

        prepared_data = list(
            filter(
                lambda x: x["confidence"] <= self.criteria["confidence"]
                and x["variability"] <= self.criteria["variability"],
                prepared_data,
            )
        )

        return prepared_data

    def save(self, savedir: str) -> None:
        """savedir may differ from self.workdir
        self.workdir only for processing results/caching

        Args:
            savedir (str): savedir
        """
        savedir_postfix = os.path.join(savedir, self.name)
        os.makedirs(savedir_postfix, exist_ok=True)
        self._remove_prev_exps(savedir_postfix)
        path_file = os.path.join(savedir_postfix, "datamap.pdf")
        if self.plot is not None:
            self.plot.savefig(path_file, dpi=300)
        log.info(f"{self.name} plot saved to {path_file}")

        path_file = os.path.join(savedir_postfix, f"{self.name}_by_sample_stat.csv")
        if "id" in self.prepared_data.columns:
            self.prepared_data = self.prepared_data.drop(columns=["id"])
        self.prepared_data.to_csv(path_file, index=False)

    @staticmethod
    def _plot_data_map(
        prepared_data: List[dict],
        hue_metric: str = "correct.",
        show_hist: bool = True,
        max_instances_to_plot: int = 55000,
    ) -> Figure:
        prepared_data = pd.DataFrame(prepared_data)
        # Set style.
        sns.set_theme(style="whitegrid", font_scale=1.6, context="paper")
        log.info("Plotting...")

        # Subsample data to plot, so the plot is not too busy.
        dataframe = prepared_data.sample(
            n=(
                max_instances_to_plot
                if prepared_data.shape[0] > max_instances_to_plot
                else len(prepared_data)
            )
        )

        # Normalize correctness to a value between 0 and 1.
        dataframe = dataframe.assign(
            corr_frac=lambda d: d.correctness / d.correctness.max()
        )
        dataframe["correct."] = [f"{x:.1f}" for x in dataframe["corr_frac"]]

        main_metric = "variability"
        other_metric = "confidence"

        hue = hue_metric
        num_hues = len(dataframe[hue].unique().tolist())
        style = hue_metric if num_hues < 8 else None

        if not show_hist:
            fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
        else:
            fig = plt.figure(
                figsize=(14, 10),
            )
            gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
            ax0 = fig.add_subplot(gs[:, 0])

        # Make the scatterplot.
        # Choose a palette.
        pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

        plot = sns.scatterplot(
            x=main_metric,
            y=other_metric,
            ax=ax0,
            data=dataframe,
            hue=hue,
            palette=pal,
            style=style,
            s=30,
        )

        # Annotate Regions.
        def bb(c):
            return dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")

        def func_annotate(text, xyc, bbc):
            return ax0.annotate(
                text,
                xy=xyc,
                xycoords="axes fraction",
                fontsize=15,
                color="black",
                va="center",
                ha="center",
                rotation=350,
                bbox=bb(bbc),
            )

        an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc="black")  # noqa
        an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc="r")  # noqa
        an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc="b")  # noqa

        if not show_hist:
            plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc="right")
        else:
            plot.legend(fancybox=True, shadow=True, ncol=1)
        plot.set_xlabel("variability")
        plot.set_ylabel("confidence")

        if show_hist:
            plot.set_title("Data Map", fontsize=17)

            # Make the histograms.
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[2, 1])

            plott0 = dataframe.hist(column=["confidence"], ax=ax1, color="#622a87")
            plott0[0].set_title("")
            plott0[0].set_xlabel("confidence")
            plott0[0].set_ylabel("density")

            plott1 = dataframe.hist(column=["variability"], ax=ax2, color="teal")
            plott1[0].set_title("")
            plott1[0].set_xlabel("variability")
            plott1[0].set_ylabel("density")

            plot2 = sns.countplot(x="correct.", data=dataframe, ax=ax3, color="#86bf91")
            ax3.xaxis.grid(True)  # Show the vertical gridlines

            plot2.set_title("")
            plot2.set_xlabel("correctness")
            plot2.set_ylabel("density")

        fig.tight_layout()

        return fig

    def _variability(self, data: List[float | int]):
        # Functions to be applied to the data.
        if self.include_ci:
            variab = np.sqrt(
                np.var(data) + np.var(data) * np.var(data) / (len(data) - 1)
            )
            return variab

        return np.std(data)

    @staticproperty
    def supported_tasks() -> List[TaskType]:
        return [TaskType.MULTICLASS]


class VInfo(LogitBasedMethod):
    need_last_epoch = True

    def __init__(
        self,
        criteria: Dict[str, float] = {"v_info": 0.5},
    ):
        super().__init__(criteria)

    def _candidates_multiclass(
        self, data: dict
    ) -> dict:  # TODO: refactor format (need more general)
        proc_data = {}
        for stage in data:
            proc_data[stage] = self.__class__._v_info_prep(data[stage])

        prepared_data = proc_data["stage 0"]

        prepared_data["null_logits"] = proc_data["stage 1"]["logits"]
        prepared_data["score"] = prepared_data["null_logits"] - prepared_data["logits"]
        prepared_data = prepared_data.sort_values("score")
        prepared_data = prepared_data[prepared_data["score"] <= self.criteria["v_info"]]

        return prepared_data

    def _preprocess_data(
        self,
        dataset: DatasetDict,
        workdir: str,
        tuner: DynamicTuner,
    ) -> Dict[
        str, Dict[str, Dict[str, int | List[float]]]
    ]:  # TODO: need dataclasses/pydantic for complex types
        data_0 = self._load_data_from_tuner(workdir)

        # null model
        dataset = dataset.map(lambda _: {"text": " "})
        tuner.training_args.output_dir = os.path.join(workdir, "null_model")
        # train in tuner reload and reinit
        tuner.train(dataset["train"])

        data_1 = self._load_data_from_tuner(tuner.training_args.output_dir)

        preproc_data = {}
        preproc_data["stage 0"] = data_0
        preproc_data["stage 1"] = data_1

        return preproc_data

    @staticmethod
    def _v_info_prep(raw_data_stage: Dict[str, int | List[float]]) -> pd.DataFrame:
        def prob_gold(sample):
            sample["logits"] = sample["logits"][sample["gold"]]
            sample["logits"] = -torch.log2(sample["logits"]).item()

            return sample

        df = pd.DataFrame.from_dict(raw_data_stage, orient="index")
        df["logits"] = df["logits"].apply(
            lambda x: torch.softmax(torch.tensor(x), dim=-1)
        )
        df["pred_label"] = df["logits"].apply(lambda x: torch.argmax(x, dim=-1).item())
        df = df.apply(prob_gold, axis=1)
        df["id_sample"] = df.index
        df.drop("gold", axis=1, inplace=True)

        return df

    @staticproperty
    def need_tuner_instance():
        return True

    @staticproperty
    def supported_tasks() -> List[TaskType]:
        return [TaskType.MULTICLASS]


class Uncertainty(LogitBasedMethod):
    need_last_epoch = True

    def __init__(
        self,
        criteria: dict = {"uncertainty_threshold": 0.95},
    ):
        super().__init__(criteria)

    def _candidates_ner(
        self, data: Dict[int, List[float]]
    ) -> List[Dict[str, int | float]]:
        uncertainty_scores = []
        for idx in data:
            gold = data[idx]["gold"]
            logits = data[idx]["logits"]
            probs = torch.softmax(torch.tensor(logits), dim=-1)
            i = 0
            for _gold, _probs in zip(gold, probs):
                if _gold != -100:
                    row = {
                        "id_sample": idx,
                        "id_token": i,
                        "pred_label": _probs.argmax().item(),
                    }
                    uncertainty = 1 - _probs[_gold]
                    row["score"] = round(uncertainty.item(), 2)

                    uncertainty_scores.append(row)
                    i += 1

        uncertainty_scores = list(
            filter(
                lambda x: x["score"] > self.criteria["uncertainty_threshold"],
                uncertainty_scores,
            )
        )

        return uncertainty_scores

    def _candidates_multiclass(
        self, data: Dict[int, List[float]]
    ) -> List[Dict[str, int | float]]:
        uncertainty_scores = []
        for idx in data:
            gold = data[idx]["gold"]
            logits = data[idx]["logits"]
            probs = torch.softmax(torch.tensor(logits), dim=-1)
            row = {"id_sample": idx, "pred_label": probs.argmax().item()}
            uncertainty = 1 - probs[gold]
            row["score"] = round(uncertainty.item(), 2)
            uncertainty_scores.append(row)

        uncertainty_scores = list(
            filter(
                lambda x: x["score"] > self.criteria["uncertainty_threshold"],
                uncertainty_scores,
            )
        )

        return uncertainty_scores

    @staticproperty
    def supported_tasks() -> List[TaskType]:
        return [TaskType.NER, TaskType.MULTICLASS]


class Retag(LogitBasedMethod):
    need_last_epoch = True

    def __init__(
        self,
        criteria: dict = {},
    ):
        super().__init__(criteria)

    def _candidates_ner(
        self, data: Dict[int, List[float]]
    ) -> List[Dict[str, int | float]]:
        candidates = []
        for idx in data:
            gold = np.array(data[idx]["gold"])
            idx_true = np.where(gold != -100)[0]
            logits = np.array(data[idx]["logits"])

            # filter -100 tokens
            gold = gold[idx_true]
            logits = logits[idx_true]

            labels = logits.argmax(axis=-1).tolist()
            cols = np.where((gold != labels) & (gold != -100))[0].tolist()
            rows = [
                {"id_sample": idx, "id_token": col, "pred_label": label}
                for col, label in zip(cols, labels)
            ]
            candidates.extend(rows)

        return candidates

    def _candidates_multiclass(self, data) -> List[Dict[str, int | float]]:
        candidates = []
        for idx in data:
            gold = data[idx]["gold"]
            logits = data[idx]["logits"]
            label = np.array(logits).argmax(axis=-1)
            if gold != label:
                row = {"id_sample": idx, "pred_label": label}
                candidates.append(row)

        return candidates

    @staticproperty
    def supported_tasks() -> List[TaskType]:
        return [TaskType.NER, TaskType.MULTICLASS]


class LabelAggregation(ModelBasedMethod):
    def __init__(
        self,
        criteria: dict = {},
        num_repetitions: int = 10,
    ):
        super().__init__(criteria)
        self.num_repetitions = num_repetitions

    # TODO: Label Aggregation for sequence classification
    def candidates_to_remove(
        self,
        dataset: DatasetDict,
        workdir: str,
        task_type: TaskType,
        id2label: Optional[Dict[int, str]] = None,
        tuner: DynamicTuner = None,
    ) -> List[int]:
        raw_data, ground_truth = self._preprocess_data(dataset, workdir, tuner=tuner)
        ground_truth_flat = ground_truth.reshape(-1)
        # (num_rep, N, max_length, num_classes)
        mc_dropout_resulit = np.array(raw_data)
        # (num_rep, N, max_length)
        label_result = mc_dropout_resulit.argmax(axis=-1)
        # (num_rep, N*max_length)
        label_result = label_result.reshape(label_result.shape[0], -1)
        # (N*max_length, num_rep)
        label_result = label_result.transpose()

        tasks = np.repeat(np.arange(label_result.shape[0]), self.num_repetitions)
        performers = np.tile(np.arange(self.num_repetitions), label_result.shape[0])
        labels = label_result.reshape(-1)

        assert len(labels) == tasks.shape[0] == performers.shape[0]

        data = {"task": tasks, "worker": performers, "label": labels}
        df = pd.DataFrame(data)
        log.info("Started aggregation model")
        log.info(f"size for DawidSkene model: {df.shape}")
        dawid_skene_model = DawidSkene(n_iter=100)
        encoded_predictions = dawid_skene_model.fit_predict(df)
        encoded_predictions = encoded_predictions.to_numpy()
        log.info("Finished aggregation model")

        flags_condition = (encoded_predictions != ground_truth_flat) & (
            ground_truth_flat != -100
        )
        # flags_condition = (encoded_predictions[ground_truth!=-100] != ground_truth[ground_truth!=-100])
        flags_flat = np.where(flags_condition)[0]

        if flags_flat.shape[0] == 0:
            self._finalize_report(dataset["train"], [], id2label)
            return []

        N, token_n = np.unravel_index(flags_flat, ground_truth.shape)
        flags = list(zip(N, token_n))

        prepared_data = []
        # disalignment
        for i, flag in enumerate(flags):
            m, n = flag
            correction = sum(ground_truth[m][:n] == -100)
            row = {
                "id_sample": m,
                "id_token": n - correction,
                "pred_label": encoded_predictions[(m * ground_truth.shape[1]) + n],
            }
            prepared_data.append(row)

        self._finalize_report(dataset["train"], prepared_data, id2label)

        return self.prepared_data["id_sample"].tolist()

    def _preprocess_data(
        self, dataset: DatasetDict, workdir: str, tuner: DynamicTuner
    ) -> Dict[str, Dict[str, int | List[float]]]:
        raw_data = tuner.mc_droupout(
            dataset["train"], num_repetitions=self.num_repetitions
        )

        return raw_data

    @staticproperty
    def need_tuned_model():
        return True

    @staticproperty
    def need_tuner_instance():
        return True

    @staticproperty
    def supported_tasks() -> List[TaskType]:
        return [TaskType.NER]
