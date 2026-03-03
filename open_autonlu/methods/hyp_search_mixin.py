import logging
from dataclasses import replace
from typing import Callable, Dict, Tuple, Optional

import numpy as np
import optuna
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import HPSearchBackend, IntervalStrategy

from ..constants import HPO_STAGE

from .finetuning_training_args import OptimizableHyperParameter, OptunaSuggestionDtype

log = logging.getLogger(__name__)


class EarlyStoppingCallbackPatched(EarlyStoppingCallback):
    """
    Patched version of the EarlyStoppingCallback to handle specific requirements.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Method called at the beginning of training.

        Args:
            args: TrainingArguments object.
            state: TrainerState object.
            control: TrainerControl object.
            **kwargs: Additional keyword arguments.
        """
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"
        self.early_stopping_patience_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Method called during evaluation.

        Args:
            args: TrainingArguments object.
            state: TrainerState object.
            control: TrainerControl object.
            metrics: Dictionary of evaluation metrics.
            **kwargs: Additional keyword arguments.
        """
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            log.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)

        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or operator(metric_value, state.best_metric):
            state.best_metric = metric_value


class ImprovementActivatedEarlyStopping(EarlyStoppingCallback):
    """Early stopping callback that activates only after initial improvements.

    Attributes:
        early_stopping_patience: Number of evaluations without improvement
            before stopping (after activation).
        early_stopping_threshold: Minimum change to qualify as improvement.
        improvement_threshold: Number of improvements required to activate
            early stopping.
    """

    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: Optional[float] = 0.0,
        improvement_threshold: Optional[int] = 3,
    ):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.early_stopping_patience = early_stopping_patience
        self.improvement_threshold = improvement_threshold
        self.best_metric = None
        self.patience_counter = 0
        self.improvement_counter = 0
        self.early_stopping_active = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        # Replace 'eval_loss' with your metric name if different
        current_metric = metrics.get("eval_loss")

        if self.best_metric is None:
            # First evaluation
            self.best_metric = current_metric
            return

        if (
            current_metric < self.best_metric
        ):  # Metric improved (use > if lower is better)
            self.improvement_counter += 1
            self.best_metric = current_metric
            self.patience_counter = 0  # Reset patience counter on improvement

            # Activate early stopping after threshold improvements
            if (
                not self.early_stopping_active
                and self.improvement_counter >= self.improvement_threshold
            ):
                print(
                    f"Early stopping activated after {self.improvement_threshold} improvements"
                )
                self.early_stopping_active = True
        else:
            # Metric didn't improve
            if self.early_stopping_active:
                self.patience_counter += 1
                print(
                    f"Early stopping patience: {self.patience_counter}/{self.early_stopping_patience}"
                )

                if self.patience_counter >= self.early_stopping_patience:
                    print("Triggering early stopping")
                    control.should_training_stop = True


class FinetuningHyperparameterSearchMixin:
    """Mixin providing Optuna-based hyperparameter optimization for finetuning.

    Adds hyperparameter search capabilities to transformer training classes.
    Uses Optuna with TPE (Tree-structured Parzen Estimator) sampler for
    efficient hyperparameter exploration.

    Usage:
        Inherit from this mixin and call `optimize_hyperparameters()` with a
        configured Trainer and hyperparameter search space.

    Attributes:
        hyp_search_done: Flag indicating if HPO has been completed.
    """

    @staticmethod
    def _prepare_hp_space(
        **hp_space_config: Dict[str, OptimizableHyperParameter],
    ) -> callable:
        """
        Prepares the hyperparameter space for Optuna.

        Args:
            **hp_space_config: Dictionary of hyperparameter configurations.

        Returns:
            callable: A function that defines the hyperparameter space for Optuna.
        """

        def optuna_hp_space(trial):
            hp_space = {}
            for hp_name, optimizable in hp_space_config.items():
                if optimizable.dtype != OptunaSuggestionDtype.CATEGORICAL:
                    hp_space[hp_name] = getattr(
                        trial, f"suggest_{optimizable.dtype.value}"
                    )(hp_name, *optimizable.range, **optimizable.optuna_kwargs)
                else:
                    hp_space[hp_name] = getattr(
                        trial, f"suggest_{optimizable.dtype.value}"
                    )(hp_name, optimizable.range)
            return hp_space

        return optuna_hp_space

    @staticmethod
    def _prepare_args(args: TrainingArguments) -> Tuple[TrainingArguments, dict]:
        """
        Prepares the training arguments for hyperparameter search.

        Args:
            args (TrainingArguments): The training arguments.

        Returns:
            dict: A dictionary containing the modified arguments and a backup of the original values.
        """
        backup = {
            "load_best_model_at_end": args.load_best_model_at_end,
            "save_strategy": args.save_strategy,
        }
        args.load_best_model_at_end = False
        args.save_strategy = IntervalStrategy.NO
        return args, backup

    @staticmethod
    def _restore_args(args_backup: dict, args: TrainingArguments) -> TrainingArguments:
        """
        Restores the training arguments to their original values.

        Args:
            args_backup (dict): A dictionary containing the backup of the original arguments.
            args (TrainingArguments): The training arguments.

        Returns:
            TrainingArguments: The restored training arguments.
        """
        args = replace(args, **args_backup)
        return args

    @staticmethod
    def _patch_or_add_early_stopping(trainer: Trainer):
        """
        Patches the early stopping callback in the trainer.

        Args:
            trainer (Trainer): The trainer object.

        Returns:
            Trainer: The patched trainer object.
        """
        old_callback = trainer.callback_handler.pop_callback(EarlyStoppingCallback)
        if old_callback:
            trainer.add_callback(
                EarlyStoppingCallbackPatched(
                    old_callback.early_stopping_patience,
                    old_callback.early_stopping_threshold,
                )
            )
        else:
            trainer.add_callback(EarlyStoppingCallbackPatched(5))
        return trainer, old_callback

    @staticmethod
    def _restore_early_stopping_setup(
        trainer: Trainer, old_callback: Optional[EarlyStoppingCallback] = None
    ):
        """
        Restores the original early stopping callback in the trainer.

        Args:
            trainer (Trainer): The trainer object.

        Returns:
            Trainer: The restored trainer object.
        """
        patched_callback = trainer.callback_handler.pop_callback(
            EarlyStoppingCallbackPatched
        )
        if old_callback:
            trainer.add_callback(old_callback)
        elif patched_callback:
            trainer.add_callback(
                EarlyStoppingCallback(
                    patched_callback.early_stopping_patience,
                    patched_callback.early_stopping_threshold,
                )
            )
        return trainer

    def optimize_hyperparameters(
        self,
        trainer: Trainer,
        optimizables: Dict[str, OptimizableHyperParameter],
        num_trials: int,
        progress_reporter: Callable[[str, float], None] = lambda x, y: None,
    ):
        """
        Performs hyperparameter search using Optuna.

        Returns:
            dict: The best hyperparameters found by Optuna.
        """
        assert (
            not self.hyp_search_done
        ), "The hyperparameter search has already been done"
        trainer.hp_space = self._prepare_hp_space(**optimizables)
        trainer.hp_search_backend = HPSearchBackend.OPTUNA
        trainer.compute_objective = lambda x: x["eval_f1"]

        trainer.args, backup = self._prepare_args(trainer.args)
        trainer, early_stopping_callback = self._patch_or_add_early_stopping(trainer)

        def objective(trial):
            trainer.train(trial=trial)
            return trainer.state.best_metric

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=trainer.args.seed),
        )
        progress_callback = OptunaProgressReporter(progress_reporter, num_trials)
        study.enqueue_trial(trainer.args.hyp_search_defaults)
        study.optimize(objective, n_trials=num_trials, callbacks=[progress_callback])

        trainer.args = self._restore_args(backup, trainer.args)
        trainer = self._restore_early_stopping_setup(trainer, early_stopping_callback)
        return study.best_trial.params


class OptunaProgressReporter:
    """Optuna callback for reporting HPO progress.

    Reports progress as a fraction of completed trials to a progress reporter
    function. Used to integrate Optuna optimization with the library's
    progress reporting system.

    Attributes:
        progress_reporter: Callable taking (stage_name, progress_fraction).
        n_trials: Total number of trials planned.
        fraction_per_trial: Progress increment per completed trial.
    """

    def __init__(self, progress_reporter: Callable[[str, float], None], n_trials: int):
        self.progress_reporter = progress_reporter
        self.n_trials = n_trials
        self.fraction_per_trial = 1 / n_trials

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state.is_finished():
            self.progress_reporter(
                HPO_STAGE, (trial.number + 1) * self.fraction_per_trial
            )
