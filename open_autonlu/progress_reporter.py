from typing import Callable, Optional

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .data_types import ProgressCallbackPayload


class ProgressReporter:
    """
    A class for reporting progress during training.

    Attributes:
        progress_callback (Callable[[ProgressCallbackPayload], None]): A callback function to report progress.
        _stages (Optional[list]): A list of stages in the training process.
        state (ProgressCallbackPayload): The current state of the training progress.

    Methods:
        __call__(stage_name: str, stage_progress: float): Updates and reports the progress based on the current stage and its progress.
    """

    def __init__(
        self,
        progress_callback: Optional[Callable[[ProgressCallbackPayload], None]] = None,
    ) -> None:
        self._stages: Optional[list] = None
        self.is_passthrough = False
        if progress_callback is None:
            self.is_passthrough = True
        self.progress_callback = progress_callback
        self.state = ProgressCallbackPayload("", 0)

    @property
    def stages(self):
        return self._stages

    @stages.setter
    def stages(self, value):
        """
        Setter method for the stages attribute.
        Updates the list of stages and calculates the percentage of progress per stage.

        Args:
            value (list): A list of stage names in the training process.
        """
        self._stages = value
        self._percent_per_stage = 1 / len(value)

    def __call__(self, stage_name: str, stage_progress: float):
        """
        Updates and reports the progress based on the current stage and its progress.

        Requirements:
            - The stages attribute must be set before calling this method.
            - stage_name (str): The name of the current stage.
            - stage_progress (float): The progress within the current stage, ranging from 0 to 1.
        """
        if self.is_passthrough:
            return
        assert self._stages is not None, "Stage name list is not set"
        self.state.stage_name = stage_name
        self.state.percent = (
            self._percent_per_stage * self._stages.index(stage_name)
        ) + self._percent_per_stage * stage_progress
        self.progress_callback(self.state)


class TrainerProgressReporterCallback(TrainerCallback):
    """
    A custom trainer callback for reporting progress during training.

    This class extends the TrainerCallback and provides a way to report progress at different stages of the training process.
    It can be used in conjunction with ProgressReporter to update the progress and trigger callbacks.

    Attributes:
        progress_reporter (ProgressReporter): The instance of ProgressReporter responsible for updating and reporting progress.
        stage (str): The current stage name being processed.

    Methods:
        on_train_begin(args: TrainingArguments, state: TrainerState, control: TrainerControl): Called at the beginning of training. Updates the progress reporter with initial stage and progress.
        on_train_end(args: TrainingArguments, state: TrainerState, control: TrainerControl): Called when training is complete. Updates the progress reporter with final stage and progress.
        on_step_end(args: TrainingArguments, state: TrainerState, control: TrainerControl): Called at the end of each step. Updates the progress reporter with current stage and progress.
    """

    def __init__(self, progress_reporter: ProgressReporter, stage: str) -> None:
        """
        Initializes the TrainerProgressReporterCallback.

        Args:
            progress_reporter (ProgressReporter): The instance of ProgressReporter responsible for updating and reporting progress.
            stage (str): The current stage name being processed.
        """
        self.progress_reporter = progress_reporter
        self.stage = stage

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the beginning of training.

        Updates the progress reporter with initial stage and progress.
        """
        self.progress_reporter(self.stage, 0.0)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called when training is complete.

        Updates the progress reporter with final stage and progress.
        """
        self.progress_reporter(self.stage, 1.0)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the end of each step.

        Updates the progress reporter with current stage and progress.
        """
        self.progress_reporter(self.stage, state.global_step / state.max_steps)
