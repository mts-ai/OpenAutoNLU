from dataclasses import dataclass


@dataclass
class ProgressCallbackPayload:
    stage_name: str
    percent: float
