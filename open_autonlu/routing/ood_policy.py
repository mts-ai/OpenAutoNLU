"""Map pipeline OOD settings to routing OOD policies."""

from __future__ import annotations

from ..methods.data_types import OodMethod
from .task_spec import OOD_POLICY_DETECTOR, OOD_POLICY_LOGIT_CLASS, OOD_POLICY_NONE


def ood_method_to_policy(ood_method: OodMethod) -> str:
    if ood_method in (OodMethod.NONE,):
        return OOD_POLICY_NONE
    if ood_method in (OodMethod.LOGIT,):
        return OOD_POLICY_LOGIT_CLASS
    return OOD_POLICY_DETECTOR
