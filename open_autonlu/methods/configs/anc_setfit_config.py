from dataclasses import dataclass
from typing import Optional

from ..constants import DEFAULT_TEMPLATE_BY_LANGUAGE
from .setfit_config import SetFitMethodConfig


@dataclass
class AncSetFitConfig(SetFitMethodConfig):
    """Configuration for Anchor-based SetFit (AncSetFit) method.

    AncSetFit extends SetFit with anchor-based contrastive learning using triplet loss.
    Instead of comparing sentence pairs, it uses anchor templates that combine a prompt
    template with human-readable class descriptions (anc_label).

    The anc_label is a human-readable description of the class that gets concatenated
    with the template to form an anchor. For example, if template is
    "User asks the bot to perform a request using the skill: " and anc_label is
    "weather forecast", the anchor becomes "User asks the bot to perform a request
    using the skill: weather forecast". This anchor is then used in triplet loss
    training where:
    - Anchor: template + anc_label (class description)
    - Positive: sentence belonging to that class
    - Negative: sentence from a different class

    The anc_label column must be present in the training dataset. If absent,
    the pipeline will fall back to standard SetFit.

    Attributes:
        margin: Margin for triplet loss. Controls the minimum distance between
            positive and negative pairs.
        add_normalization_layer: Whether to add L2 normalization to embeddings.
        batch_size: Training batch size for triplet learning.
        epochs: Number of training epochs.
        body_lr: Learning rate for the sentence transformer body.
        template: Prompt template prepended to anc_label to form anchors.
            If None, resolved from language by DEFAULT_TEMPLATE_BY_LANGUAGE.
    """

    margin: float = 0.25
    add_normalization_layer: bool = False
    batch_size: int = 64
    epochs: int = 1
    body_lr: float = 1e-4
    template: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.template is None:
            self.template = DEFAULT_TEMPLATE_BY_LANGUAGE[self.language]
