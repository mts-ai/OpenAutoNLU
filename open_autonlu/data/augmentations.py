import re
from augmentex import WordAug, CharAug
import random
import numpy as np


class WordCharAugmentation:
    """Text augmentation class that applies character-level and word-level corruptions.

    Attributes:
        char_actions: List of character-level augmentation actions.
        word_actions: List of word-level augmentation actions.
        seed: Random seed for reproducibility (default: 42).
        word_aug: WordAug instance from augmentex.
        char_aug: CharAug instance from augmentex.

    Example:
        >>> aug = WordCharAugmentation()
        >>> corruptions = aug("Привет мир")
        >>> print(corruptions)
        ['Привет мри', 'Прривет мир', 'Првет мир', ...]
    """

    def __init__(self, *args, **kwargs):
        self.char_actions = ["orfo", "typo", "delete", "multiply", "swap", "insert"]
        self.word_actions = ["replace", "swap"]
        self.seed = 42
        self.word_aug = WordAug(
            unit_prob=1.0,
            # Percentage of the phrase to which augmentations will be applied
            min_aug=1,  # Minimum number of augmentations
            max_aug=2,  # Maximum number of augmentations
            # following arguments are supported in augmentex 1.3.1
            lang="rus",  # supports: "rus", "eng"
            platform="pc",  # supports: "pc", "mobile"
            random_seed=self.seed,
        )
        self.char_aug = CharAug(
            unit_prob=1.0,  # Percentage of the phrase to which augmentations will be applied
            min_aug=1,  # Minimum number of augmentations
            max_aug=2,  # Maximum number of augmentations
            lang="rus",  # supports: "rus", "eng"
            platform="pc",  # supports: "pc", "mobile"
            random_seed=self.seed,
        )
        self._set_seed(self.seed)

    @staticmethod
    def _clean_for_aug(x: str) -> str:
        return re.sub("[^\\sа-яА-ЯёЁa-zA-Z0-9'.,!?;\\]\\[]", "", x.replace("-", " "))

    def _set_seed(self, seed: int = None):
        if seed is None:
            seed = 42
        random.seed(seed)
        np.random.seed(seed)

    def __call__(self, text: str) -> list[str]:
        corruptions = []
        for action in self.char_actions:
            try:
                self._set_seed()
                aug_text = self.char_aug.augment(text, action=action)
                corruptions.append(aug_text)
            except Exception as e:
                print(f"Augmentation '{action}' failed on text '{text}': {e}")
        for action in self.word_actions:
            try:
                cleaned_text = self._clean_for_aug(text)
                self._set_seed()
                aug_text = self.word_aug.augment(cleaned_text, action=action)
                corruptions.append(aug_text)
            except Exception as e:
                print(f"Augmentation '{action}' failed on text '{text}': {e}")

        corruptions = sorted(list(set(corruptions)))
        return corruptions
