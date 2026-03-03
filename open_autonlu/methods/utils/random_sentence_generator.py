from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass

# taken from https://foliant-docs.github.io/docs/tutorials/preprocessor/generator/ and modified to be deterministic


@dataclass
class GibberishDatasetConfig:
    num_words_per_sentence: Tuple[int, int] = (1, 15)
    num_characters_per_word: Tuple[int, int] = (2, 9)
    seed: int = 0
    rare_letters: str = "ьъы"
    vowels: str = "аеёиоуыэюя"
    consonants: str = "бвгджзйклмнпрстфхцчшщ"


@dataclass
class GibberishDatasetConfigEN:
    num_words_per_sentence: Tuple[int, int] = (1, 15)
    num_characters_per_word: Tuple[int, int] = (2, 9)
    seed: int = 0
    rare_letters: str = "qzx"
    vowels: str = "aeiou"
    consonants: str = "bcdfghjklmnpqrstvwxyz"


class GibberishDatasetGenerator:
    def __init__(self, seed=0, config: Optional[GibberishDatasetConfig] = None):
        if config is not None:
            self.config = config
        else:
            self.config = GibberishDatasetConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(self, num_rows: int):
        return [
            self._gen_sentence(
                num_words=self.rng.integers(*self.config.num_words_per_sentence)
            )
            for _ in range(num_rows)
        ]

    def _gen_sentence(self, num_words=7):
        words = (self._gen_word() for _ in range(num_words))
        return " ".join(words)

    def _gen_word(self):
        word_len = self.rng.integers(*self.config.num_characters_per_word)
        return "".join(self._pick_letter() for _ in range(word_len))

    def _pick_letter(self):
        pick = self.rng.random()
        if pick > 0.9:
            idx = self.rng.choice(len(self.config.rare_letters))
            return self.config.rare_letters[idx]
        elif pick > 0.25:
            idx = self.rng.choice(len(self.config.vowels))
            return self.config.vowels[idx]
        else:
            idx = self.rng.choice(len(self.config.consonants))
            return self.config.consonants[idx]


class GibberishDatasetGeneratorEN:
    def __init__(self, seed=0, config: Optional[GibberishDatasetConfigEN] = None):
        if config is not None:
            self.config = config
        else:
            self.config = GibberishDatasetConfigEN()
        self.rng = np.random.default_rng(seed)

    def __call__(self, num_rows: int):
        return [
            self._gen_sentence(
                num_words=self.rng.integers(*self.config.num_words_per_sentence)
            )
            for _ in range(num_rows)
        ]

    def _gen_sentence(self, num_words=7):
        words = (self._gen_word() for _ in range(num_words))
        return " ".join(words)

    def _gen_word(self):
        word_len = self.rng.integers(*self.config.num_characters_per_word)
        return "".join(self._pick_letter() for _ in range(word_len))

    def _pick_letter(self):
        pick = self.rng.random()
        if pick > 0.9:
            idx = self.rng.choice(len(self.config.rare_letters))
            return self.config.rare_letters[idx]
        elif pick > 0.25:
            idx = self.rng.choice(len(self.config.vowels))
            return self.config.vowels[idx]
        else:
            idx = self.rng.choice(len(self.config.consonants))
            return self.config.consonants[idx]


if __name__ == "__main__":
    ds_generator = GibberishDatasetGenerator(seed=15)
    ds = ds_generator(5)
    print(ds)
