import json
from typing import List

from datasets import Dataset, DatasetDict

from .simple_data_provider import AbstractDataProvider
from .utils import convert_offsets_to_bio


class SimpleNerDataProvider(AbstractDataProvider):
    def _load_data(self) -> DatasetDict:
        splits = {}
        for split_name in ["train", "dev", "test"]:
            if path := getattr(self, f"{split_name}_path"):
                with open(path) as f:
                    records = json.load(f)
                data_set = []
                if "text" in records[0]:
                    if "spans" in records[0]:
                        for record in records:
                            tokens, bio_tags = convert_offsets_to_bio(
                                record["text"],
                                record["spans"],
                                language=self.language,
                            )
                            data_set.append(
                                {
                                    "text": record["text"],
                                    "tokens": tokens,
                                    "labels": bio_tags,
                                }
                            )
                    else:
                        for record in records:
                            tokens, bio_tags = self.from_brackets(record["text"])
                            data_set.append(
                                {
                                    "text": " ".join(tokens),
                                    "tokens": tokens,
                                    "labels": bio_tags,
                                }
                            )

                splits[split_name] = Dataset.from_list(data_set)
        return DatasetDict(splits)

    @staticmethod
    def from_brackets(text: str) -> List[str]:
        """
        For texts without any punctuation
        """
        # TODO: Rewrite using razdel tokenizer
        # to make it able to work with punctuation

        tags, tokens = [], []

        bio_mode = False
        cpt_bio = 0
        current_tag = None

        split_iter = iter(text.split(" "))

        for s in split_iter:
            if s.startswith("["):
                current_tag = s.strip("[")
                bio_mode = True
                cpt_bio += 1
                next(split_iter)
                continue

            elif s.endswith("]"):
                bio_mode = False
                if cpt_bio == 1:
                    prefix = "B-"
                else:
                    prefix = "I-"
                token = prefix + current_tag
                word = s.strip("]")
                current_tag = None
                cpt_bio = 0

            else:
                if bio_mode:
                    if cpt_bio == 1:
                        prefix = "B-"
                    else:
                        prefix = "I-"
                    token = prefix + current_tag
                    word = s
                    cpt_bio += 1
                else:
                    token = "O"
                    word = s

            tags.append(token)
            tokens.append(word)

        return tokens, tags
