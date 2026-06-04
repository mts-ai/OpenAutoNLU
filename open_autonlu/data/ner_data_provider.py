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
                splits[split_name] = Dataset.from_list(
                    [self._parse_record(r) for r in records]
                )
        return DatasetDict(splits)

    def _parse_record(self, record: dict) -> dict:
        if "tokens" in record and "labels" in record:
            tokens, bio_tags = record["tokens"], record["labels"]
            text = record.get("text", " ".join(tokens))
        elif "text" in record and "spans" in record:
            tokens, bio_tags = convert_offsets_to_bio(record["text"], record["spans"])
            text = record["text"]
        elif "text" in record:
            tokens, bio_tags = self.from_brackets(record["text"])
            text = " ".join(tokens)
        else:
            raise ValueError(
                "Record must contain either 'tokens'+'labels', "
                "'text'+'spans', or bracket format."
            )
        return {"text": text, "tokens": tokens, "labels": bio_tags}

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
