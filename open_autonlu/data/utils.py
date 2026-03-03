from typing import NamedTuple

import spacy
from datasets import Dataset

SUPPORTED_LANGUAGES = ("ru", "en")

_spacy_models: dict[str, spacy.language.Language] = {}


def _get_spacy_model(language: str) -> spacy.language.Language:
    if language not in _spacy_models:
        _spacy_models[language] = spacy.blank(language)
    return _spacy_models[language]


class Token(NamedTuple):
    start: int
    stop: int
    text: str


def tokenize_with_offsets(text: str, language: str = "en") -> list[Token]:
    """Tokenize text and return list of (start, stop, text) for each token."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{language}'. Supported: {SUPPORTED_LANGUAGES}"
        )

    nlp = _get_spacy_model(language)
    doc = nlp(text)
    return [
        Token(
            start=token.idx if token.idx is not None else 0,
            stop=token.idx + len(token.text)
            if token.idx is not None
            else len(token.text),
            text=token.text,
        )
        for token in doc
    ]


def align_labels_with_tokens(labels, word_ids, b_to_i_label, label_all_tokens=True):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label_all_tokens:
                new_labels.append(b_to_i_label[label])
            else:
                new_labels.append(-100)

    return new_labels


def convert_offsets_to_bio(
    text: str,
    labels: list,
    label_key: str = "label",
    language: str = "en",
) -> tuple[list[str], list[str]]:
    substrings = tokenize_with_offsets(text, language=language)
    tokens: list[str] = []
    bio_tags: list[str] = []

    def find_entity_for_token(t_start: int, t_end: int):
        for ent in labels:
            ent_start = ent["start"]
            ent_end = ent["end"]
            if not (t_end <= ent_start or t_start >= ent_end):
                return ent
        return None

    last_label = None

    for token in substrings:
        start, end = token.start, token.stop
        tokens.append(token.text)

        ent = find_entity_for_token(start, end)
        if ent is None:
            bio_tags.append("O")
            last_label = None
            continue

        label = ent[label_key]
        prefix = "I-" if last_label == label else "B-"
        bio_tags.append(prefix + label)
        last_label = label

    return tokens, bio_tags


def convert_bio_to_spans(
    text: str,
    tokens: list[str],
    bio_tags: list[str],
) -> list[dict]:
    """Convert token-level BIO tags back to character-offset spans"""
    if not tokens or not bio_tags or len(tokens) != len(bio_tags):
        return []

    pos = 0
    token_positions = []
    for t in tokens:
        idx = text.find(t, pos)
        if idx == -1:
            idx = text.find(t)
        if idx == -1:
            return []
        token_positions.append(Token(start=idx, stop=idx + len(t), text=t))
        pos = idx + len(t)

    spans = []
    i = 0
    while i < len(bio_tags):
        tag = bio_tags[i] if isinstance(bio_tags[i], str) else str(bio_tags[i])
        if tag.startswith("B-"):
            label = tag[2:]
            start = token_positions[i].start
            j = i + 1
            while j < len(bio_tags):
                next_tag = (
                    bio_tags[j] if isinstance(bio_tags[j], str) else str(bio_tags[j])
                )
                if next_tag != f"I-{label}":
                    break
                j += 1
            end = token_positions[j - 1].stop
            spans.append({"start": start, "end": end, "label": label})
            i = j
        else:
            i += 1
    return spans


def anc_label_is_present(dataset: Dataset) -> bool:
    return "anc_label" in dataset.column_names


def pop_default(input_list, index=-1, default_value=None):
    try:
        return input_list.pop(index)
    except IndexError:
        return default_value
