import re
import unicodedata
from typing import NamedTuple

from datasets import Dataset

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)

_TOKEN_RE = re.compile(r"\w+(?:-\w+)*|\S", re.UNICODE)


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_mark(char: str) -> bool:
    return unicodedata.category(char)[0] == "M"


class Token(NamedTuple):
    start: int
    stop: int
    text: str


def tokenize_with_offsets(text: str) -> list[Token]:
    """Tokenize text into words/punctuation and return character offsets.

    Uses a Unicode-aware regex. CJK characters are split individually since they have no whitespace word boundaries.
    Combining marks are merged back into the preceding token.
    """
    raw: list[Token] = []
    for m in _TOKEN_RE.finditer(text):
        word = m.group()
        offset = m.start()
        if len(word) > 1 and any(_is_cjk(c) for c in word):
            buf_start: int | None = None
            for i, ch in enumerate(word):
                if _is_cjk(ch):
                    if buf_start is not None:
                        raw.append(
                            Token(
                                start=offset + buf_start,
                                stop=offset + i,
                                text=word[buf_start:i],
                            )
                        )
                        buf_start = None
                    raw.append(Token(start=offset + i, stop=offset + i + 1, text=ch))
                else:
                    if buf_start is None:
                        buf_start = i
            if buf_start is not None:
                raw.append(
                    Token(
                        start=offset + buf_start,
                        stop=offset + len(word),
                        text=word[buf_start:],
                    )
                )
        else:
            raw.append(Token(start=offset, stop=m.end(), text=word))

    if not raw:
        return raw

    merged: list[Token] = [raw[0]]
    for tok in raw[1:]:
        prev = merged[-1]
        if tok.start == prev.stop and (
            _is_mark(tok.text[0]) or _is_mark(prev.text[-1])
        ):
            merged[-1] = Token(
                start=prev.start, stop=tok.stop, text=text[prev.start : tok.stop]
            )
        else:
            merged.append(tok)
    return merged


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
) -> tuple[list[str], list[str]]:
    substrings = tokenize_with_offsets(text)
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
