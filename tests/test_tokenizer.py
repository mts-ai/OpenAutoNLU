import json
from pathlib import Path

import pytest

from open_autonlu.data.ner_data_provider import SimpleNerDataProvider
from open_autonlu.data.utils import tokenize_with_offsets

_DATA_DIR = Path(__file__).parent / "data"
_provider = SimpleNerDataProvider(train_path="unused.json")


def _load_ner_texts() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in sorted(_DATA_DIR.glob("*_ner/*.json")):
        lang = path.parent.name.removesuffix("_ner")
        split = path.stem
        with open(path, encoding="utf-8") as f:
            for i, rec in enumerate(json.load(f)):
                if isinstance(rec, dict):
                    parsed = _provider._parse_record(rec)
                    out.append((f"{lang}_{split}_{i}", parsed["text"]))
    return out


_NER_TEXTS = _load_ner_texts()


# We assert:
# 1. text[tok.start:tok.stop] == tok.text,
# 2. no overlapping tokens,
# 3. every non-whitespace character is in exactly one token.


class TestTokenizeBasicContract:
    @pytest.mark.parametrize(
        "text",
        [""] + [t for _, t in _NER_TEXTS],
        ids=["empty"] + [tid for tid, _ in _NER_TEXTS],
    )
    def test_offset_consistency(self, text: str):
        tokens = tokenize_with_offsets(text)
        for tok in tokens:
            assert text[tok.start : tok.stop] == tok.text

    @pytest.mark.parametrize(
        "text", [t for _, t in _NER_TEXTS], ids=[tid for tid, _ in _NER_TEXTS]
    )
    def test_no_overlap(self, text: str):
        tokens = tokenize_with_offsets(text)
        for a, b in zip(tokens, tokens[1:]):
            assert a.stop <= b.start

    @pytest.mark.parametrize(
        "text", [t for _, t in _NER_TEXTS], ids=[tid for tid, _ in _NER_TEXTS]
    )
    def test_full_coverage(self, text: str):
        tokens = tokenize_with_offsets(text)
        for tok in tokens:
            assert 0 <= tok.start <= tok.stop <= len(text)
        covered = set()
        for tok in tokens:
            covered.update(range(tok.start, tok.stop))
        for i, c in enumerate(text):
            if not c.isspace():
                assert i in covered, f"Non-whitespace at {i!r} not in any token"
