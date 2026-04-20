from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z0-9']+|[\u3400-\u9fff]")
NORMALIZE_RE = re.compile(r"[^a-z0-9\u3400-\u9fff']+")


def tokenize_text(text: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(text)]


def normalize_word(word: str) -> str:
    normalized = NORMALIZE_RE.sub("", word.lower()).strip("'")
    return normalized


def tokenize_sentence(sentence: str) -> list[str]:
    return [token for token in (normalize_word(word) for word in tokenize_text(sentence)) if token]

