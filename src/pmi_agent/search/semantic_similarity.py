"""Deterministic lexical similarity helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "will",
    "with",
}


def token_set(text: str) -> set[str]:
    """Return normalized content tokens for a text string."""

    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def text_similarity(a: str, b: str) -> float:
    """Return deterministic 0-1 similarity for two text strings."""

    left = _normalize_text(a)
    right = _normalize_text(b)
    if not left or not right:
        return 0.0

    if fuzz is not None:
        token_score = fuzz.token_set_ratio(left, right) / 100.0
        partial_score = fuzz.partial_ratio(left, right) / 100.0
        return _clamp((0.75 * token_score) + (0.25 * partial_score))

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    jaccard = 0.0
    if left_tokens and right_tokens:
        jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    sequence = SequenceMatcher(None, left, right).ratio()
    return _clamp((0.70 * jaccard) + (0.30 * sequence))


def jaccard_similarity(left: str, right: str) -> float:
    """Compute token Jaccard similarity between two strings."""

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def max_similarity(query: str, candidates: Iterable[str]) -> float:
    """Return the maximum text similarity against candidate strings."""

    scores = [text_similarity(query, candidate) for candidate in candidates if candidate]
    return max(scores, default=0.0)


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
