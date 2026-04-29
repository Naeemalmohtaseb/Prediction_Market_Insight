"""Deterministic text similarity helpers."""

import re
from collections.abc import Iterable


def token_set(text: str) -> set[str]:
    """Return normalized alphanumeric tokens for a text string."""

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard_similarity(left: str, right: str) -> float:
    """Compute Jaccard similarity between two strings."""

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def max_similarity(query: str, candidates: Iterable[str]) -> float:
    """Return the maximum Jaccard similarity against candidate texts."""

    scores = [jaccard_similarity(query, candidate) for candidate in candidates if candidate]
    return max(scores, default=0.0)
