"""Search term expansion for market discovery."""

import re

from pmi_agent.schemas import InterpretedQuestion

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
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
}


class SearchTermExpander:
    """Create deterministic search terms from an interpreted question."""

    def expand(self, interpreted_question: InterpretedQuestion) -> list[str]:
        """Return ordered search terms without using probabilities or scores."""

        terms = [interpreted_question.normalized_question]
        terms.extend(interpreted_question.entities)

        keywords = _keywords(interpreted_question.core_event)
        if keywords:
            terms.append(" ".join(keywords[:8]))

        return _dedupe([term for term in terms if term.strip()])


def _keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped
