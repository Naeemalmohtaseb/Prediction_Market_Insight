"""Question interpretation scaffold."""

import re

from pmi_agent.schemas import InterpretedQuestion


class QueryInterpreter:
    """Convert a user question into a structured forecasting query.

    This deterministic fallback can later be replaced or augmented by an LLM.
    The LLM must not produce probabilities or scores.
    """

    def interpret(self, question: str) -> InterpretedQuestion:
        """Interpret the user question using conservative deterministic parsing."""

        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("Question cannot be empty.")

        entities = _extract_entities(normalized)
        timeframe = _extract_timeframe(normalized)

        return InterpretedQuestion(
            original_question=question,
            normalized_question=normalized,
            core_event=normalized.rstrip("?"),
            timeframe=timeframe,
            entities=entities,
            search_terms=[normalized],
        )


def _extract_entities(text: str) -> list[str]:
    """Extract simple title-cased entity candidates."""

    candidates = re.findall(r"\b(?:[A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+)*)", text)
    seen: set[str] = set()
    entities: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            entities.append(candidate)
    return entities


def _extract_timeframe(text: str) -> str | None:
    """Extract a rough timeframe phrase if the question contains one."""

    patterns = [
        r"\bby\s+[^?.,;]+",
        r"\bbefore\s+[^?.,;]+",
        r"\bin\s+20\d{2}\b",
        r"\bthis\s+(?:year|month|week|quarter)\b",
        r"\bnext\s+(?:year|month|week|quarter)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None
