"""Search term expansion for market discovery."""

import re

from pmi_agent.schemas import InterpretedQuestion

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "before",
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


class SearchTermExpander:
    """Create focused deterministic search terms from an interpreted question."""

    def expand(self, question: InterpretedQuestion) -> list[str]:
        """Return 8 to 15 useful search terms where possible."""

        terms: list[str] = [
            question.original_question,
            question.normalized_question,
            question.target_event,
            question.expected_outcome,
        ]
        terms.extend(question.entities)
        terms.extend(question.related_concepts)

        if question.timeframe:
            terms.append(question.timeframe)
            for entity in question.entities[:3]:
                terms.append(f"{entity} {question.timeframe}")

        terms.extend(_compact_phrases(question))
        return _dedupe(terms)[:15]


def _compact_phrases(question: InterpretedQuestion) -> list[str]:
    text = " ".join(
        [
            question.target_event,
            question.expected_outcome,
            " ".join(question.entities),
            " ".join(question.related_concepts),
        ]
    )
    keywords = _keywords(text)
    phrases: list[str] = []

    if "Fed" in question.entities or "Federal Reserve" in question.entities:
        phrases.extend(["Fed cut rates", "interest rates", "rate cut", "FOMC"])
        if question.timeframe:
            phrases.append(f"Fed {question.timeframe}")
    if "gas" in question.normalized_question.lower():
        phrases.extend(["gas prices", "gasoline", "oil prices", "inflation", "OPEC"])
        if question.timeframe:
            phrases.append(f"{question.timeframe} gas prices")
    if "OpenAI" in question.entities:
        phrases.extend(["OpenAI IPO", "AI IPO", "OpenAI public listing"])
    if "Air Jordans" in question.entities:
        phrases.extend(["Air Jordans release", "Nike Jordan release", "new Jordans"])
    if "Iran" in question.entities:
        phrases.extend(["U.S. Iran conflict", "Iran war", "Middle East conflict"])
    if "Trump" in question.entities:
        phrases.extend(["Trump impeachment", "Donald Trump impeached", "Congress impeachment"])

    if keywords:
        phrases.append(" ".join(keywords[:6]))
    return phrases


def _keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        clean = " ".join(str(value).strip().split())
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped
