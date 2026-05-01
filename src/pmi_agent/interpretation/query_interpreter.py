"""Forecast question interpretation."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import ValidationError

from pmi_agent.schemas import InterpretedQuestion, QuestionCategory

logger = logging.getLogger(__name__)

CATEGORIES: set[str] = {
    "politics",
    "macroeconomics",
    "finance",
    "technology",
    "consumer_products",
    "entertainment",
    "geopolitical_risk",
    "sports",
    "other",
}

KNOWN_ENTITY_PATTERNS = {
    "Fed": ["fed", "federal reserve", "fomc"],
    "Federal Reserve": ["federal reserve"],
    "FOMC": ["fomc"],
    "OpenAI": ["openai"],
    "Air Jordans": ["air jordans", "jordans"],
    "U.S.": ["u.s.", "us ", "united states", "america"],
    "Iran": ["iran"],
    "Trump": ["trump", "donald trump"],
    "OPEC": ["opec"],
}


class QueryInterpreter:
    """Convert a natural-language question into a structured forecast event."""

    def interpret(self, question: str) -> InterpretedQuestion:
        """Interpret a question using optional LLM parsing with deterministic fallback."""

        normalized = _normalize_question(question)
        if not normalized:
            raise ValueError("Question cannot be empty.")

        llm_result = self._interpret_with_llm(normalized, original_question=question)
        if llm_result is not None:
            return _finalize(llm_result)

        return self._interpret_deterministically(question)

    def _interpret_with_llm(
        self,
        normalized_question: str,
        original_question: str,
    ) -> InterpretedQuestion | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            from openai import OpenAI
        except ImportError:
            return None

        system_prompt = (
            "You interpret current-world forecast questions for a market-implied "
            "forecasting dashboard. Do not estimate probability. Do not provide "
            "betting or trading advice. Do not invent market data. Return only JSON "
            "matching the InterpretedQuestion schema. Categories must be one of: "
            f"{', '.join(sorted(CATEGORIES))}."
        )
        user_prompt = (
            "Return JSON with keys: original_question, normalized_question, category, "
            "target_event, expected_outcome, entities, geography, timeframe, "
            "resolution_criteria, search_terms, related_concepts.\n\n"
            f"Question: {normalized_question}"
        )

        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.parse(
                model=os.getenv("PMI_OPENAI_INTERPRETER_MODEL", "gpt-4o-mini"),
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=InterpretedQuestion,
            )
            parsed = getattr(response, "output_parsed", None)
            if isinstance(parsed, InterpretedQuestion):
                return parsed
            output_text = getattr(response, "output_text", None)
            if output_text:
                return InterpretedQuestion.model_validate_json(output_text)
        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            logger.info("LLM question interpretation failed; using deterministic fallback: %s", exc)
            return None

        return None

    def _interpret_deterministically(self, question: str) -> InterpretedQuestion:
        normalized = _normalize_question(question)
        lowered = normalized.lower()
        entities = _extract_entities(normalized)
        category = _infer_category(lowered)
        timeframe = _extract_timeframe(normalized)
        geography = _infer_geography(lowered, entities)
        expected_outcome = _infer_expected_outcome(lowered)
        target_event = _infer_target_event(normalized, expected_outcome)
        related_concepts = _related_concepts(lowered, category, entities)
        search_terms = _base_search_terms(
            normalized,
            target_event,
            expected_outcome,
            entities,
            related_concepts,
            timeframe,
        )

        return _finalize(
            InterpretedQuestion(
                original_question=question,
                normalized_question=normalized,
                category=category,
                target_event=target_event,
                expected_outcome=expected_outcome,
                entities=entities,
                geography=geography,
                timeframe=timeframe,
                resolution_criteria=_resolution_criteria(target_event, expected_outcome, timeframe),
                search_terms=search_terms,
                related_concepts=related_concepts,
            )
        )


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().split())


def _finalize(question: InterpretedQuestion) -> InterpretedQuestion:
    data = question.model_dump()
    data["normalized_question"] = _normalize_question(data["normalized_question"])
    data["category"] = data["category"] if data["category"] in CATEGORIES else "other"
    data["entities"] = _dedupe([item for item in data["entities"] if item])
    data["related_concepts"] = _dedupe([item for item in data["related_concepts"] if item])
    data["search_terms"] = _dedupe([item for item in data["search_terms"] if item])[:15]
    if not data["search_terms"]:
        data["search_terms"] = [data["normalized_question"]]
    return InterpretedQuestion.model_validate(data)


def _infer_category(lowered: str) -> QuestionCategory:
    if any(term in lowered for term in ("fed", "federal reserve", "interest rate", "inflation", "gas price", "oil price", "opec")):
        return "macroeconomics"
    if any(term in lowered for term in ("ipo", "stock", "earnings", "bitcoin", "crypto")):
        return "finance"
    if any(term in lowered for term in ("openai", "ai ", "artificial intelligence", "semiconductor")):
        return "technology"
    if any(term in lowered for term in ("air jordan", "jordans", "iphone", "product release", "sneaker")):
        return "consumer_products"
    if any(term in lowered for term in ("iran", "war", "conflict", "invasion", "military")):
        return "geopolitical_risk"
    if any(term in lowered for term in ("trump", "impeach", "election", "senate", "congress", "president")):
        return "politics"
    if any(term in lowered for term in ("movie", "album", "oscar", "grammy", "box office")):
        return "entertainment"
    if any(term in lowered for term in ("nba", "nfl", "mlb", "nhl", "world cup", "super bowl")):
        return "sports"
    return "other"


def _extract_entities(text: str) -> list[str]:
    lowered = text.lower() + " "
    entities: list[str] = []
    for canonical, patterns in KNOWN_ENTITY_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            entities.append(canonical)

    title_candidates = re.findall(
        r"\b(?:[A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+)*)",
        text,
    )
    ignored = {
        "Will",
        "Does",
        "Do",
        "Can",
        "Could",
        "Would",
        "Is",
        "Are",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
    for candidate in title_candidates:
        clean = candidate.rstrip("?")
        first_word = clean.split()[0] if clean.split() else clean
        if clean in ignored or first_word in {"Will", "Does", "Do", "Can", "Could", "Would", "Is", "Are"}:
            continue
        entities.append(clean)
    return _dedupe(entities)


def _extract_timeframe(text: str) -> str | None:
    patterns = [
        r"\bby\s+[^?.,;]+",
        r"\bbefore\s+[^?.,;]+",
        r"\bin\s+20\d{2}\b",
        r"\bthrough\s+20\d{2}\b",
        r"\bthis\s+(?:year|summer|winter|spring|fall|month|week|quarter)\b",
        r"\bnext\s+(?:year|summer|winter|spring|fall|month|week|quarter)\b",
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\bsummer\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _infer_geography(lowered: str, entities: list[str]) -> str | None:
    if "Iran" in entities:
        return "Iran"
    if "U.S." in entities or any(term in lowered for term in ("fed", "federal reserve", "trump", "u.s.", "united states")):
        return "United States"
    if "opec" in lowered or "oil prices" in lowered:
        return "Global"
    return None


def _infer_expected_outcome(lowered: str) -> str:
    if any(term in lowered for term in ("cut rates", "rate cut", "decrease interest", "lower rates")):
        return "interest rates are cut"
    if any(term in lowered for term in ("gas prices rise", "gas price rise", "rise this summer", "increase gas")):
        return "gas prices rise"
    if "ipo" in lowered:
        return "company completes or announces IPO"
    if any(term in lowered for term in ("release", "launch", "come out")):
        return "product is released"
    if any(term in lowered for term in ("enter a conflict", "enter conflict", "conflict with", "war with")):
        return "military conflict occurs"
    if "impeach" in lowered:
        return "impeachment occurs"
    if any(term in lowered for term in ("win", "wins")):
        return "specified win occurs"
    if any(term in lowered for term in ("rise", "increase")):
        return "specified increase occurs"
    if any(term in lowered for term in ("fall", "decrease", "drop")):
        return "specified decrease occurs"
    return "event occurs"


def _infer_target_event(normalized: str, expected_outcome: str) -> str:
    text = normalized.rstrip("?")
    text = re.sub(r"^(will|does|do|can|could|would|is|are)\s+", "", text, flags=re.IGNORECASE)
    if expected_outcome == "interest rates are cut" and re.search(r"\bfed\b|\bfederal reserve\b", normalized, re.IGNORECASE):
        return "Federal Reserve cuts interest rates"
    if expected_outcome == "gas prices rise":
        return "gas prices rise"
    if "openai" in normalized.lower() and "ipo" in normalized.lower():
        return "OpenAI IPO"
    if "air jordan" in normalized.lower() or "jordans" in normalized.lower():
        return "new Air Jordans release"
    if "iran" in normalized.lower() and re.search(r"\bu\.s\.|\bus\b|united states", normalized, re.IGNORECASE):
        return "U.S. enters military conflict with Iran"
    if "trump" in normalized.lower() and "impeach" in normalized.lower():
        return "Trump is impeached"
    return text


def _related_concepts(lowered: str, category: QuestionCategory, entities: list[str]) -> list[str]:
    concepts: list[str] = []
    if "Fed" in entities or "Federal Reserve" in entities or "fomc" in lowered:
        concepts.extend(["interest rates", "rate cut", "FOMC", "monetary policy"])
    if "gas price" in lowered:
        concepts.extend(["gasoline", "oil prices", "inflation", "OPEC", "summer gas prices"])
    if "OpenAI" in entities:
        concepts.extend(["AI", "artificial intelligence", "IPO", "private company"])
    if "Air Jordans" in entities:
        concepts.extend(["Nike", "sneakers", "Jordan release", "consumer products"])
    if "Iran" in entities:
        concepts.extend(["Middle East", "military conflict", "geopolitical risk", "war"])
    if "Trump" in entities:
        concepts.extend(["Donald Trump", "impeachment", "Congress", "presidency"])
    if category == "finance":
        concepts.extend(["markets", "public listing"])
    return _dedupe(concepts)


def _base_search_terms(
    normalized: str,
    target_event: str,
    expected_outcome: str,
    entities: list[str],
    related_concepts: list[str],
    timeframe: str | None,
) -> list[str]:
    terms = [normalized, target_event, expected_outcome]
    terms.extend(entities)
    terms.extend(related_concepts)
    if timeframe:
        terms.append(timeframe)
        if entities:
            terms.append(f"{entities[0]} {timeframe}")
    return _dedupe(terms)


def _resolution_criteria(target_event: str, expected_outcome: str, timeframe: str | None) -> str:
    timeframe_text = f" within {timeframe}" if timeframe else ""
    return f"Resolves based on whether {expected_outcome} for {target_event}{timeframe_text}."


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
