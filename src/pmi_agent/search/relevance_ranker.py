"""Deterministic market relevance ranking."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pmi_agent.schemas import EvidenceType, InterpretedQuestion, NormalizedMarket, QuestionCategory, RankedMarket
from pmi_agent.search.semantic_similarity import max_similarity, text_similarity, token_set


class RelevanceRanker:
    """Rank normalized markets using deterministic evidence signals."""

    def rank(
        self,
        question: InterpretedQuestion,
        markets: list[NormalizedMarket],
        min_score: float = 0.0,
    ) -> list[RankedMarket]:
        """Return markets sorted by deterministic relevance."""

        ranked = [self._rank_one(question, market) for market in markets]
        filtered = [market for market in ranked if market.relevance_score >= min_score]
        return sorted(filtered, key=lambda item: item.relevance_score, reverse=True)

    def _rank_one(self, question: InterpretedQuestion, market: NormalizedMarket) -> RankedMarket:
        market_text = _market_text(market)
        query_texts = [
            question.normalized_question,
            question.target_event,
            question.expected_outcome,
            question.resolution_criteria or "",
            " ".join(question.entities),
            " ".join(question.related_concepts),
        ]

        market_candidates = [market.title, market.description or "", market_text]
        semantic = max(max_similarity(query_text, market_candidates) for query_text in query_texts if query_text)
        entity = _entity_overlap(question, market_text)
        timeframe, timeframe_conflict = _timeframe_alignment(question.timeframe, market)
        outcome, outcome_conflict = _outcome_alignment(question.expected_outcome, market_text)
        category = _category_alignment(question.category, market_text)
        clarity = _resolution_clarity(market)
        quality = _market_quality(market)
        is_conditional, conditional_reasons = detect_conditional_market(market.title)
        is_compound, compound_reasons = detect_compound_market(market.title, question)
        has_scope_mismatch, scope_reasons = detect_scope_mismatch(question, market)
        detected_timeframe_conflict, timeframe_reasons = detect_timeframe_conflict(question, market_text)
        has_entity_conflict, entity_reasons = detect_entity_conflict(question, market.title)
        has_timeframe_conflict = timeframe_conflict or detected_timeframe_conflict

        relevance = (
            0.30 * semantic
            + 0.20 * entity
            + 0.15 * outcome
            + 0.15 * timeframe
            + 0.10 * category
            + 0.05 * clarity
            + 0.05 * quality
        )

        caps: list[str] = []
        penalty_reasons: list[str] = []
        condition_in_question = _condition_present_in_question(question, market_text)
        suppress_condition_mismatch = (is_conditional or is_compound) and condition_in_question
        effective_entity_conflict = has_entity_conflict and not suppress_condition_mismatch
        effective_scope_mismatch = has_scope_mismatch and not suppress_condition_mismatch
        if (is_conditional or is_compound) and not condition_in_question:
            relevance = min(relevance, 0.45)
            penalty_reasons.append("Conditional or compound market does not directly resolve the user question.")
            caps.append("conditional/compound cap")
        elif is_conditional or is_compound:
            penalty_reasons.extend(conditional_reasons + compound_reasons)
        if outcome_conflict:
            relevance = min(relevance, 0.40)
            caps.append("outcome conflict cap")
            penalty_reasons.append("Market outcome appears to conflict with the expected outcome.")
        if has_timeframe_conflict:
            relevance = min(relevance, 0.50)
            caps.append("timeframe conflict cap")
            penalty_reasons.append("Market timeframe differs from the user's timeframe.")
        if effective_entity_conflict:
            relevance = min(relevance, 0.40)
            caps.append("entity conflict cap")
            penalty_reasons.append("Market centers on a different entity or sub-event.")
        if effective_scope_mismatch:
            relevance = min(relevance, 0.45)
            caps.append("scope mismatch cap")
            penalty_reasons.append("Market scope is narrower or different than the interpreted event.")
        if entity <= 0.0:
            relevance = min(relevance, 0.249)
            caps.append("no shared entity/concept cap")
            penalty_reasons.append("No shared core event terms found.")
        if question.entities and not _has_entity_phrase(question.entities, market_text):
            relevance = min(relevance, 0.249)
            caps.append("no shared named entity cap")
            penalty_reasons.append("No shared core event terms found.")
        if not _has_shared_core_event_terms(question, market_text):
            relevance = min(relevance, 0.249)
            caps.append("no shared core event cap")
            penalty_reasons.append("No shared core event terms found.")
        if (market.closed is True or market.active is False) and not _question_is_historical(question):
            relevance = min(relevance, 0.30)
            caps.append("inactive/closed cap")
            penalty_reasons.append("Inactive or closed market.")

        penalty_reasons.extend(conditional_reasons + compound_reasons + timeframe_reasons)
        if effective_scope_mismatch:
            penalty_reasons.extend(scope_reasons)
        if effective_entity_conflict:
            penalty_reasons.extend(entity_reasons)
        penalty_reasons = _dedupe(penalty_reasons)

        relevance = _clamp(relevance)
        evidence_type = _classify_evidence(
            relevance,
            has_major_mismatch=(
                ((is_conditional or is_compound) and not condition_in_question)
                or effective_scope_mismatch
                or effective_entity_conflict
                or has_timeframe_conflict
            ),
        )
        rationale = _rationale(
            semantic,
            entity,
            timeframe,
            outcome,
            category,
            clarity,
            quality,
            caps,
            penalty_reasons,
        )

        return RankedMarket(
            market=market,
            relevance_score=relevance,
            evidence_type=evidence_type,
            semantic_similarity=semantic,
            entity_overlap=entity,
            timeframe_alignment=timeframe,
            outcome_alignment=outcome,
            category_alignment=category,
            resolution_clarity=clarity,
            market_quality=quality,
            rationale=rationale,
            directness_score=entity,
            liquidity_score=quality,
            confidence_score=relevance,
            rank_reasons=[rationale],
            is_conditional=is_conditional,
            is_compound=is_compound,
            has_scope_mismatch=effective_scope_mismatch,
            has_timeframe_conflict=has_timeframe_conflict,
            has_entity_conflict=effective_entity_conflict,
            penalty_reasons=penalty_reasons,
        )


def _market_text(market: NormalizedMarket) -> str:
    return " ".join(
        item
        for item in [
            market.title,
            market.description or "",
            market.slug or "",
            " ".join(market.outcome_prices.keys()),
        ]
        if item
    )


def _entity_overlap(question: InterpretedQuestion, market_text: str) -> float:
    candidates = question.entities + question.related_concepts + question.search_terms
    clean_candidates = [candidate for candidate in candidates if len(candidate.strip()) > 2]
    if not clean_candidates:
        return 0.0

    market_lower = market_text.lower()
    phrase_hits = sum(1 for candidate in clean_candidates if candidate.lower() in market_lower)
    phrase_score = phrase_hits / len(clean_candidates)

    candidate_tokens = set()
    for candidate in clean_candidates:
        candidate_tokens.update(token_set(candidate))
    market_tokens = token_set(market_text)
    token_score = 0.0
    if candidate_tokens:
        token_score = len(candidate_tokens & market_tokens) / len(candidate_tokens)

    return _clamp(max(phrase_score, token_score))


def _has_entity_phrase(entities: list[str], market_text: str) -> bool:
    market_lower = market_text.lower()
    aliases = {
        "fed": ["fed", "federal reserve", "fomc"],
        "u.s.": ["u.s.", "us", "united states", "america"],
        "trump": ["trump", "donald trump"],
    }
    for entity in entities:
        clean = entity.strip().lower()
        if len(clean) <= 1:
            continue
        candidates = aliases.get(clean, [clean])
        for candidate in candidates:
            if candidate == "us":
                if re.search(r"\bus\b", market_lower):
                    return True
                continue
            if candidate in market_lower:
                return True
    return False


def detect_conditional_market(market_text: str) -> tuple[bool, list[str]]:
    """Detect markets whose resolution depends on a separate condition."""

    lowered = market_text.lower()
    reasons: list[str] = []
    conditional_phrases = (
        "if ",
        "conditional on",
        "conditioned on",
        "assuming",
        "unless",
        "while ",
        "given that",
        "in the event that",
        "provided that",
    )
    if any(phrase in lowered for phrase in conditional_phrases):
        reasons.append("Conditional phrasing detected.")

    for marker in ("before", "after"):
        for match in re.finditer(rf"\b{marker}\s+([^?.,;]+)", lowered):
            phrase = match.group(1).strip()
            if not _is_plain_time_expression(phrase):
                reasons.append(f"Conditional {marker}-event phrase detected.")

    return bool(reasons), reasons


def detect_compound_market(
    market_text: str,
    question: InterpretedQuestion,
) -> tuple[bool, list[str]]:
    """Detect markets that combine the target event with another event/entity."""

    lowered = market_text.lower()
    reasons: list[str] = []
    if re.search(r"\b(and|or)\b", lowered) and _contains_multiple_event_verbs(lowered):
        reasons.append("Compound event wording detected.")

    market_entities = _named_entities(market_text)
    question_entities = _normalized_entity_set(question.entities)
    extra_entities = [
        entity
        for entity in market_entities
        if _normalize_entity(entity) not in question_entities and not _is_ignorable_entity(entity)
    ]
    if len(extra_entities) >= 1 and detect_conditional_market(market_text)[0]:
        reasons.append("Separate named entity appears inside a conditional clause.")
    elif len(extra_entities) >= 2:
        reasons.append("Multiple unrelated named entities appear in one market.")

    return bool(reasons), reasons


def detect_scope_mismatch(
    question: InterpretedQuestion,
    market: NormalizedMarket,
) -> tuple[bool, list[str]]:
    """Detect markets that answer a narrower or different question."""

    market_text = _market_text(market).lower()
    question_text = f"{question.normalized_question} {question.target_event} {question.expected_outcome}".lower()
    reasons: list[str] = []

    if detect_conditional_market(market.title)[0] and not _condition_present_in_question(question, market_text):
        reasons.append("Market introduces a condition absent from the user question.")

    if "gas" in question_text and "oil" in market_text and "gas" not in market_text and "gasoline" not in market_text:
        reasons.append("Market tracks oil rather than gas prices.")

    if _has_numeric_threshold(market_text) and not _has_numeric_threshold(question_text):
        reasons.append("Market adds a numeric threshold absent from the user question.")

    if "market cap" in market_text and "market cap" not in question_text:
        reasons.append("Market asks about a post-event subcondition.")

    return bool(reasons), reasons


def detect_timeframe_conflict(
    question: InterpretedQuestion,
    market_text: str,
) -> tuple[bool, list[str]]:
    """Detect clear timeframe conflicts."""

    if not question.timeframe:
        return False, []

    lowered = market_text.lower()
    expected = question.timeframe.lower()
    reasons: list[str] = []

    expected_years = set(re.findall(r"20\d{2}", expected))
    market_years = set(re.findall(r"20\d{2}", lowered))
    if expected_years and market_years and expected_years.isdisjoint(market_years):
        reasons.append("Market year differs from question year.")

    expected_months = _months(expected)
    market_months = _months(lowered)
    if expected_months and market_months and expected_months.isdisjoint(market_months):
        if not _is_adjacent_or_related_month(expected_months, market_months):
            reasons.append("Market month/season differs from question timeframe.")

    if "summer" in expected_months and re.search(r"\bthis week\b|\btoday\b|\btomorrow\b", lowered):
        reasons.append("Market timeframe is much shorter than question timeframe.")

    return bool(reasons), reasons


def detect_entity_conflict(
    question: InterpretedQuestion,
    market_text: str,
) -> tuple[bool, list[str]]:
    """Detect markets centered on a conflicting named entity."""

    question_entities = _normalized_entity_set(question.entities)
    market_entities = _named_entities(market_text)
    reasons: list[str] = []

    if not question_entities:
        return False, []

    has_question_entity = _has_entity_phrase(question.entities, market_text)
    meaningful_market_entities = [
        entity for entity in market_entities if not _is_ignorable_entity(entity)
    ]
    extra_entities = [
        entity
        for entity in meaningful_market_entities
        if _normalize_entity(entity) not in question_entities
    ]

    if not has_question_entity and meaningful_market_entities:
        reasons.append("Market centers on a different named entity.")
    elif extra_entities and detect_conditional_market(market_text)[0]:
        reasons.append("Market adds a separate named entity sub-event.")

    return bool(reasons), reasons


def _timeframe_alignment(timeframe: str | None, market: NormalizedMarket) -> tuple[float, bool]:
    if not timeframe:
        return 0.7, False

    market_text = _market_text(market).lower()
    expected = timeframe.lower()
    if expected in market_text:
        return 1.0, False

    expected_years = set(re.findall(r"20\d{2}", expected))
    market_years = set(re.findall(r"20\d{2}", market_text))
    if market.close_time is not None:
        market_years.add(str(market.close_time.year))

    if expected_years and market_years and expected_years.isdisjoint(market_years):
        return 0.3, True

    expected_months = _months(expected)
    market_months = _months(market_text)
    if expected_months and market_months:
        if expected_months & market_months:
            return 1.0, False
        return 0.7, False

    if expected_years and not market_years:
        return 0.7, False
    return 0.7, False


def _outcome_alignment(expected_outcome: str, market_text: str) -> tuple[float, bool]:
    expected = expected_outcome.lower()
    market_lower = market_text.lower()

    opposite_pairs = [
        (("cut", "decrease", "lower"), ("hike", "increase", "raise")),
        (("rise", "increase"), ("fall", "decrease", "drop")),
        (("ipo", "public listing"), ("no ipo", "not ipo")),
        (("conflict", "war"), ("peace", "ceasefire")),
        (("impeach", "impeachment"), ("not impeached", "avoid impeachment")),
    ]
    for expected_terms, opposite_terms in opposite_pairs:
        if any(term in expected for term in expected_terms) and any(term in market_lower for term in opposite_terms):
            return 0.0, True

    if "cut" in expected and any(term in market_lower for term in ("rate cut", "rates cut", "cut rates", "decrease interest rates")):
        return 1.0, False
    if "gas prices" in expected and "gas" not in market_lower and "gasoline" not in market_lower:
        if "oil" in market_lower and any(term in market_lower for term in ("rise", "increase", "higher")):
            return 0.7, False
    if "rise" in expected and any(term in market_lower for term in ("rise", "increase", "higher")):
        return 1.0, False
    if "ipo" in expected and "ipo" in market_lower:
        return 1.0, False
    if "impeach" in expected and "impeach" in market_lower:
        return 1.0, False
    if "conflict" in expected and any(term in market_lower for term in ("conflict", "war", "military")):
        return 1.0, False

    expected_tokens = token_set(expected)
    market_tokens = token_set(market_text)
    if expected_tokens and expected_tokens <= market_tokens:
        return 1.0, False

    overlap = len(expected_tokens & market_tokens) / len(expected_tokens) if expected_tokens else 0.0
    if overlap >= 0.5:
        return 0.7, False
    if overlap > 0:
        return 0.3, False
    return 0.3, False


def _category_alignment(category: QuestionCategory, market_text: str) -> float:
    inferred = _infer_market_category(market_text.lower())
    if inferred == category:
        return 1.0
    if inferred == "other" or category == "other":
        return 0.5
    related = {
        ("macroeconomics", "finance"),
        ("finance", "macroeconomics"),
        ("politics", "geopolitical_risk"),
        ("geopolitical_risk", "politics"),
        ("technology", "finance"),
        ("finance", "technology"),
        ("consumer_products", "technology"),
    }
    return 0.7 if (category, inferred) in related else 0.2


def _resolution_clarity(market: NormalizedMarket) -> float:
    description = (market.description or "").strip()
    if len(description) >= 80 and any(term in description.lower() for term in ("resolve", "resolution", "source", "will")):
        return 1.0
    if market.title and len(market.title) >= 20:
        return 0.6
    return 0.3


def _market_quality(market: NormalizedMarket) -> float:
    score = 0.0
    if market.active is True and market.closed is not True:
        score += 0.35
    elif market.closed is True or market.active is False:
        score += 0.05
    else:
        score += 0.20

    volume = market.volume_usd or 0.0
    liquidity = market.liquidity_usd or 0.0
    score += 0.30 * min(volume / 1_000_000, 1.0)
    score += 0.25 * min(liquidity / 250_000, 1.0)
    if market.close_time is not None:
        score += 0.10
        if market.close_time.replace(tzinfo=market.close_time.tzinfo or UTC) < datetime.now(UTC):
            score -= 0.15
    return _clamp(score)


def _infer_market_category(lowered: str) -> QuestionCategory:
    if any(term in lowered for term in ("fed", "interest rate", "inflation", "gas price", "oil price", "opec")):
        return "macroeconomics"
    if any(term in lowered for term in ("ipo", "stock", "earnings", "bitcoin", "crypto")):
        return "finance"
    if any(term in lowered for term in ("openai", "ai ", "artificial intelligence")):
        return "technology"
    if any(term in lowered for term in ("air jordan", "nike", "sneaker", "iphone")):
        return "consumer_products"
    if any(term in lowered for term in ("iran", "war", "conflict", "military")):
        return "geopolitical_risk"
    if any(term in lowered for term in ("trump", "impeach", "election", "senate", "congress")):
        return "politics"
    if any(term in lowered for term in ("movie", "oscar", "grammy", "box office")):
        return "entertainment"
    if any(term in lowered for term in ("nba", "nfl", "mlb", "nhl", "world cup")):
        return "sports"
    return "other"


def _months(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december|summer|winter|spring|fall)\b",
            text,
        )
    )


def _classify_evidence(relevance: float, has_major_mismatch: bool = False) -> EvidenceType:
    if relevance >= 0.80 and not has_major_mismatch:
        return "Direct"
    if relevance >= 0.65 and not has_major_mismatch:
        return "Near-direct"
    if relevance >= 0.45:
        return "Related"
    if relevance >= 0.25:
        return "Weak"
    return "Irrelevant"


def _rationale(
    semantic: float,
    entity: float,
    timeframe: float,
    outcome: float,
    category: float,
    clarity: float,
    quality: float,
    caps: list[str],
    penalty_reasons: list[str],
) -> str:
    parts = [
        f"semantic={semantic:.2f}",
        f"entity={entity:.2f}",
        f"outcome={outcome:.2f}",
        f"timeframe={timeframe:.2f}",
        f"category={category:.2f}",
        f"clarity={clarity:.2f}",
        f"quality={quality:.2f}",
    ]
    if caps:
        parts.append("caps=" + ", ".join(caps))
    if penalty_reasons:
        parts.append("penalties=" + "; ".join(penalty_reasons))
    return "; ".join(parts)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _condition_present_in_question(question: InterpretedQuestion, market_text: str) -> bool:
    question_text = question.normalized_question.lower()
    market_lower = market_text.lower()
    if text_similarity(question_text, market_lower) >= 0.92:
        return True
    for marker in ("before", "after", "if", "unless", "while"):
        for phrase in _condition_phrases(marker, market_lower):
            phrase_tokens = token_set(phrase)
            if phrase_tokens and phrase_tokens <= token_set(question_text):
                return True
    return False


def _condition_phrases(marker: str, text: str) -> list[str]:
    if marker in {"before", "after"}:
        return [match.group(1).strip() for match in re.finditer(rf"\b{marker}\s+([^?.,;]+)", text)]
    if marker in text:
        return [text.split(marker, 1)[1]]
    return []


def _is_plain_time_expression(phrase: str) -> bool:
    clean = phrase.lower().strip()
    if re.fullmatch(r"(?:the\s+)?(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)(?:\s+meeting)?(?:\s+20\d{2})?", clean):
        return True
    if re.fullmatch(r"20\d{2}", clean):
        return True
    if re.fullmatch(r"(?:this|next)\s+(?:year|month|week|quarter|summer|winter|spring|fall)", clean):
        return True
    if re.fullmatch(r"(?:end|eoy|q[1-4])(?:\s+of)?(?:\s+20\d{2})?", clean):
        return True
    return False


def _contains_multiple_event_verbs(text: str) -> bool:
    verbs = ("cut", "confirm", "impeach", "acquire", "release", "rise", "fall", "win", "lose", "enter", "close")
    return sum(1 for verb in verbs if re.search(rf"\b{verb}", text)) >= 2


def _named_entities(text: str) -> list[str]:
    ignored_first_words = {"Will", "Does", "Do", "Can", "Could", "Would", "Is", "Are", "This", "The"}
    entities: list[str] = []
    for candidate in re.findall(r"\b(?:[A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+)*)", text):
        clean = candidate.strip().rstrip("?")
        words = clean.split()
        if not words or words[0] in ignored_first_words:
            if len(words) <= 1:
                continue
            clean = " ".join(words[1:])
        if clean:
            entities.append(clean)
    known = {
        "fed": "Fed",
        "federal reserve": "Federal Reserve",
        "fomc": "FOMC",
        "trump": "Trump",
        "biden": "Biden",
        "openai": "OpenAI",
        "kevin warsh": "Kevin Warsh",
        "powell": "Powell",
    }
    lowered = text.lower()
    for needle, entity in known.items():
        if needle in lowered:
            entities.append(entity)
    return _dedupe(entities)


def _normalized_entity_set(entities: list[str]) -> set[str]:
    normalized: set[str] = set()
    aliases = {
        "fed": {"fed", "federal reserve", "fomc"},
        "federal reserve": {"fed", "federal reserve", "fomc"},
        "fomc": {"fed", "federal reserve", "fomc"},
        "u.s.": {"u.s.", "us", "united states", "america"},
        "trump": {"trump", "donald trump"},
    }
    for entity in entities:
        clean = _normalize_entity(entity)
        normalized.add(clean)
        normalized.update(aliases.get(clean, set()))
    return normalized


def _normalize_entity(entity: str) -> str:
    return " ".join(re.findall(r"[a-z0-9.]+", entity.lower()))


def _is_ignorable_entity(entity: str) -> bool:
    clean = _normalize_entity(entity)
    return clean in {
        "",
        "yes",
        "no",
        "this",
        "market",
        "september",
        "october",
        "november",
        "december",
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
    }


def _has_numeric_threshold(text: str) -> bool:
    return bool(re.search(r"(?:above|below|over|under|less than|greater than|\$|%)\s*[\d,.]+|[\d,.]+\s*(?:bps|basis points|%)", text))


def _is_adjacent_or_related_month(expected_months: set[str], market_months: set[str]) -> bool:
    month_order = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    expected_numbers = {month_order[item] for item in expected_months if item in month_order}
    market_numbers = {month_order[item] for item in market_months if item in month_order}
    return bool(expected_numbers and market_numbers and min(abs(a - b) for a in expected_numbers for b in market_numbers) <= 2)


def _has_shared_core_event_terms(question: InterpretedQuestion, market_text: str) -> bool:
    core_text = f"{question.target_event} {question.expected_outcome}"
    core_tokens = token_set(core_text) - {"event", "occurs", "specified", "will", "market"}
    market_tokens = token_set(market_text)
    if not core_tokens:
        return True
    return bool(core_tokens & market_tokens)


def _question_is_historical(question: InterpretedQuestion) -> bool:
    text = question.normalized_question.lower()
    return any(term in text for term in ("did ", "was ", "were ", "happened", "resolved", "in 2024", "in 2025"))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
