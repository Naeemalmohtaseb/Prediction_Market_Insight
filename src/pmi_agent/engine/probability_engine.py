"""Deterministic market probability extraction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pmi_agent.schemas import InterpretedQuestion, MarketProbability, RankedMarket


class ProbabilityEngine:
    """Extract target-outcome probabilities without model judgment."""

    def normalize_probability(self, value: float) -> float:
        """Normalize a probability expressed as 0-1 or 0-100 into 0-1."""

        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 100.0:
            return value / 100.0
        raise ValueError("Probability must be between 0 and 1, or between 1 and 100 percent.")

    def infer_target_outcome(
        self,
        question: InterpretedQuestion,
        ranked_market: RankedMarket,
    ) -> str | None:
        """Infer which market outcome corresponds to the user's question."""

        market = ranked_market.market
        outcome_names = {name.casefold(): name for name in market.outcome_prices}
        has_binary = "yes" in outcome_names and "no" in outcome_names
        if not has_binary:
            return _best_outcome_match(
                f"{question.expected_outcome} {question.target_event}",
                market.outcome_prices.keys(),
            )

        if _question_asks_negative(question):
            return outcome_names["no"]
        return outcome_names["yes"]

    def extract_probability(
        self,
        question: InterpretedQuestion,
        ranked_market: RankedMarket,
    ) -> MarketProbability:
        """Extract the implied probability for the target outcome."""

        market = ranked_market.market
        warnings: list[str] = []
        if ranked_market.evidence_type == "Irrelevant":
            warnings.append("irrelevant market skipped")
        if ranked_market.evidence_type in {"Related", "Weak"}:
            warnings.append("weak/indirect evidence")
        warnings.extend(ranked_market.penalty_reasons)
        if ranked_market.is_conditional:
            warnings.append("conditional market")
        if ranked_market.is_compound:
            warnings.append("compound market")
        if ranked_market.has_scope_mismatch:
            warnings.append("scope mismatch")
        if ranked_market.has_timeframe_conflict:
            warnings.append("timeframe conflict")
        if ranked_market.has_entity_conflict:
            warnings.append("entity conflict")
        if market.active is False or market.closed is True:
            warnings.append("inactive or closed market")
        if market.liquidity_usd is not None and market.liquidity_usd < 10_000:
            warnings.append("low liquidity")

        spread = _extract_spread(market.raw)
        quality = _quality_components(market)
        if market.liquidity_usd is not None and market.liquidity_usd <= 0:
            warnings.append("Provider reported zero liquidity; weight uses fallback market-quality logic.")
        if spread is None and _has_price(market):
            warnings.append("No bid-ask spread available; confidence may be overstated.")
        if market.volume_usd is None or market.volume_usd <= 0:
            warnings.append("Low volume or missing volume.")
        if spread is not None and spread >= 0.10:
            warnings.append("wide spread")

        target_outcome = self.infer_target_outcome(question, ranked_market)
        if _market_title_is_inverse(question, ranked_market):
            warnings.append("market title may be inverse")

        probability: float | None = None
        probability_source: str | None = None
        if target_outcome is None:
            warnings.append("unclear outcome mapping")
        elif target_outcome.casefold() == "yes" and market.yes_price is not None:
            probability = market.yes_price
            probability_source = "yes_price"
        elif target_outcome.casefold() == "no" and market.no_price is not None:
            probability = market.no_price
            probability_source = "no_price"
        elif target_outcome in market.outcome_prices:
            probability = market.outcome_prices[target_outcome]
            probability_source = f"outcome_prices[{target_outcome}]"
        elif market.implied_probability is not None and target_outcome.casefold() == "yes":
            probability = market.implied_probability
            probability_source = "implied_probability"

        if probability is None:
            warnings.append("missing price")
            if target_outcome is not None:
                warnings.append("unclear outcome mapping")

        market_weight = self.compute_market_weight(ranked_market, probability)
        return MarketProbability(
            market=market,
            evidence_type=ranked_market.evidence_type,
            relevance_score=ranked_market.relevance_score,
            target_outcome=target_outcome,
            implied_probability=probability,
            probability_source=probability_source,
            market_weight=market_weight,
            warnings=_dedupe(warnings),
            liquidity_score=quality.liquidity_score,
            volume_score=quality.volume_score,
            spread_score=quality.spread_score,
            recency_score=quality.recency_score,
            resolution_score=quality.resolution_score,
            provider_quality_note=quality.provider_quality_note,
        )

    def compute_market_weight(
        self,
        ranked_market: RankedMarket,
        implied_probability: float | None,
    ) -> float:
        """Compute deterministic market aggregation weight."""

        if implied_probability is None:
            return 0.0

        market = ranked_market.market
        quality = _quality_components(market)
        weight = (
            ranked_market.relevance_score**2
            * quality.liquidity_score
            * quality.volume_score
            * quality.spread_score
            * quality.recency_score
            * quality.resolution_score
        )
        if ranked_market.is_conditional:
            weight *= 0.55
        if ranked_market.is_compound:
            weight *= 0.60
        if ranked_market.has_scope_mismatch:
            weight *= 0.50
        if ranked_market.has_timeframe_conflict:
            weight *= 0.60
        if ranked_market.has_entity_conflict:
            weight *= 0.50
        return max(0.0, weight)


@dataclass(frozen=True)
class QualityComponents:
    liquidity_score: float
    volume_score: float
    spread_score: float
    recency_score: float
    resolution_score: float
    provider_quality_note: str | None = None


def liquidity_score(
    liquidity: float | None,
    volume: float | None = None,
    spread: float | None = None,
) -> float:
    if liquidity is not None and liquidity > 0:
        return min(1.0, math.log1p(liquidity) / math.log1p(100_000))
    if _positive(volume) and spread is not None:
        return 0.55
    if _positive(volume):
        return 0.45
    return 0.25


def volume_score(volume: float | None, price_or_spread_exists: bool = False) -> float:
    if volume is not None and volume > 0:
        return min(1.0, math.log1p(volume) / math.log1p(1_000_000))
    if price_or_spread_exists:
        return 0.35
    return 0.2


def spread_score(
    spread: float | None,
    provider: str | None = None,
    has_yes_no_prices: bool = False,
    has_any_price: bool = False,
) -> float:
    if spread is None:
        if provider == "kalshi" and has_yes_no_prices:
            return 0.75
        if has_any_price:
            return 0.65
        return 0.4
    return max(0.1, 1.0 - spread / 0.20)


def recency_score(active: bool | None, closed: bool | None, close_time: datetime | None) -> float:
    if closed is True or active is False:
        score = 0.3
    else:
        score = 1.0

    if close_time is None:
        return score

    close_time = close_time if close_time.tzinfo is not None else close_time.replace(tzinfo=UTC)
    age_days = (datetime.now(UTC) - close_time).days
    if age_days > 365:
        return min(score, 0.15)
    if age_days > 90:
        return min(score, 0.25)
    return score


def resolution_score(title: str, description: str | None) -> float:
    description = (description or "").strip()
    if description:
        return 1.0
    if len(title.strip()) >= 20:
        return 0.6
    return 0.3


def _quality_components(market: Any) -> QualityComponents:
    spread = _extract_spread(market.raw)
    has_any_price = _has_price(market)
    has_yes_no_prices = market.yes_price is not None and market.no_price is not None
    provider_quality_note = None
    if (market.liquidity_usd is None or market.liquidity_usd <= 0) and _positive(market.volume_usd) and spread is not None:
        provider_quality_note = "Liquidity unavailable/zero; using bid-ask/volume fallback."

    return QualityComponents(
        liquidity_score=liquidity_score(market.liquidity_usd, market.volume_usd, spread),
        volume_score=volume_score(market.volume_usd, has_any_price or spread is not None),
        spread_score=spread_score(
            spread,
            provider=market.source,
            has_yes_no_prices=has_yes_no_prices,
            has_any_price=has_any_price,
        ),
        recency_score=recency_score(market.active, market.closed, market.close_time),
        resolution_score=resolution_score(market.title, market.description),
        provider_quality_note=provider_quality_note,
    )


def _question_asks_negative(question: InterpretedQuestion) -> bool:
    text = f"{question.normalized_question} {question.expected_outcome}".lower()
    return any(
        re.search(pattern, text)
        for pattern in (
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bfail",
            r"\breject",
            r"\bdenied\b",
            r"\bdoesn't\b",
            r"\bwon't\b",
        )
    )


def _best_outcome_match(expected_outcome: str, outcomes: Any) -> str | None:
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected_outcome.lower()))
    best_outcome: str | None = None
    best_overlap = 0
    for outcome in outcomes:
        outcome_text = str(outcome)
        outcome_tokens = set(re.findall(r"[a-z0-9]+", outcome_text.lower()))
        overlap = len(expected_tokens & outcome_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_outcome = outcome_text
    return best_outcome if best_overlap > 0 else None


def _market_title_is_inverse(question: InterpretedQuestion, ranked_market: RankedMarket) -> bool:
    q_negative = _question_asks_negative(question)
    title = ranked_market.market.title.lower()
    title_negative = any(term in title for term in (" no ", " not ", " fail", " rejected", "without"))
    return q_negative != title_negative and ("not" in question.normalized_question.lower() or title_negative)


def _extract_spread(raw: dict[str, Any]) -> float | None:
    for key in ("spread", "bidAskSpread", "bid_ask_spread"):
        value = raw.get(key)
        if value is None or value == "":
            continue
        try:
            spread = float(value)
        except (TypeError, ValueError):
            continue
        if spread > 1.0 and spread <= 100.0:
            return spread / 100.0
        if 0.0 <= spread <= 1.0:
            return spread
    return None


def _has_price(market: Any) -> bool:
    return (
        market.yes_price is not None
        or market.no_price is not None
        or market.implied_probability is not None
        or bool(market.outcome_prices)
    )


def _positive(value: float | None) -> bool:
    return value is not None and value > 0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
