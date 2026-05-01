"""Forecast probability aggregation."""

from __future__ import annotations

import math

from pmi_agent.engine.confidence_engine import ConfidenceEngine
from pmi_agent.engine.probability_engine import ProbabilityEngine
from pmi_agent.engine.provider_comparison import compare_providers
from pmi_agent.schemas import ForecastResult, InterpretedQuestion, MarketProbability, RankedMarket

DECENT_DIRECT_WEIGHT = 0.15


class AggregationEngine:
    """Aggregate market-implied probabilities deterministically."""

    def __init__(
        self,
        probability_engine: ProbabilityEngine | None = None,
        confidence_engine: ConfidenceEngine | None = None,
    ) -> None:
        self.probability_engine = probability_engine or ProbabilityEngine()
        self.confidence_engine = confidence_engine or ConfidenceEngine()

    def aggregate(
        self,
        question: InterpretedQuestion,
        ranked_markets: list[RankedMarket],
    ) -> ForecastResult:
        """Extract probabilities, aggregate them, and compute confidence."""

        key_warnings: list[str] = []
        market_probabilities: list[MarketProbability] = []
        for ranked_market in ranked_markets:
            if ranked_market.evidence_type == "Irrelevant":
                continue
            market_probabilities.append(self.probability_engine.extract_probability(question, ranked_market))

        downweighted_top_count = sum(
            1
            for item in ranked_markets[:10]
            if item.relevance_score >= 0.45 and (item.is_conditional or item.is_compound)
        )
        if downweighted_top_count >= 2:
            key_warnings.append("Several high-similarity markets were conditional or compound and were downweighted.")

        usable = [item for item in market_probabilities if item.implied_probability is not None and item.market_weight > 0]
        direct = [item for item in usable if item.evidence_type in {"Direct", "Near-direct"} and _is_clean_direct(item)]
        related = [item for item in usable if item.evidence_type in {"Related", "Weak"} or not _is_clean_direct(item)]

        direct_probability = _weighted_probability(direct)
        related_probability = _weighted_probability(related)
        direct_weight = sum(item.market_weight for item in direct)

        if direct_probability is not None and direct_weight >= DECENT_DIRECT_WEIGHT:
            if related_probability is None:
                estimated_probability = direct_probability
            else:
                estimated_probability = (0.85 * direct_probability) + (0.15 * related_probability)
        elif direct_probability is not None:
            if related_probability is None:
                estimated_probability = direct_probability
                key_warnings.append("Direct market weight is weak; confidence is limited.")
            else:
                estimated_probability = (0.65 * direct_probability) + (0.35 * related_probability)
        elif related_probability is not None:
            estimated_probability = related_probability
            key_warnings.append("No direct market found; estimate is based on related or weak market signals.")
        else:
            estimated_probability = None
            key_warnings.append("No usable market-implied probability found.")

        disagreement_score = _weighted_standard_deviation(usable)
        provider_summary, provider_disagreement_score, provider_notes = compare_providers(market_probabilities)
        confidence_score, confidence_label, confidence_warnings = self.confidence_engine.score(
            question,
            market_probabilities,
            direct_probability,
            related_probability,
            disagreement_score,
            provider_disagreement_score,
        )
        key_warnings.extend(confidence_warnings)
        key_warnings.extend(provider_notes)
        key_warnings.extend(_collect_market_warnings(market_probabilities))

        return ForecastResult(
            estimated_probability=estimated_probability,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            direct_market_probability=direct_probability,
            related_signal_probability=related_probability,
            disagreement_score=disagreement_score,
            direct_markets_count=len(direct),
            related_markets_count=len(related),
            markets_used_count=len(usable),
            key_warnings=_dedupe(key_warnings),
            market_probabilities=market_probabilities,
            provider_summary=provider_summary,
            provider_disagreement_score=provider_disagreement_score,
            provider_notes=provider_notes,
        )

    def aggregate_probability(self, ranked_markets: list[RankedMarket]) -> float | None:
        """Backward-compatible weighted mean over ranked market probabilities."""

        usable = [
            (item.market.implied_probability, item.confidence_score)
            for item in ranked_markets
            if item.market.implied_probability is not None and item.confidence_score > 0
        ]
        total_weight = sum(weight for _, weight in usable)
        if total_weight <= 0:
            return None
        return sum(probability * weight for probability, weight in usable if probability is not None) / total_weight


def _weighted_probability(items: list[MarketProbability]) -> float | None:
    total_weight = sum(item.market_weight for item in items)
    if total_weight <= 0:
        return None
    return sum((item.implied_probability or 0.0) * item.market_weight for item in items) / total_weight


def _weighted_standard_deviation(items: list[MarketProbability]) -> float | None:
    usable = [item for item in items if item.implied_probability is not None and item.market_weight > 0]
    if len(usable) < 2:
        return None
    mean = _weighted_probability(usable)
    if mean is None:
        return None
    total_weight = sum(item.market_weight for item in usable)
    variance = sum(item.market_weight * ((item.implied_probability or 0.0) - mean) ** 2 for item in usable) / total_weight
    return math.sqrt(max(0.0, variance))


def _collect_market_warnings(items: list[MarketProbability]) -> list[str]:
    warnings: list[str] = []
    for item in items:
        warnings.extend(item.warnings)
    return warnings


def _is_clean_direct(item: MarketProbability) -> bool:
    mismatch_markers = {
        "conditional market",
        "compound market",
        "scope mismatch",
        "timeframe conflict",
        "entity conflict",
        "Conditional or compound market does not directly resolve the user question.",
        "Market scope is narrower or different than the interpreted event.",
        "Market centers on a different entity or sub-event.",
        "Market timeframe differs from the user's timeframe.",
    }
    return not any(marker in item.warnings for marker in mismatch_markers)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
