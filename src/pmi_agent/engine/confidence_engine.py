"""Forecast confidence scoring."""

from __future__ import annotations

import math

from pmi_agent.schemas import InterpretedQuestion, MarketProbability


class ConfidenceEngine:
    """Compute deterministic 0-100 confidence scores from market evidence."""

    def score(
        self,
        question: InterpretedQuestion,
        market_probabilities: list[MarketProbability],
        direct_probability: float | None,
        related_probability: float | None,
        disagreement_score: float | None,
        provider_disagreement_score: float | None = None,
    ) -> tuple[float, str, list[str]]:
        """Return confidence score, label, and confidence warnings."""

        _ = question
        usable = [item for item in market_probabilities if item.implied_probability is not None and item.market_weight > 0]
        direct = [item for item in usable if item.evidence_type in {"Direct", "Near-direct"} and _is_clean_direct(item)]
        warnings: list[str] = []

        direct_market_coverage = min(sum(item.market_weight for item in direct) / 0.50, 1.0)
        liquidity_quality = _weighted_average(usable, [_liquidity_quality(item) for item in usable])
        spread_quality = _weighted_average(usable, [_spread_quality(item) for item in usable])
        agreement_across_markets = 1.0 - min((disagreement_score or 0.0) / 0.50, 1.0)
        resolution_clarity = _weighted_average(usable, [_resolution_clarity(item) for item in usable])
        recency = _weighted_average(usable, [_recency(item) for item in usable])
        external_context_support = 0.0

        confidence = (
            25 * direct_market_coverage
            + 20 * liquidity_quality
            + 15 * spread_quality
            + 15 * agreement_across_markets
            + 10 * resolution_clarity
            + 10 * recency
            + 5 * external_context_support
        )

        if provider_disagreement_score is not None:
            if provider_disagreement_score > 0.15:
                confidence -= 8
                warnings.append("Prediction market providers disagree materially.")
            elif provider_disagreement_score > 0.05:
                confidence -= 4

        if not usable:
            warnings.append("No usable probabilities found")
        if not direct or direct_probability is None:
            warnings.append("No direct market found")
        elif direct_market_coverage < 0.35:
            warnings.append("Low direct market coverage")
        if liquidity_quality < 0.45:
            warnings.append("Thin liquidity")
        if spread_quality < 0.55:
            warnings.append("Wide spreads")
        if disagreement_score is not None and disagreement_score >= 0.20:
            warnings.append("High disagreement across markets")
        if related_probability is not None and direct_probability is None:
            warnings.append("Estimate relies on related or weak market signals")

        confidence = round(max(0.0, min(100.0, confidence)), 1)
        return confidence, _label(confidence), _dedupe(warnings)


def _weighted_average(items: list[MarketProbability], values: list[float]) -> float:
    if not items:
        return 0.0
    total_weight = sum(item.market_weight for item in items)
    if total_weight <= 0:
        return sum(values) / len(values)
    return sum(value * item.market_weight for item, value in zip(items, values)) / total_weight


def _liquidity_quality(item: MarketProbability) -> float:
    if item.liquidity_score is not None and item.volume_score is not None:
        return (item.liquidity_score + item.volume_score) / 2
    liquidity = item.market.liquidity_usd
    volume = item.market.volume_usd
    liquidity_part = 0.5 if liquidity is None else min(1.0, math.log1p(liquidity) / math.log1p(100_000))
    volume_part = 0.5 if volume is None else min(1.0, math.log1p(volume) / math.log1p(1_000_000))
    return (liquidity_part + volume_part) / 2


def _spread_quality(item: MarketProbability) -> float:
    if item.spread_score is not None:
        return item.spread_score
    return 0.45 if "wide spread" in item.warnings else 0.8


def _resolution_clarity(item: MarketProbability) -> float:
    if item.resolution_score is not None:
        return item.resolution_score
    if item.market.description:
        return 1.0
    if len(item.market.title) >= 20:
        return 0.6
    return 0.3


def _recency(item: MarketProbability) -> float:
    if item.recency_score is not None:
        return item.recency_score
    if item.market.closed is True or item.market.active is False:
        return 0.3
    return 1.0


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


def _label(confidence: float) -> str:
    if confidence >= 70:
        return "High"
    if confidence >= 40:
        return "Medium"
    return "Low"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
