"""Provider-level comparison helpers for normalized market evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pmi_agent.schemas import MarketProbability


def compare_providers(
    market_probabilities: list[MarketProbability],
) -> tuple[dict[str, dict[str, Any]], float | None, list[str]]:
    """Summarize usable probability evidence by provider."""

    grouped: dict[str, list[MarketProbability]] = defaultdict(list)
    for item in market_probabilities:
        grouped[item.market.source].append(item)

    summary: dict[str, dict[str, Any]] = {}
    for provider, items in sorted(grouped.items()):
        usable = [item for item in items if item.implied_probability is not None and item.market_weight > 0]
        total_weight = sum(item.market_weight for item in usable)
        summary[provider] = {
            "markets_count": len(items),
            "usable_markets_count": len(usable),
            "direct_or_near_direct_count": sum(
                1 for item in usable if item.evidence_type in {"Direct", "Near-direct"} and _is_clean_direct(item)
            ),
            "weighted_probability": _weighted_probability(usable),
            "average_relevance": _average([item.relevance_score for item in usable]),
            "total_weight": total_weight,
            "average_liquidity_score": _average([item.liquidity_score for item in usable]),
            "average_volume_score": _average([item.volume_score for item in usable]),
            "average_spread_score": _average([item.spread_score for item in usable]),
            "warnings_count": sum(len(item.warnings) for item in items),
        }

    provider_probabilities = [
        data["weighted_probability"]
        for data in summary.values()
        if data["weighted_probability"] is not None
    ]
    provider_disagreement_score = None
    if len(provider_probabilities) >= 2:
        provider_disagreement_score = max(provider_probabilities) - min(provider_probabilities)

    return summary, provider_disagreement_score, _provider_notes(summary, provider_disagreement_score)


def _weighted_probability(items: list[MarketProbability]) -> float | None:
    total_weight = sum(item.market_weight for item in items)
    if total_weight <= 0:
        return None
    return sum((item.implied_probability or 0.0) * item.market_weight for item in items) / total_weight


def _average(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _provider_notes(
    summary: dict[str, dict[str, Any]],
    provider_disagreement_score: float | None,
) -> list[str]:
    notes: list[str] = []
    usable_providers = [
        provider
        for provider, data in summary.items()
        if data["weighted_probability"] is not None and data["total_weight"] > 0
    ]

    if len(usable_providers) < 2:
        notes.append("Only one provider produced usable probability evidence.")
    elif provider_disagreement_score is not None:
        if provider_disagreement_score <= 0.05:
            notes.append("Providers broadly agree.")
        elif provider_disagreement_score <= 0.15:
            notes.append("Providers show moderate disagreement.")
        else:
            notes.append("Providers show substantial disagreement.")

    weighted_providers = [
        (provider, data["total_weight"])
        for provider, data in summary.items()
        if data["total_weight"] > 0
    ]
    if len(weighted_providers) >= 2:
        strongest_weight = max(weight for _, weight in weighted_providers)
        weak = [
            provider
            for provider, weight in weighted_providers
            if strongest_weight > 0 and weight < strongest_weight * 0.25
        ]
        if weak:
            notes.append(
                "Provider comparison is limited by weaker market quality from "
                + ", ".join(sorted(weak))
                + "."
            )

    return _dedupe(notes)


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
