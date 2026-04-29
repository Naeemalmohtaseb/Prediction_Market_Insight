"""Forecast confidence scoring."""

from pmi_agent.schemas import RankedMarket


class ConfidenceEngine:
    """Compute deterministic confidence scores from ranked market evidence."""

    def score(self, ranked_markets: list[RankedMarket]) -> float:
        """Return an overall confidence score in the 0-1 range."""

        if not ranked_markets:
            return 0.0

        top_markets = ranked_markets[:5]
        average_market_confidence = sum(market.confidence_score for market in top_markets) / len(
            top_markets
        )
        evidence_depth = min(len(ranked_markets) / 5.0, 1.0)
        return _clamp((0.8 * average_market_confidence) + (0.2 * evidence_depth))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
