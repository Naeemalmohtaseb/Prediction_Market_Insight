"""Forecast probability aggregation."""

from pmi_agent.schemas import RankedMarket


class AggregationEngine:
    """Aggregate market-implied probabilities deterministically."""

    def aggregate_probability(self, ranked_markets: list[RankedMarket]) -> float | None:
        """Return a confidence-weighted mean probability.

        Returns None when there are no markets with positive confidence.
        """

        weighted_values: list[tuple[float, float]] = []
        for ranked_market in ranked_markets:
            weight = ranked_market.confidence_score
            probability = ranked_market.market.implied_probability
            if weight <= 0 or probability is None:
                continue
            weighted_values.append((probability, weight))

        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight <= 0:
            return None

        return sum(probability * weight for probability, weight in weighted_values) / total_weight
