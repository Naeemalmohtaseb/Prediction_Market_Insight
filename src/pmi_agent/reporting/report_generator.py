"""Forecast report generation."""

from pmi_agent.schemas import ForecastReport, ForecastResult, RankedMarket


class ReportGenerator:
    """Generate natural-language reports from deterministic forecast results."""

    def generate(self, result: ForecastResult) -> ForecastReport:
        """Create a report without inventing probabilities or scores."""

        executive_summary = _summary(result)
        market_evidence = [_market_line(market) for market in result.ranked_markets[:5]]
        uncertainty_notes = _uncertainty_notes(result)

        return ForecastReport(
            result=result,
            executive_summary=executive_summary,
            market_evidence=market_evidence,
            uncertainty_notes=uncertainty_notes,
        )


def _summary(result: ForecastResult) -> str:
    if result.aggregate_probability is None:
        return "No market-implied forecast is available because no relevant markets were found."

    percent = result.aggregate_probability * 100
    confidence = result.confidence_score * 100
    return (
        f"Market-implied aggregate probability: {percent:.1f}%. "
        f"Deterministic confidence score: {confidence:.1f}%."
    )


def _market_line(ranked_market: RankedMarket) -> str:
    probability = ranked_market.market.implied_probability
    relevance = ranked_market.relevance_score * 100
    confidence = ranked_market.confidence_score * 100
    probability_text = "unknown" if probability is None else f"{probability * 100:.1f}%"
    return (
        f"{ranked_market.market.title} | probability={probability_text} | "
        f"relevance={relevance:.1f}% | confidence={confidence:.1f}%"
    )


def _uncertainty_notes(result: ForecastResult) -> list[str]:
    if not result.ranked_markets:
        return ["No market evidence was available from configured clients."]

    notes = ["Scores are deterministic functions of market data and text similarity."]
    if result.confidence_score < 0.4:
        notes.append("Overall confidence is low; treat the aggregate as weak market evidence.")
    return notes
