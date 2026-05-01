"""Tests for deterministic report generation."""

from pmi_agent.engine.aggregation_engine import AggregationEngine
from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.reporting.report_generator import ReportGenerator
from pmi_agent.schemas import ContextItem, NormalizedMarket, RankedMarket


def test_report_includes_probability_confidence_table_warnings_and_disclaimer() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    result = AggregationEngine().aggregate(
        question,
        [
            RankedMarket(
                market=NormalizedMarket(
                    market_id="m1",
                    source="polymarket",
                    title="Will the Fed cut interest rates by September?",
                    description="This market resolves Yes if the Federal Reserve cuts interest rates.",
                    implied_probability=0.7,
                    yes_price=0.7,
                    no_price=0.3,
                    outcome_prices={"Yes": 0.7, "No": 0.3},
                    volume_usd=1_000_000,
                    liquidity_usd=100_000,
                    active=True,
                    closed=False,
                ),
                evidence_type="Direct",
                relevance_score=0.9,
                semantic_similarity=0.9,
                entity_overlap=0.9,
                timeframe_alignment=1.0,
                outcome_alignment=1.0,
                category_alignment=1.0,
                resolution_clarity=1.0,
                market_quality=1.0,
                rationale="test",
            )
        ],
    )

    report = ReportGenerator().generate_markdown(question, result)

    assert "70.0%" in report
    assert "Confidence" in report
    assert "| Evidence type | Market title | Outcome used |" in report
    assert "# Provider Comparison" in report
    assert "# Context Layer" in report
    assert "Context items provide background only and are not used as direct probability inputs." in report
    assert "Market-implied probabilities are not guarantees." in report
    assert "This is not financial, investment, trading, or betting advice." in report
    assert "you should bet" not in report.lower()
    assert "you should trade" not in report.lower()
    assert "profit" not in report.lower()


def test_report_includes_context_items_when_present() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    result = AggregationEngine().aggregate(question, [])
    result.context_summary = "Recent context surfaced Fed and FOMC items."
    result.context_items = [
        ContextItem(
            title="Fed rate cut expectations shift",
            source="Example News",
            url="https://example.com/fed",
            published_at="Wed, 29 Apr 2026 12:00:00 GMT",
            summary="Background only.",
            relevance_score=0.82,
        )
    ]

    report = ReportGenerator().generate_markdown(question, result)

    assert "Fed rate cut expectations shift" in report
    assert "Example News" in report
    assert "0.82" in report
