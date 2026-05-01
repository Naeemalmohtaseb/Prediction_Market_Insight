"""Tests for lightweight RSS context retrieval."""

import requests

from pmi_agent.context.news_context import NewsContextService
from pmi_agent.engine.aggregation_engine import AggregationEngine
from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.schemas import ContextItem, NormalizedMarket, RankedMarket


RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>Test News</title>
  <item>
    <title>Federal Reserve officials discuss rate cuts before September meeting</title>
    <link>https://example.com/fed-rate-cuts</link>
    <source>Example Macro News</source>
    <pubDate>Wed, 29 Apr 2026 12:00:00 GMT</pubDate>
    <description>FOMC officials discussed inflation and interest rates.</description>
  </item>
  <item>
    <title>Federal Reserve officials discuss rate cuts before September meeting</title>
    <link>https://example.com/fed-rate-cuts</link>
    <source>Example Macro News</source>
    <pubDate>Wed, 29 Apr 2026 12:00:00 GMT</pubDate>
    <description>Duplicate item.</description>
  </item>
  <item>
    <title>Local sports team wins playoff game</title>
    <link>https://example.com/sports</link>
    <source>Example Sports</source>
    <pubDate>Wed, 29 Apr 2026 12:00:00 GMT</pubDate>
    <description>Unrelated sports item.</description>
  </item>
</channel>
</rss>
"""


def test_context_service_parses_and_deduplicates_rss_items() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    service = NewsContextService(session=_FakeSession(RSS_FIXTURE))

    items = service.fetch_context(question, max_items=5)

    urls = [item.url for item in items]
    assert urls.count("https://example.com/fed-rate-cuts") == 1
    assert any("Federal Reserve" in item.title for item in items)
    assert all(0 <= item.relevance_score <= 1 for item in items)


def test_relevance_ranking_puts_relevant_title_above_irrelevant() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    service = NewsContextService()
    ranked = service.rank_context_items(
        question,
        [
            ContextItem(title="Local sports team wins playoff game", source="Sports", url="https://x/s", published_at=None, summary=None, relevance_score=0),
            ContextItem(title="Federal Reserve rate cut expectations rise", source="Macro", url="https://x/f", published_at=None, summary="FOMC rates", relevance_score=0),
        ],
    )

    assert ranked[0].title.startswith("Federal Reserve")


def test_fetch_failure_returns_empty_list_and_warning() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    service = NewsContextService(session=_FailingSession())

    items = service.fetch_context(question, max_items=5)

    assert items == []
    assert service.last_warnings
    assert any("failed" in warning.lower() for warning in service.last_warnings)


def test_deterministic_summary_with_and_without_items() -> None:
    question = QueryInterpreter().interpret("Will OpenAI IPO before 2027?")
    service = NewsContextService()
    item = ContextItem(
        title="OpenAI IPO speculation follows valuation discussions",
        source="Example",
        url="https://example.com/openai",
        published_at=None,
        summary="OpenAI valuation and public offering context.",
        relevance_score=0.8,
    )

    assert "do not modify" in service.summarize_context(question, [item])
    assert "relies only on prediction-market evidence" in service.summarize_context(question, [])


def test_estimated_probability_is_unchanged_when_context_is_added() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    result = AggregationEngine().aggregate(question, [_ranked_market()])
    before = result.estimated_probability

    result.context_items = [
        ContextItem(
            title="Federal Reserve context item",
            source="Example",
            url="https://example.com/fed",
            published_at=None,
            summary=None,
            relevance_score=0.8,
        )
    ]
    result.context_summary = "Context attached for background only."

    assert result.estimated_probability == before


class _FakeSession:
    def __init__(self, text: str) -> None:
        self.text = text

    def get(self, *args, **kwargs):
        return _FakeResponse(self.text)


class _FailingSession:
    def get(self, *args, **kwargs):
        raise requests.RequestException("network unavailable")


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def _ranked_market() -> RankedMarket:
    return RankedMarket(
        market=NormalizedMarket(
            market_id="m1",
            source="polymarket",
            title="Will the Fed cut interest rates by September?",
            description="This market resolves Yes if the Federal Reserve cuts rates.",
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
