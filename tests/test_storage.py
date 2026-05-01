"""Tests for SQLite persistence."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from pmi_agent.schemas import ForecastResult, InterpretedQuestion, MarketProbability, NormalizedMarket
from pmi_agent.storage.db import StorageManager


def test_init_db_creates_database_and_tables(tmp_path) -> None:
    db_path = tmp_path / "test.sqlite"
    manager = StorageManager(db_path)

    manager.init_db()

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        table_names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "analyses" in table_names
    assert "market_snapshots" in table_names


def test_save_list_get_delete_analysis(tmp_path) -> None:
    manager = StorageManager(tmp_path / "test.sqlite")
    question = _question()
    result = _result()
    markdown = "# Forecast Summary\nSaved report"

    analysis_id = manager.save_analysis(question, result, markdown)

    assert isinstance(analysis_id, int)
    recent = manager.list_recent_analyses()
    assert recent[0]["id"] == analysis_id
    assert recent[0]["user_question"] == question.original_question

    saved = manager.get_analysis(analysis_id)
    assert saved is not None
    assert saved["markdown_report"] == markdown
    assert saved["interpreted_question"]["target_event"] == question.target_event
    assert saved["forecast_result"]["confidence_label"] == result.confidence_label
    assert len(saved["market_snapshots"]) == 1
    assert saved["market_snapshots"][0]["title"] == "Will the Fed cut rates by September?"

    assert manager.delete_analysis(analysis_id) is True
    assert manager.get_analysis(analysis_id) is None
    with manager._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
    assert count == 0


def test_json_fields_are_valid_json(tmp_path) -> None:
    manager = StorageManager(tmp_path / "test.sqlite")
    analysis_id = manager.save_analysis(_question(), _result(), "# Report")

    with manager._connect() as conn:
        analysis = conn.execute(
            "SELECT key_warnings_json, interpreted_question_json, forecast_result_json FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
        snapshot = conn.execute(
            "SELECT warnings_json, penalty_reasons_json, raw_market_json FROM market_snapshots WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()

    for value in analysis:
        assert json.loads(value) is not None
    for value in snapshot:
        assert json.loads(value) is not None


def test_catalog_upsert_search_filter_clear_and_conversion(tmp_path) -> None:
    manager = StorageManager(tmp_path / "test.sqlite")
    market = _catalog_market("m1", "polymarket", "Will the Fed cut rates by September?", yes=0.4)

    assert manager.upsert_catalog_markets([market]) == 1
    assert manager.upsert_catalog_markets([market.model_copy(update={"yes_price": 0.45})]) == 1

    stats = manager.get_catalog_stats()
    assert stats["total_markets"] == 1
    assert stats["by_provider"] == {"polymarket": 1}

    results = manager.search_catalog("Fed rate cut", limit=5)
    assert len(results) == 1
    assert results[0].market_id == "m1"
    assert results[0].source == "polymarket"
    assert results[0].raw["event_id"] == "event-m1"

    assert manager.search_catalog("Fed", providers=["kalshi"]) == []
    assert manager.clear_market_catalog(provider="polymarket") == 1
    assert manager.get_catalog_stats()["total_markets"] == 0


def test_prune_old_catalog(tmp_path) -> None:
    manager = StorageManager(tmp_path / "test.sqlite")
    manager.upsert_catalog_markets([_catalog_market("old", "kalshi", "Old inflation market")])
    old_timestamp = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    with manager._connect() as conn:
        conn.execute("UPDATE market_catalog SET last_seen_at = ?", (old_timestamp,))

    assert manager.prune_old_catalog(days=14) == 1
    assert manager.get_catalog_stats()["total_markets"] == 0


def _question() -> InterpretedQuestion:
    return InterpretedQuestion(
        original_question="Will the Fed cut rates by September?",
        normalized_question="Will the Fed cut rates by September?",
        category="macroeconomics",
        target_event="Federal Reserve cuts interest rates",
        expected_outcome="interest rates are cut",
        geography="United States",
        timeframe="by September",
        search_terms=["Fed cut rates"],
        related_concepts=["interest rates"],
    )


def _result() -> ForecastResult:
    market = NormalizedMarket(
        market_id="123",
        source="polymarket",
        title="Will the Fed cut rates by September?",
        description="This market resolves Yes if the Fed cuts rates by September.",
        implied_probability=0.4,
        yes_price=0.4,
        no_price=0.6,
        outcome_prices={"Yes": 0.4, "No": 0.6},
        volume_usd=100_000,
        liquidity_usd=25_000,
        active=True,
        closed=False,
        raw={"event_id": "event-1", "conditionId": "abc"},
    )
    probability = MarketProbability(
        market=market,
        evidence_type="Direct",
        relevance_score=0.9,
        target_outcome="Yes",
        implied_probability=0.4,
        probability_source="yes_price",
        market_weight=0.5,
        warnings=["low liquidity", "scope mismatch"],
    )
    return ForecastResult(
        estimated_probability=0.4,
        confidence_score=72.0,
        confidence_label="High",
        direct_market_probability=0.4,
        related_signal_probability=None,
        disagreement_score=None,
        direct_markets_count=1,
        related_markets_count=0,
        markets_used_count=1,
        key_warnings=["low liquidity"],
        market_probabilities=[probability],
    )


def _catalog_market(market_id: str, source: str, title: str, yes: float = 0.5) -> NormalizedMarket:
    return NormalizedMarket(
        market_id=market_id,
        source=source,
        title=title,
        description="This cached market resolves from public market rules.",
        implied_probability=yes,
        yes_price=yes,
        no_price=1 - yes,
        outcome_prices={"Yes": yes, "No": 1 - yes},
        volume_usd=10_000,
        liquidity_usd=1_000,
        active=True,
        closed=False,
        raw={"event_id": f"event-{market_id}", "spread": 0.02, "open_interest": 100},
    )
