"""Tests for market catalog refresh and hybrid search."""

from collections.abc import Mapping
from typing import Any

from pmi_agent.clients.base import PredictionMarketClient
from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.schemas import NormalizedMarket
from pmi_agent.search.market_catalog import MarketCatalogService
from pmi_agent.search.market_search import MarketSearchService
from pmi_agent.storage.db import StorageManager


def test_market_catalog_service_refresh_all_stores_provider_markets(tmp_path) -> None:
    storage = StorageManager(tmp_path / "catalog.sqlite")
    service = MarketCatalogService(storage)

    counts = service.refresh_all(
        [
            _FakeProvider("polymarket", [_market("shared", "polymarket", "Will the Fed cut rates?")]),
            _FakeProvider("kalshi", [_market("kalshi-1", "kalshi", "Will CPI inflation be above 3%?")]),
        ],
        limit_per_provider=10,
    )

    assert counts == {"polymarket": 1, "kalshi": 1}
    stats = storage.get_catalog_stats()
    assert stats["total_markets"] == 2
    assert stats["by_provider"] == {"kalshi": 1, "polymarket": 1}


def test_hybrid_search_deduplicates_catalog_and_live_results(tmp_path) -> None:
    storage = StorageManager(tmp_path / "catalog.sqlite")
    catalog_market = _market("fed-1", "polymarket", "Will the Fed cut rates by September?")
    storage.upsert_catalog_markets([catalog_market])
    provider = _FakeProvider(
        "polymarket",
        active_markets=[],
        search_markets=[catalog_market, _market("fed-2", "polymarket", "Fed rate cut by June?")],
    )
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    search = MarketSearchService(catalog_service=MarketCatalogService(storage))

    markets = search.search(question, providers=[provider], limit_per_term=10, search_mode="hybrid")

    ids = [(market.source, market.market_id) for market in markets]
    assert ids.count(("polymarket", "fed-1")) == 1
    assert ("polymarket", "fed-2") in ids
    assert search.last_search_sources["catalog"] >= 1
    assert search.last_search_sources["live"] >= 1


def test_catalog_only_search_uses_no_live_provider_calls(tmp_path) -> None:
    storage = StorageManager(tmp_path / "catalog.sqlite")
    storage.upsert_catalog_markets([_market("fed-1", "polymarket", "Will the Fed cut rates by September?")])
    provider = _FakeProvider("polymarket", active_markets=[], search_markets=[])
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")

    markets = MarketSearchService(catalog_service=MarketCatalogService(storage)).search(
        question,
        providers=[provider],
        limit_per_term=10,
        search_mode="catalog",
    )

    assert markets
    assert provider.search_calls == 0


class _FakeProvider(PredictionMarketClient):
    def __init__(
        self,
        provider: str,
        active_markets: list[NormalizedMarket],
        search_markets: list[NormalizedMarket] | None = None,
    ) -> None:
        self.provider = provider
        self.active_markets = active_markets
        self._search_markets = search_markets or active_markets
        self.search_calls = 0

    def list_active_markets(self, limit: int = 100) -> list[NormalizedMarket]:
        return self.active_markets[:limit]

    def search_markets(self, query: str, limit: int = 25) -> list[NormalizedMarket]:
        self.search_calls += 1
        return self._search_markets[:limit]

    def normalize_market(self, raw_market: Mapping[str, Any]) -> NormalizedMarket:
        raise NotImplementedError


def _market(market_id: str, source: str, title: str) -> NormalizedMarket:
    return NormalizedMarket(
        market_id=market_id,
        source=source,
        title=title,
        description="Cached public market",
        implied_probability=0.4,
        yes_price=0.4,
        no_price=0.6,
        outcome_prices={"Yes": 0.4, "No": 0.6},
        volume_usd=10_000,
        liquidity_usd=1_000,
        active=True,
        closed=False,
        raw={"event_id": f"event-{market_id}", "spread": 0.03},
    )
