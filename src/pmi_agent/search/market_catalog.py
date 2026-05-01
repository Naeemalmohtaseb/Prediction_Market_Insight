"""Local market catalog refresh and search service."""

from __future__ import annotations

import logging
from typing import Any

from pmi_agent.clients.base import PredictionMarketClient
from pmi_agent.schemas import NormalizedMarket
from pmi_agent.storage.db import StorageManager

logger = logging.getLogger(__name__)

CATALOG_SEED_TERMS = (
    "Fed rate cut",
    "Trump impeachment",
    "inflation CPI",
    "OpenAI IPO",
)


class MarketCatalogService:
    """Refresh and query the local market catalog."""

    def __init__(self, storage: StorageManager | None = None) -> None:
        self.storage = storage or StorageManager()

    def refresh_provider(self, provider_client: PredictionMarketClient, limit: int = 500) -> int:
        """Fetch active markets from one provider and upsert them into the catalog."""

        provider_name = _provider_name(provider_client)
        list_active = getattr(provider_client, "list_active_markets", None)
        if list_active is None:
            logger.warning("Provider does not support list_active_markets provider=%s", provider_name)
            return 0

        markets = list_active(limit=limit)
        if _should_seed(provider_client):
            markets.extend(_seed_search_markets(provider_client, limit=max(10, min(50, limit // 10))))
        markets = _dedupe_markets(markets)
        count = self.storage.upsert_catalog_markets(markets)
        logger.info("Refreshed market catalog provider=%s count=%d", provider_name, count)
        return count

    def refresh_all(
        self,
        provider_clients: list[PredictionMarketClient],
        limit_per_provider: int = 500,
    ) -> dict[str, int]:
        """Refresh the catalog for multiple providers."""

        counts: dict[str, int] = {}
        for provider_client in provider_clients:
            provider_name = _provider_name(provider_client)
            try:
                counts[provider_name] = self.refresh_provider(provider_client, limit=limit_per_provider)
            except Exception as exc:
                logger.warning("Catalog refresh failed provider=%s error=%s", provider_name, exc)
                counts[provider_name] = 0
        return counts

    def search(
        self,
        query: str,
        providers: list[str] | None = None,
        limit: int = 50,
    ) -> list[NormalizedMarket]:
        """Search the local catalog."""

        return self.storage.search_catalog(query, providers=providers, limit=limit)

    def stats(self) -> dict[str, Any]:
        """Return catalog statistics."""

        return self.storage.get_catalog_stats()


def _provider_name(provider_client: PredictionMarketClient) -> str:
    explicit = getattr(provider_client, "provider", None) or getattr(provider_client, "source", None)
    if explicit:
        return str(explicit)
    name = provider_client.__class__.__name__.lower()
    if "polymarket" in name:
        return "polymarket"
    if "kalshi" in name:
        return "kalshi"
    return name.removesuffix("client")


def _should_seed(provider_client: PredictionMarketClient) -> bool:
    name = provider_client.__class__.__name__.lower()
    return "polymarket" in name or "kalshi" in name


def _seed_search_markets(provider_client: PredictionMarketClient, limit: int) -> list[NormalizedMarket]:
    markets: list[NormalizedMarket] = []
    for term in CATALOG_SEED_TERMS:
        try:
            markets.extend(provider_client.search_markets(term, limit=limit))
        except Exception as exc:
            logger.warning("Catalog seed search failed provider=%s term=%r error=%s", _provider_name(provider_client), term, exc)
    return markets


def _dedupe_markets(markets: list[NormalizedMarket]) -> list[NormalizedMarket]:
    deduped: list[NormalizedMarket] = []
    seen: set[tuple[str, str]] = set()
    for market in markets:
        identity = (market.source, market.market_id)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(market)
    return deduped
