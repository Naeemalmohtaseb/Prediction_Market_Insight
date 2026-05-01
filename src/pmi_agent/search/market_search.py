"""Market search orchestration."""

import logging
from typing import Literal

from pmi_agent.clients.base import PredictionMarketClient
from pmi_agent.config import get_config
from pmi_agent.interpretation.search_term_expander import SearchTermExpander
from pmi_agent.search.market_catalog import MarketCatalogService
from pmi_agent.schemas import InterpretedQuestion, NormalizedMarket

logger = logging.getLogger(__name__)


class MarketSearchService:
    """Search provider clients and preserve all deduplicated candidates."""

    def __init__(
        self,
        clients: list[PredictionMarketClient] | None = None,
        expander: SearchTermExpander | None = None,
        catalog_service: MarketCatalogService | None = None,
    ) -> None:
        self.clients = clients or []
        self.expander = expander or SearchTermExpander()
        self.catalog_service = catalog_service or MarketCatalogService()
        self.last_search_sources: dict[str, int] = {"catalog": 0, "live": 0}

    def search(
        self,
        question: InterpretedQuestion,
        providers: list[PredictionMarketClient] | None = None,
        limit_per_term: int | None = None,
        search_mode: Literal["live", "catalog", "hybrid"] = "hybrid",
    ) -> list[NormalizedMarket]:
        """Search all providers for markets related to an interpreted question."""

        active_providers = providers or self.clients
        if not active_providers and search_mode == "live":
            return []

        config = get_config()
        term_limit = limit_per_term or config.max_search_results
        markets: list[NormalizedMarket] = []
        seen_ids: set[tuple[str, str]] = set()
        self.last_search_sources = {"catalog": 0, "live": 0}
        provider_names = [_provider_source(provider) for provider in active_providers]

        for term in self.expander.expand(question):
            if search_mode in {"catalog", "hybrid"}:
                try:
                    catalog_markets = self.catalog_service.search(
                        term,
                        providers=provider_names or None,
                        limit=term_limit,
                    )
                except Exception as exc:
                    logger.warning("Catalog search failed term=%r error=%s", term, exc)
                    catalog_markets = []
                self._append_deduped(catalog_markets, markets, seen_ids, source="catalog")

            if search_mode == "catalog":
                continue

            live_limit = term_limit if search_mode == "live" else max(3, term_limit // 3)
            for provider in active_providers:
                provider_name = provider.__class__.__name__
                try:
                    term_markets = provider.search_markets(term, limit=live_limit)
                except Exception as exc:
                    logger.warning(
                        "Provider search failed provider=%s term=%r error=%s",
                        provider_name,
                        term,
                        exc,
                    )
                    continue

                logger.info(
                    "Provider search provider=%s term=%r count=%d",
                    provider_name,
                    term,
                    len(term_markets),
                )
                self._append_deduped(term_markets, markets, seen_ids, source="live")

        return markets

    def _append_deduped(
        self,
        term_markets: list[NormalizedMarket],
        markets: list[NormalizedMarket],
        seen_ids: set[tuple[str, str]],
        source: str,
    ) -> None:
        for market in term_markets:
            identity = (market.source, market.market_id)
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            markets.append(market)
            self.last_search_sources[source] = self.last_search_sources.get(source, 0) + 1


def _provider_source(provider: PredictionMarketClient) -> str:
    explicit = getattr(provider, "provider", None) or getattr(provider, "source", None)
    if explicit:
        return str(explicit)
    name = provider.__class__.__name__.lower()
    if "polymarket" in name:
        return "polymarket"
    if "kalshi" in name:
        return "kalshi"
    return name.removesuffix("client")
