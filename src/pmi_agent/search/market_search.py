"""Market search orchestration."""

from pmi_agent.clients.base import PredictionMarketClient
from pmi_agent.config import get_config
from pmi_agent.interpretation.search_term_expander import SearchTermExpander
from pmi_agent.schemas import InterpretedQuestion, NormalizedMarket


class MarketSearchService:
    """Search provider clients and normalize market records."""

    def __init__(
        self,
        clients: list[PredictionMarketClient],
        expander: SearchTermExpander | None = None,
    ) -> None:
        self.clients = clients
        self.expander = expander or SearchTermExpander()

    def search(self, question: InterpretedQuestion) -> list[NormalizedMarket]:
        """Search all configured clients for markets related to a question."""

        config = get_config()
        markets: list[NormalizedMarket] = []
        seen_ids: set[tuple[str, str]] = set()

        for term in self.expander.expand(question):
            for client in self.clients:
                for market in client.search_markets(term, limit=config.max_search_results):
                    identity = (market.source, market.market_id)
                    if identity in seen_ids:
                        continue
                    seen_ids.add(identity)
                    markets.append(market)

        return markets
