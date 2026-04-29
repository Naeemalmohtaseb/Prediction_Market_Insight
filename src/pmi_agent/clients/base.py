"""Prediction market client interfaces."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pmi_agent.schemas import NormalizedMarket


class PredictionMarketClient(ABC):
    """Abstract interface for provider-specific market clients."""

    @abstractmethod
    def search_markets(self, query: str, limit: int = 25) -> list[NormalizedMarket]:
        """Return normalized market records for a query."""

    @abstractmethod
    def normalize_market(self, raw_market: Mapping[str, Any]) -> NormalizedMarket:
        """Convert a provider-specific market record into a normalized schema."""
