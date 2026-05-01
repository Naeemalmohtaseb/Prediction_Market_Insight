"""Refresh the local active market catalog for read-only providers."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.clients.kalshi_client import KalshiClient
from pmi_agent.clients.polymarket_client import PolymarketClient
from pmi_agent.search.market_catalog import MarketCatalogService


def main() -> None:
    service = MarketCatalogService()
    counts = service.refresh_all([PolymarketClient(), KalshiClient()], limit_per_provider=500)
    stats = service.stats()

    print("Catalog refresh counts:")
    for provider, count in counts.items():
        print(f"  {provider}: {count}")

    print("\nCatalog stats:")
    print(f"  total_markets: {stats.get('total_markets', 0)}")
    print(f"  by_provider: {stats.get('by_provider', {})}")
    print(f"  active_by_provider: {stats.get('active_by_provider', {})}")
    print(f"  latest_fetched_at: {stats.get('latest_fetched_at')}")
    print(f"  latest_last_seen_at: {stats.get('latest_last_seen_at')}")


if __name__ == "__main__":
    main()
