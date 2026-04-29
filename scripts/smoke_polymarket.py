"""Manual smoke test for read-only Polymarket market discovery."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.clients.polymarket_client import PolymarketClient
from pmi_agent.schemas import NormalizedMarket


def main() -> None:
    client = PolymarketClient()

    for query in ("Fed", "Trump"):
        print(f"\nsearch_markets({query!r})")
        _print_markets(client.search_markets(query, limit=5))
        if client.last_error:
            print(f"warning: {client.last_error}")

    print("\nlist_active_markets(limit=10)")
    _print_markets(client.list_active_markets(limit=10))
    if client.last_error:
        print(f"warning: {client.last_error}")


def _print_markets(markets: list[NormalizedMarket]) -> None:
    if not markets:
        print("No markets returned.")
        return

    header = f"{'ID':<10} {'YES':>6} {'NO':>6} {'VOL':>12} {'LIQ':>12} TITLE"
    print(header)
    print("-" * len(header))
    for market in markets:
        print(
            f"{market.market_id[:10]:<10} "
            f"{_fmt_prob(market.yes_price):>6} "
            f"{_fmt_prob(market.no_price):>6} "
            f"{_fmt_money(market.volume_usd):>12} "
            f"{_fmt_money(market.liquidity_usd):>12} "
            f"{market.title[:80]}"
        )


def _fmt_prob(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


if __name__ == "__main__":
    main()
