"""Manual smoke test for read-only Kalshi market ingestion."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.clients.kalshi_client import KalshiClient


QUERIES = ["Fed", "Trump", "inflation"]


def main() -> None:
    client = KalshiClient()

    for query in QUERIES:
        print(f"\nKalshi search: {query}")
        markets = client.search_markets(query, limit=8)
        if client.last_error:
            print(f"  warning: {client.last_error}")
        _print_markets(markets)

    print("\nKalshi active markets:")
    markets = client.list_active_markets(limit=10)
    if client.last_error:
        print(f"  warning: {client.last_error}")
    _print_markets(markets)


def _print_markets(markets) -> None:
    if not markets:
        print("  no markets returned")
        return

    print(f"  {'ticker':<36} {'yes':>7} {'no':>7} {'volume':>10} {'liq':>10} {'active':>7} title")
    for market in markets:
        print(
            f"  {market.market_id[:36]:<36} "
            f"{_fmt_prob(market.yes_price):>7} "
            f"{_fmt_prob(market.no_price):>7} "
            f"{_fmt_num(market.volume_usd):>10} "
            f"{_fmt_num(market.liquidity_usd):>10} "
            f"{str(market.active):>7} "
            f"{market.title[:80]}"
        )


def _fmt_prob(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}"


if __name__ == "__main__":
    main()
