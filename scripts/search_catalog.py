"""Search the local market catalog."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.search.market_catalog import MarketCatalogService


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or "Fed rate cut"
    markets = MarketCatalogService().search(query, limit=20)

    print(f"Catalog search: {query}")
    if not markets:
        print("  no cached markets matched")
        return

    print(f"  {'provider':<11} {'id':<34} {'yes':>7} {'no':>7} {'volume':>10} {'liq':>10} {'close':<19} title")
    for market in markets:
        print(
            f"  {market.source:<11} "
            f"{market.market_id[:34]:<34} "
            f"{_fmt_prob(market.yes_price):>7} "
            f"{_fmt_prob(market.no_price):>7} "
            f"{_fmt_num(market.volume_usd):>10} "
            f"{_fmt_num(market.liquidity_usd):>10} "
            f"{(market.close_time.isoformat()[:19] if market.close_time else '-'):<19} "
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
