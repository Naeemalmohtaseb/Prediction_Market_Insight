"""Streamlit entrypoint for the Prediction Market Intelligence Agent."""

from pathlib import Path
import sys
from typing import Any

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.clients.polymarket_client import PolymarketClient
from pmi_agent.schemas import NormalizedMarket


def main() -> None:
    """Render the Streamlit dashboard."""

    st.set_page_config(page_title="Prediction Market Intelligence Agent", layout="wide")
    st.title("Prediction Market Intelligence Agent")

    st.caption(
        "Market-implied forecasting dashboard. No betting, trading, financial, "
        "legal, or investment advice."
    )

    query = st.text_input(
        "Search query or forecast question",
        placeholder="Fed decision, Trump, AI regulation, election",
    )

    limit = st.slider("Result limit", min_value=5, max_value=100, value=25, step=5)

    if st.button("Search Polymarket", type="primary", disabled=not query.strip()):
        client = PolymarketClient()
        markets = client.search_markets(query, limit=limit)

        if client.last_error:
            st.warning(client.last_error)

        if not markets:
            st.info("No normalized markets were returned for this query.")
            return

        st.subheader("Normalized Markets")
        st.dataframe(
            [_market_row(market) for market in markets],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Raw normalized JSON"):
            st.json([market.model_dump(mode="json") for market in markets])


def _market_row(market: NormalizedMarket) -> dict[str, Any]:
    return {
        "market_title": market.title,
        "yes_price": market.yes_price,
        "no_price": market.no_price,
        "volume": market.volume_usd,
        "liquidity": market.liquidity_usd,
        "close_date": market.close_time.isoformat() if market.close_time else None,
        "active": market.active,
        "closed": market.closed,
        "provider_market_id": market.market_id,
    }


if __name__ == "__main__":
    main()
