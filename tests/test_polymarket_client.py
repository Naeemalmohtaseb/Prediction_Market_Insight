"""Tests for Polymarket normalization and graceful HTTP behavior."""

import requests

from pmi_agent.clients.polymarket_client import PolymarketClient


def test_parses_outcomes_and_prices_from_json_strings() -> None:
    market = PolymarketClient().normalize_market(
        {
            "id": "123",
            "question": "Will rates change?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.37", "0.63"]',
            "volume": "1000.5",
            "liquidity": "250.25",
        }
    )

    assert market.yes_price == 0.37
    assert market.no_price == 0.63
    assert market.implied_probability == 0.37
    assert market.outcome_prices == {"Yes": 0.37, "No": 0.63}
    assert market.volume_usd == 1000.5
    assert market.liquidity_usd == 250.25


def test_parses_outcomes_and_prices_from_lists() -> None:
    market = PolymarketClient().normalize_market(
        {
            "id": "456",
            "question": "Will a bill pass?",
            "outcomes": ["Yes", "No"],
            "outcomePrices": [0.12, 0.88],
        }
    )

    assert market.outcome_prices["Yes"] == 0.12
    assert market.outcome_prices["No"] == 0.88


def test_converts_string_prices_to_floats() -> None:
    market = PolymarketClient().normalize_market(
        {
            "id": "789",
            "question": "Will it happen?",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["42", "58"],
        }
    )

    assert market.yes_price == 0.42
    assert market.no_price == 0.58


def test_handles_missing_volume_and_liquidity() -> None:
    market = PolymarketClient().normalize_market(
        {
            "id": "abc",
            "question": "Will data be sparse?",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.5", "0.5"],
        }
    )

    assert market.volume_usd is None
    assert market.liquidity_usd is None


def test_binary_yes_no_normalization() -> None:
    market = PolymarketClient().normalize_market(
        {
            "conditionId": "condition-1",
            "question": "Will a binary market normalize?",
            "slug": "binary-market",
            "outcomes": '["No", "Yes"]',
            "outcomePrices": '["0.24", "0.76"]',
            "active": "true",
            "closed": False,
        }
    )

    assert market.market_id == "condition-1"
    assert market.yes_price == 0.76
    assert market.no_price == 0.24
    assert market.implied_probability == 0.76
    assert market.active is True
    assert market.closed is False
    assert market.slug == "binary-market"
    assert market.raw["conditionId"] == "condition-1"


def test_search_returns_empty_list_on_http_failure() -> None:
    class FailingSession:
        def get(self, *args, **kwargs):
            raise requests.RequestException("network unavailable")

    client = PolymarketClient(session=FailingSession())

    assert client.search_markets("Fed") == []
    assert client.last_error is not None
    assert "failed" in client.last_error
