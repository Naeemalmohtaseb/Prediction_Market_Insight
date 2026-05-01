"""Tests for Kalshi normalization and graceful HTTP behavior."""

import requests

from pmi_agent.clients.kalshi_client import KalshiClient, _normalize_price


def test_price_normalization_from_cents_to_probability() -> None:
    assert _normalize_price(63) == 0.63
    assert _normalize_price("42") == 0.42


def test_price_normalization_when_already_probability() -> None:
    assert _normalize_price(0.63) == 0.63
    assert _normalize_price("0.2500") == 0.25


def test_yes_bid_ask_midpoint_and_spread() -> None:
    market = KalshiClient().normalize_market(
        {
            "ticker": "KXFEDCUT-26SEP",
            "event_ticker": "KXFEDCUT",
            "title": "Will the Fed cut rates by September?",
            "status": "active",
            "yes_bid": 61,
            "yes_ask": 65,
            "no_bid": 35,
            "no_ask": 39,
            "volume": 1200,
            "liquidity": 5000,
        }
    )

    assert market.source == "kalshi"
    assert market.market_id == "KXFEDCUT-26SEP"
    assert market.yes_price == 0.63
    assert market.no_price == 0.37
    assert market.outcome_prices == {"Yes": 0.63, "No": 0.37}
    assert market.raw["spread"] == 0.04
    assert market.active is True
    assert market.closed is False


def test_no_price_inferred_from_yes_price() -> None:
    market = KalshiClient().normalize_market(
        {
            "ticker": "KXOPENAIIPO-27",
            "title": "Will OpenAI IPO before 2027?",
            "status": "open",
            "last_price": 32,
        }
    )

    assert market.yes_price == 0.32
    assert market.no_price == 0.68
    assert market.outcome_prices == {"Yes": 0.32, "No": 0.68}


def test_field_normalization_combines_description_and_dates() -> None:
    market = KalshiClient().normalize_market(
        {
            "ticker": "KXINFLATION-26",
            "event_ticker": "KXINFLATION",
            "title": "Will inflation be above 3%?",
            "subtitle": "CPI annual rate",
            "rules_primary": "Resolves according to official CPI data.",
            "rules_secondary": "Secondary rule text.",
            "close_time": "2026-12-31T23:59:00Z",
            "yes_bid_dollars": "0.4000",
            "yes_ask_dollars": "0.4400",
            "liquidity_dollars": "1234.50",
            "volume_fp": "555.00",
            "open_interest_fp": "10.00",
            "category": "Economics",
        }
    )

    assert market.description is not None
    assert "CPI annual rate" in market.description
    assert "official CPI" in market.description
    assert market.close_time is not None
    assert market.liquidity_usd == 1234.50
    assert market.volume_usd == 555.0
    assert market.raw["event_id"] == "KXINFLATION"
    assert market.raw["open_interest"] == 10.0


def test_missing_fields_do_not_crash() -> None:
    market = KalshiClient().normalize_market({"ticker": "KXSPARSE"})

    assert market.market_id == "KXSPARSE"
    assert market.title == "Kalshi market KXSPARSE"
    assert market.yes_price is None
    assert market.no_price is None
    assert market.outcome_prices == {}


def test_search_returns_empty_list_on_http_failure() -> None:
    class FailingSession:
        def get(self, *args, **kwargs):
            raise requests.RequestException("network unavailable")

    client = KalshiClient(session=FailingSession())

    assert client.search_markets("Fed") == []
    assert client.last_error is not None
    assert "failed" in client.last_error


def test_client_side_search_filtering_returns_relevant_markets() -> None:
    client = KalshiClient(session=_FakeSession(
        [
            {
                "markets": [
                    {
                        "ticker": "KXFEDCUT-26SEP",
                        "title": "Will the Fed cut rates by September?",
                        "yes_bid": 60,
                        "yes_ask": 64,
                    },
                    {
                        "ticker": "KXNBA-PLAYOFFS",
                        "title": "Will the Lakers win tonight?",
                        "yes_bid": 50,
                        "yes_ask": 54,
                    },
                ]
            }
        ]
    ))

    markets = client.search_markets("Fed cut rates", limit=5)

    assert [market.market_id for market in markets] == ["KXFEDCUT-26SEP"]


def test_pagination_cursor_handling() -> None:
    client = KalshiClient(
        session=_FakeSession(
            [
                {
                    "cursor": "next",
                    "markets": [{"ticker": "KXONE", "title": "First market"}],
                },
                {
                    "markets": [{"ticker": "KXTWO", "title": "Second market"}],
                },
            ]
        ),
        max_pages=3,
    )

    markets = client.list_active_markets(limit=10)

    assert [market.market_id for market in markets] == ["KXONE", "KXTWO"]


class _FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self.payloads:
            return _FakeResponse({"markets": []})
        return _FakeResponse(self.payloads.pop(0))


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload
