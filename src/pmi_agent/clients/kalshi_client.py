"""Read-only Kalshi public market data client."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pmi_agent.clients.base import PredictionMarketClient
from pmi_agent.config import AppConfig, get_config
from pmi_agent.schemas import NormalizedMarket
from pmi_agent.search.semantic_similarity import text_similarity, token_set

logger = logging.getLogger(__name__)


class KalshiClient(PredictionMarketClient):
    """Read-only adapter for Kalshi public market discovery endpoints."""

    def __init__(
        self,
        config: AppConfig | None = None,
        session: Session | None = None,
        max_pages: int = 4,
    ) -> None:
        self.config = config or get_config()
        self.session = session or _build_session()
        self.max_pages = max_pages
        self.last_error: str | None = None
        self._active_markets_cache: list[Mapping[str, Any]] | None = None
        self._series_markets_cache: dict[str, list[Mapping[str, Any]]] = {}

    def search_markets(self, query: str, limit: int = 25) -> list[NormalizedMarket]:
        """Search active Kalshi markets with bounded pagination and local filtering."""

        query = query.strip()
        if not query:
            return []

        raw_markets = list(self._get_active_market_candidates(limit=max(100, min(500, limit * 20))))
        raw_markets.extend(self._fetch_hint_markets(query, limit=max(50, limit * 4)))
        if not raw_markets:
            return []

        scored: list[tuple[float, Mapping[str, Any]]] = []
        seen_tickers: set[str] = set()
        for raw_market in raw_markets:
            ticker = _text(raw_market.get("ticker"))
            if ticker and ticker in seen_tickers:
                continue
            if ticker:
                seen_tickers.add(ticker)
            score = _market_similarity(query, raw_market)
            if score >= 0.20 and _has_shared_search_tokens(query, raw_market, score):
                scored.append((score, raw_market))

        scored.sort(key=lambda item: item[0], reverse=True)
        return self._normalize_many([raw for _, raw in scored], limit=limit)

    def list_active_markets(self, limit: int = 100) -> list[NormalizedMarket]:
        """List active/open Kalshi markets from the public API."""

        raw_markets = self._fetch_market_pages(
            limit=max(1, min(500, limit)),
            max_pages=self.max_pages,
            status="open",
        )
        return self._normalize_many(raw_markets, limit=limit)

    def get_market_by_id(self, market_id: str) -> NormalizedMarket | None:
        """Fetch one Kalshi market by ticker."""

        return self.get_market_by_ticker(market_id)

    def get_market_by_ticker(self, ticker: str) -> NormalizedMarket | None:
        """Fetch one Kalshi market by ticker."""

        clean_ticker = ticker.strip()
        if not clean_ticker:
            return None

        payload = self._get_json(f"/markets/{clean_ticker}", fallback_404=True)
        if payload is None:
            return None

        raw_market: Mapping[str, Any] | None = None
        if isinstance(payload, Mapping):
            nested = payload.get("market")
            if isinstance(nested, Mapping):
                raw_market = nested
            else:
                raw_market = payload

        if raw_market is None:
            return None

        try:
            return self.normalize_market(raw_market)
        except ValueError as exc:
            logger.warning("Unable to normalize Kalshi market %s: %s", clean_ticker, exc)
            self.last_error = str(exc)
            return None

    def get_orderbook(self, ticker: str) -> dict[str, Any] | None:
        """Fetch a public Kalshi orderbook for one market ticker."""

        clean_ticker = ticker.strip()
        if not clean_ticker:
            return None
        payload = self._get_json(f"/markets/{clean_ticker}/orderbook", fallback_404=True)
        return dict(payload) if isinstance(payload, Mapping) else None

    def normalize_market(self, raw_market: Mapping[str, Any]) -> NormalizedMarket:
        """Normalize a raw Kalshi market record."""

        ticker = _text(raw_market.get("ticker"))
        if not ticker:
            raise ValueError("Kalshi record is missing ticker.")

        title = _text(raw_market.get("title")) or _text(raw_market.get("subtitle"))
        if not title:
            title = f"Kalshi market {ticker}"

        yes_bid = _first_price(raw_market.get("yes_bid_dollars"), raw_market.get("yes_bid"))
        yes_ask = _first_price(raw_market.get("yes_ask_dollars"), raw_market.get("yes_ask"))
        no_bid = _first_price(raw_market.get("no_bid_dollars"), raw_market.get("no_bid"))
        no_ask = _first_price(raw_market.get("no_ask_dollars"), raw_market.get("no_ask"))
        last_price = _first_price(raw_market.get("last_price_dollars"), raw_market.get("last_price"))

        yes_price = _midpoint(yes_bid, yes_ask)
        if yes_price is None:
            yes_price = _first_not_none(last_price, yes_bid, yes_ask)

        no_price = _midpoint(no_bid, no_ask)
        if no_price is None and yes_price is not None:
            no_price = _clamp_probability(1.0 - yes_price)

        outcome_prices = {
            key: value
            for key, value in {"Yes": yes_price, "No": no_price}.items()
            if value is not None
        }
        spread = None
        if yes_bid is not None and yes_ask is not None:
            spread = round(max(0.0, yes_ask - yes_bid), 6)

        raw = dict(raw_market)
        raw.update(
            {
                "event_id": raw_market.get("event_ticker"),
                "midpoint": yes_price,
                "spread": spread,
                "open_interest": _first_float(raw_market.get("open_interest"), raw_market.get("open_interest_fp")),
                "category": raw_market.get("category"),
            }
        )

        status = (_text(raw_market.get("status")) or "").lower()
        active = status in {"open", "active", "initialized"}
        closed = status in {"closed", "settled", "resolved", "finalized", "expired"}

        return NormalizedMarket(
            market_id=ticker,
            source="kalshi",
            title=title,
            description=_description(raw_market),
            url=_market_url(ticker),
            outcome_name="Yes",
            implied_probability=yes_price,
            yes_price=yes_price,
            no_price=no_price,
            outcome_prices=outcome_prices,
            volume_usd=_first_float(
                raw_market.get("volume_dollars"),
                raw_market.get("volume"),
                raw_market.get("volume_fp"),
                raw_market.get("volume_24h"),
                raw_market.get("volume_24h_fp"),
            ),
            liquidity_usd=_first_float(raw_market.get("liquidity_dollars"), raw_market.get("liquidity")),
            close_time=_parse_datetime(
                raw_market.get("close_time")
                or raw_market.get("expiration_time")
                or raw_market.get("latest_expiration_time")
            ),
            active=active,
            closed=closed,
            slug=ticker,
            raw=raw,
        )

    def _fetch_market_pages(
        self,
        limit: int,
        max_pages: int,
        status: str | None = None,
    ) -> list[Mapping[str, Any]]:
        raw_markets: list[Mapping[str, Any]] = []
        cursor: str | None = None
        page_limit = max(1, min(500, limit))

        for _ in range(max_pages):
            params: dict[str, Any] = {"limit": page_limit}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor

            payload = self._get_json("/markets", params=params)
            if payload is None:
                return raw_markets
            if not isinstance(payload, Mapping):
                return raw_markets

            page_markets = payload.get("markets")
            if not isinstance(page_markets, Sequence) or isinstance(page_markets, (str, bytes)):
                return raw_markets

            for raw_market in page_markets:
                if isinstance(raw_market, Mapping):
                    raw_markets.append(raw_market)

            cursor_value = payload.get("cursor")
            cursor = cursor_value if isinstance(cursor_value, str) and cursor_value else None
            if not cursor or len(raw_markets) >= limit:
                break

        return raw_markets

    def _fetch_hint_markets(self, query: str, limit: int) -> list[Mapping[str, Any]]:
        raw_markets: list[Mapping[str, Any]] = []
        for series_ticker in _series_ticker_hints(query):
            if series_ticker in self._series_markets_cache:
                raw_markets.extend(self._series_markets_cache[series_ticker])
                continue
            payload = self._get_json(
                "/markets",
                params={
                    "limit": max(1, min(500, limit)),
                    "status": "open",
                    "series_ticker": series_ticker,
                },
            )
            if not isinstance(payload, Mapping):
                continue
            markets = payload.get("markets")
            if not isinstance(markets, Sequence) or isinstance(markets, (str, bytes)):
                continue
            series_markets: list[Mapping[str, Any]] = []
            for raw_market in markets:
                if isinstance(raw_market, Mapping):
                    series_markets.append(raw_market)
            self._series_markets_cache[series_ticker] = series_markets
            raw_markets.extend(series_markets)
        return raw_markets

    def _get_active_market_candidates(self, limit: int) -> list[Mapping[str, Any]]:
        if self._active_markets_cache is None:
            self._active_markets_cache = self._fetch_market_pages(
                limit=limit,
                max_pages=self.max_pages,
                status="open",
            )
        return self._active_markets_cache

    def _get_json(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        fallback_404: bool = False,
    ) -> Any | None:
        """GET JSON from Kalshi with logging and graceful failure."""

        url = f"{self.config.kalshi_base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.request_timeout_seconds,
            )
            if fallback_404 and response.status_code == 404:
                return None
            response.raise_for_status()
            self.last_error = None
            return response.json()
        except requests.RequestException as exc:
            self.last_error = f"Kalshi API request failed: {exc}"
            logger.warning("%s url=%s params=%s", self.last_error, url, params)
        except ValueError as exc:
            self.last_error = f"Kalshi API returned invalid JSON: {exc}"
            logger.warning("%s url=%s params=%s", self.last_error, url, params)
        return None

    def _normalize_many(
        self,
        raw_markets: Sequence[Mapping[str, Any]],
        limit: int,
    ) -> list[NormalizedMarket]:
        markets: list[NormalizedMarket] = []
        seen_ids: set[str] = set()
        for raw_market in raw_markets:
            try:
                market = self.normalize_market(raw_market)
            except ValueError as exc:
                logger.warning("Skipping malformed Kalshi market: %s", exc)
                continue
            if market.market_id in seen_ids:
                continue
            seen_ids.add(market.market_id)
            markets.append(market)
            if len(markets) >= limit:
                break
        return markets


def _build_session() -> Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "prediction-market-intelligence-agent/0.1 read-only research app",
        }
    )
    return session


def _market_similarity(query: str, raw_market: Mapping[str, Any]) -> float:
    search_text = _search_text(raw_market)
    return text_similarity(query, search_text)


def _has_shared_search_tokens(query: str, raw_market: Mapping[str, Any], score: float) -> bool:
    if score >= 0.65:
        return True
    return bool(token_set(query) & token_set(_search_text(raw_market)))


def _series_ticker_hints(query: str) -> list[str]:
    text = query.lower()
    hints: list[str] = []
    if any(term in text for term in ("fed", "fomc", "rate cut", "interest rate", "powell")):
        hints.extend(
            [
                "KXRATECUT",
                "KXFED",
                "KXFEDDECISION",
                "KXFEDMEET",
                "KXFOMCDISSENTCOUNT",
                "KXLARGECUT",
                "KXTRYFIREPOWELL",
            ]
        )
    if any(term in text for term in ("trump", "president", "impeach", "veto")):
        hints.extend(["KXIMPEACH", "KXVETOCOUNT", "KXTRUTHSOCIAL", "KXTRYFIREPOWELL"])
    if any(term in text for term in ("inflation", "cpi", "prices")):
        hints.extend(["KXCPIYOY", "KXCPICOREYOY", "KXLCPIMAXYOY", "KXHIGHINFLATION"])
    if any(term in text for term in ("gas", "oil", "gasoline", "crude")):
        hints.extend(["KXOIL", "KXGAS"])

    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        deduped.append(hint)
    return deduped


def _search_text(raw_market: Mapping[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            _text(raw_market.get("title")),
            _text(raw_market.get("subtitle")),
            _text(raw_market.get("yes_sub_title")),
            _text(raw_market.get("no_sub_title")),
            _text(raw_market.get("rules_primary")),
            _text(raw_market.get("rules_secondary")),
            _text(raw_market.get("category")),
            _text(raw_market.get("event_ticker")),
            _text(raw_market.get("ticker")),
        )
        if part
    )


def _description(raw_market: Mapping[str, Any]) -> str | None:
    parts = [
        _text(raw_market.get("subtitle")),
        _text(raw_market.get("yes_sub_title")),
        _text(raw_market.get("no_sub_title")),
        _text(raw_market.get("rules_primary")),
        _text(raw_market.get("rules_secondary")),
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return "\n\n".join(deduped) or None


def _normalize_price(value: Any) -> float | None:
    """Normalize Kalshi prices expressed as cents, decimals, or dollar strings."""

    number = _float_or_none(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 1.0 < number <= 100.0:
        return number / 100.0
    return None


def _first_price(*values: Any) -> float | None:
    for value in values:
        price = _normalize_price(value)
        if price is not None:
            return price
    return None


def _midpoint(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return _clamp_probability((bid + ask) / 2.0)


def _clamp_probability(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def _first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        number = _float_or_none(value)
        if number is not None:
            return number
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _market_url(ticker: str | None) -> str | None:
    if not ticker:
        return None
    return f"https://kalshi.com/markets/{ticker}"


def _parse_datetime(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
