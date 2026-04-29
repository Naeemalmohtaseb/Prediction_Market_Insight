"""Read-only Polymarket Gamma API client."""

from __future__ import annotations

import json
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

logger = logging.getLogger(__name__)


class PolymarketClient(PredictionMarketClient):
    """Read-only adapter for Polymarket public market discovery endpoints."""

    def __init__(
        self,
        config: AppConfig | None = None,
        session: Session | None = None,
    ) -> None:
        self.config = config or get_config()
        self.session = session or _build_session()
        self.last_error: str | None = None

    def search_markets(self, query: str, limit: int = 25) -> list[NormalizedMarket]:
        """Search Polymarket public results and return normalized markets."""

        query = query.strip()
        if not query:
            return []

        payload = self._get_json(
            "/public-search",
            params={
                "q": query,
                "limit_per_type": limit,
                "search_tags": "false",
                "search_profiles": "false",
                "cache": "true",
            },
        )
        if payload is None:
            return []

        raw_markets = _extract_markets_from_search(payload)
        return self._normalize_many(raw_markets, limit=limit)

    def list_active_markets(self, limit: int = 100) -> list[NormalizedMarket]:
        """List active, non-closed markets through Gamma events discovery."""

        payload = self._get_json(
            "/events",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "order": "volume_24hr",
                "ascending": "false",
            },
        )
        if payload is None:
            return []

        raw_markets = _extract_markets_from_events(payload)
        return self._normalize_many(raw_markets, limit=limit)

    def get_market_by_id(self, market_id: str) -> NormalizedMarket | None:
        """Fetch one market by Gamma market id."""

        clean_id = market_id.strip()
        if not clean_id:
            return None

        payload = self._get_json(f"/markets/{clean_id}", fallback_404=True)
        if payload is None:
            payload = self._get_json("/markets", params={"id": clean_id}, fallback_404=True)

        if payload is None:
            return None

        raw_market: Mapping[str, Any] | None
        if isinstance(payload, list):
            raw_market = payload[0] if payload and isinstance(payload[0], Mapping) else None
        elif isinstance(payload, Mapping):
            raw_market = payload
        else:
            raw_market = None

        if raw_market is None:
            return None

        try:
            return self.normalize_market(raw_market)
        except ValueError as exc:
            logger.warning("Unable to normalize Polymarket market %s: %s", clean_id, exc)
            self.last_error = str(exc)
            return None

    def normalize_market(self, raw_market: Mapping[str, Any]) -> NormalizedMarket:
        """Normalize a raw Polymarket market record."""

        market_id = _text(raw_market.get("id") or raw_market.get("conditionId"))
        title = _text(raw_market.get("question") or raw_market.get("title"))
        if not market_id:
            raise ValueError("Polymarket record is missing id and conditionId.")
        if not title:
            title = f"Polymarket market {market_id}"

        outcomes = _parse_text_list(raw_market.get("outcomes"))
        prices = _parse_float_list(raw_market.get("outcomePrices"))
        outcome_prices = _map_outcome_prices(outcomes, prices)
        yes_price = _case_insensitive_lookup(outcome_prices, "Yes")
        no_price = _case_insensitive_lookup(outcome_prices, "No")

        implied_probability = yes_price
        if implied_probability is None:
            implied_probability = _first_probability(
                raw_market.get("impliedProbability"),
                raw_market.get("probability"),
                raw_market.get("lastTradePrice"),
                raw_market.get("bestAsk"),
                raw_market.get("bestBid"),
            )

        slug = _text(raw_market.get("slug"))

        return NormalizedMarket(
            market_id=market_id,
            source="polymarket",
            title=title,
            description=_text(raw_market.get("description")),
            url=_market_url(slug),
            outcome_name="Yes" if yes_price is not None else (outcomes[0] if outcomes else "Yes"),
            implied_probability=implied_probability,
            yes_price=yes_price,
            no_price=no_price,
            outcome_prices=outcome_prices,
            volume_usd=_first_float(raw_market.get("volumeNum"), raw_market.get("volume")),
            liquidity_usd=_first_float(raw_market.get("liquidityNum"), raw_market.get("liquidity")),
            close_time=_parse_datetime(
                raw_market.get("endDateIso") or raw_market.get("endDate") or raw_market.get("closeTime")
            ),
            active=_optional_bool(raw_market.get("active")),
            closed=_optional_bool(raw_market.get("closed")),
            slug=slug,
            raw=dict(raw_market),
        )

    def _get_json(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        fallback_404: bool = False,
    ) -> Any | None:
        """GET JSON from Gamma with logging and graceful failure."""

        url = f"{self.config.polymarket_base_url.rstrip('/')}/{path.lstrip('/')}"
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
            self.last_error = f"Polymarket API request failed: {exc}"
            logger.warning("%s url=%s params=%s", self.last_error, url, params)
        except ValueError as exc:
            self.last_error = f"Polymarket API returned invalid JSON: {exc}"
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
                logger.warning("Skipping malformed Polymarket market: %s", exc)
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


def _extract_markets_from_search(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []

    raw_markets: list[Mapping[str, Any]] = []
    for event in payload.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        event_context = {
            "event_id": event.get("id"),
            "event_slug": event.get("slug"),
            "event_title": event.get("title"),
        }
        for market in event.get("markets") or []:
            if isinstance(market, Mapping):
                raw_markets.append({**event_context, **market})
    return raw_markets


def _extract_markets_from_events(payload: Any) -> list[Mapping[str, Any]]:
    events = payload if isinstance(payload, list) else payload.get("events", []) if isinstance(payload, Mapping) else []
    raw_markets: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_context = {
            "event_id": event.get("id"),
            "event_slug": event.get("slug"),
            "event_title": event.get("title"),
        }
        for market in event.get("markets") or []:
            if isinstance(market, Mapping):
                raw_markets.append({**event_context, **market})
    return raw_markets


def _parse_text_list(value: Any) -> list[str]:
    parsed = _parse_json_if_string(value)
    if isinstance(parsed, list):
        return [_text(item) for item in parsed if _text(item)]
    return []


def _parse_float_list(value: Any) -> list[float | None]:
    parsed = _parse_json_if_string(value)
    if not isinstance(parsed, list):
        return []
    return [_probability_or_none(item) for item in parsed]


def _parse_json_if_string(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [text]
    return value


def _map_outcome_prices(outcomes: list[str], prices: list[float | None]) -> dict[str, float]:
    outcome_prices: dict[str, float] = {}
    for index, outcome in enumerate(outcomes):
        if index >= len(prices):
            break
        price = prices[index]
        if price is None:
            continue
        outcome_prices[outcome] = price
    return outcome_prices


def _case_insensitive_lookup(values: Mapping[str, float], target: str) -> float | None:
    for key, value in values.items():
        if key.casefold() == target.casefold():
            return value
    return None


def _first_probability(*values: Any) -> float | None:
    for value in values:
        probability = _probability_or_none(value)
        if probability is not None:
            return probability
    return None


def _probability_or_none(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 1.0 < number <= 100.0:
        return number / 100.0
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


def _market_url(slug: str | None) -> str | None:
    if not slug:
        return None
    return f"https://polymarket.com/event/{slug}"


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _parse_datetime(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
