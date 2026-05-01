"""SQLite persistence for local analysis snapshots."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from pmi_agent.schemas import ForecastResult, InterpretedQuestion, MarketProbability, NormalizedMarket
from pmi_agent.search.semantic_similarity import text_similarity, token_set


class StorageError(RuntimeError):
    """Raised when local persistence fails."""


class StorageManager:
    """Manage local SQLite persistence for analyses and market snapshots."""

    def __init__(self, db_path: str | Path = "data/pmi_agent.sqlite") -> None:
        self.db_path = Path(db_path)

    def init_db(self) -> None:
        """Create the SQLite database and tables if they do not exist."""

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(SCHEMA_SQL)
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to initialize SQLite database: {exc}") from exc
        except OSError as exc:
            raise StorageError(f"Failed to create database directory: {exc}") from exc

    def save_analysis(
        self,
        question: InterpretedQuestion,
        result: ForecastResult,
        markdown_report: str,
    ) -> int:
        """Persist an analysis and its market snapshots."""

        self.init_db()
        created_at = datetime.now(UTC).isoformat()

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO analyses (
                        created_at, user_question, normalized_question, category,
                        target_event, expected_outcome, geography, timeframe,
                        estimated_probability, confidence_score, confidence_label,
                        direct_market_probability, related_signal_probability,
                        disagreement_score, direct_markets_count, related_markets_count,
                        markets_used_count, key_warnings_json,
                        interpreted_question_json, forecast_result_json, markdown_report
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        question.original_question,
                        question.normalized_question,
                        question.category,
                        question.target_event,
                        question.expected_outcome,
                        question.geography,
                        question.timeframe,
                        result.estimated_probability,
                        result.confidence_score,
                        result.confidence_label,
                        result.direct_market_probability,
                        result.related_signal_probability,
                        result.disagreement_score,
                        result.direct_markets_count,
                        result.related_markets_count,
                        result.markets_used_count,
                        _json_dumps(result.key_warnings),
                        question.model_dump_json(),
                        result.model_dump_json(),
                        markdown_report,
                    ),
                )
                analysis_id = int(cursor.lastrowid)
                conn.executemany(
                    """
                    INSERT INTO market_snapshots (
                        analysis_id, provider, provider_market_id, event_id, title,
                        evidence_type, relevance_score, target_outcome,
                        implied_probability, probability_source, market_weight,
                        yes_price, no_price, volume, liquidity, close_date, active,
                        is_conditional, is_compound, has_scope_mismatch,
                        has_timeframe_conflict, has_entity_conflict, warnings_json,
                        penalty_reasons_json, raw_market_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [_market_snapshot_row(analysis_id, item) for item in result.market_probabilities],
                )
                return analysis_id
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to save analysis: {exc}") from exc

    def list_recent_analyses(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent analyses with summary fields."""

        self.init_db()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, created_at, user_question, normalized_question, category,
                           target_event, estimated_probability, confidence_score,
                           confidence_label, markets_used_count
                    FROM analyses
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [_row_to_dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to list analyses: {exc}") from exc

    def get_analysis(self, analysis_id: int) -> dict[str, Any] | None:
        """Return a saved analysis and its market snapshots."""

        self.init_db()
        try:
            with self._connect() as conn:
                analysis = conn.execute(
                    "SELECT * FROM analyses WHERE id = ?",
                    (analysis_id,),
                ).fetchone()
                if analysis is None:
                    return None
                snapshots = conn.execute(
                    """
                    SELECT * FROM market_snapshots
                    WHERE analysis_id = ?
                    ORDER BY id ASC
                    """,
                    (analysis_id,),
                ).fetchall()
                data = _row_to_dict(analysis)
                data["key_warnings"] = _json_loads(data.pop("key_warnings_json"), [])
                data["interpreted_question"] = _json_loads(data["interpreted_question_json"], {})
                data["forecast_result"] = _json_loads(data["forecast_result_json"], {})
                data["market_snapshots"] = [_decode_snapshot(_row_to_dict(row)) for row in snapshots]
                return data
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to load analysis {analysis_id}: {exc}") from exc

    def delete_analysis(self, analysis_id: int) -> bool:
        """Delete one analysis and its cascade-linked market snapshots."""

        self.init_db()
        try:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to delete analysis {analysis_id}: {exc}") from exc

    def upsert_catalog_markets(self, markets: list[NormalizedMarket]) -> int:
        """Insert or update normalized markets in the local search catalog."""

        self.init_db()
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO market_catalog (
                        provider, provider_market_id, event_id, title, description,
                        outcomes_json, outcome_prices_json, yes_price, no_price,
                        midpoint, spread, volume, liquidity, open_interest, close_date,
                        active, resolution_criteria, url, raw_json, fetched_at,
                        last_seen_at, search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, provider_market_id) DO UPDATE SET
                        event_id = excluded.event_id,
                        title = excluded.title,
                        description = excluded.description,
                        outcomes_json = excluded.outcomes_json,
                        outcome_prices_json = excluded.outcome_prices_json,
                        yes_price = excluded.yes_price,
                        no_price = excluded.no_price,
                        midpoint = excluded.midpoint,
                        spread = excluded.spread,
                        volume = excluded.volume,
                        liquidity = excluded.liquidity,
                        open_interest = excluded.open_interest,
                        close_date = excluded.close_date,
                        active = excluded.active,
                        resolution_criteria = excluded.resolution_criteria,
                        url = excluded.url,
                        raw_json = excluded.raw_json,
                        fetched_at = excluded.fetched_at,
                        last_seen_at = excluded.last_seen_at,
                        search_text = excluded.search_text
                    """,
                    [_catalog_row(market, now) for market in markets],
                )
                return len(markets)
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to upsert market catalog: {exc}") from exc

    def search_catalog(
        self,
        query: str,
        providers: list[str] | None = None,
        limit: int = 50,
        active_only: bool = True,
    ) -> list[NormalizedMarket]:
        """Search locally cached markets and return normalized records."""

        self.init_db()
        clean_query = query.strip()
        if not clean_query:
            return []

        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.append("(active = 1 OR active IS NULL)")
            clauses.append("(close_date IS NULL OR close_date >= ?)")
            params.append(datetime.now(UTC).isoformat())
        if providers:
            placeholders = ", ".join("?" for _ in providers)
            clauses.append(f"provider IN ({placeholders})")
            params.extend(providers)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT * FROM market_catalog
                    {where_sql}
                    ORDER BY last_seen_at DESC
                    LIMIT 1500
                    """,
                    params,
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to search market catalog: {exc}") from exc

        scored: list[tuple[float, sqlite3.Row]] = []
        query_tokens = token_set(clean_query)
        for row in rows:
            search_text = row["search_text"] or ""
            score = text_similarity(clean_query, search_text)
            shared_tokens = query_tokens & token_set(search_text)
            if score >= 0.65 or (score >= 0.20 and shared_tokens):
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [_catalog_row_to_market(row) for _, row in scored[:limit]]

    def get_catalog_stats(self) -> dict[str, Any]:
        """Return summary statistics for the local market catalog."""

        self.init_db()
        try:
            with self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) FROM market_catalog").fetchone()[0]
                by_provider = {
                    row["provider"]: row["count"]
                    for row in conn.execute(
                        "SELECT provider, COUNT(*) AS count FROM market_catalog GROUP BY provider ORDER BY provider"
                    ).fetchall()
                }
                active_by_provider = {
                    row["provider"]: row["count"]
                    for row in conn.execute(
                        """
                        SELECT provider, COUNT(*) AS count
                        FROM market_catalog
                        WHERE active = 1
                        GROUP BY provider
                        ORDER BY provider
                        """
                    ).fetchall()
                }
                row = conn.execute(
                    "SELECT MAX(fetched_at) AS latest_fetched_at, MAX(last_seen_at) AS latest_last_seen_at FROM market_catalog"
                ).fetchone()
                return {
                    "total_markets": total,
                    "by_provider": by_provider,
                    "active_by_provider": active_by_provider,
                    "latest_fetched_at": row["latest_fetched_at"] if row else None,
                    "latest_last_seen_at": row["latest_last_seen_at"] if row else None,
                }
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to load market catalog stats: {exc}") from exc

    def clear_market_catalog(self, provider: str | None = None) -> int:
        """Clear cached catalog markets, optionally for one provider."""

        self.init_db()
        try:
            with self._connect() as conn:
                if provider:
                    cursor = conn.execute("DELETE FROM market_catalog WHERE provider = ?", (provider,))
                else:
                    cursor = conn.execute("DELETE FROM market_catalog")
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to clear market catalog: {exc}") from exc

    def prune_old_catalog(self, days: int = 14) -> int:
        """Remove catalog entries that have not been seen recently."""

        self.init_db()
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        try:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM market_catalog WHERE last_seen_at < ?", (cutoff,))
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to prune market catalog: {exc}") from exc

    def _connect(self) -> Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def connect_sqlite(path: str | Path) -> Connection:
    """Open a SQLite connection for compatibility with the initial scaffold."""

    return sqlite3.connect(Path(path))


def _market_snapshot_row(analysis_id: int, item: MarketProbability) -> tuple[Any, ...]:
    market = item.market
    raw = market.raw or {}
    return (
        analysis_id,
        market.source,
        market.market_id,
        raw.get("event_id") or raw.get("eventId"),
        market.title,
        item.evidence_type,
        item.relevance_score,
        item.target_outcome,
        item.implied_probability,
        item.probability_source,
        item.market_weight,
        market.yes_price,
        market.no_price,
        market.volume_usd,
        market.liquidity_usd,
        market.close_time.isoformat() if market.close_time else None,
        _bool_to_int(market.active),
        _warning_flag(item.warnings, "conditional"),
        _warning_flag(item.warnings, "compound"),
        _warning_flag(item.warnings, "scope"),
        _warning_flag(item.warnings, "timeframe"),
        _warning_flag(item.warnings, "entity"),
        _json_dumps(item.warnings),
        _json_dumps(_penalty_warnings(item.warnings)),
        _json_dumps(raw),
    )


def _catalog_row(market: NormalizedMarket, timestamp: str) -> tuple[Any, ...]:
    raw = market.raw or {}
    event_id = raw.get("event_id") or raw.get("eventId") or raw.get("event_ticker")
    resolution_criteria = raw.get("resolution_criteria") or raw.get("rules_primary") or raw.get("rules")
    outcomes = list(market.outcome_prices.keys())
    search_text = _catalog_search_text(market, event_id, resolution_criteria)
    return (
        market.source,
        market.market_id,
        event_id,
        market.title,
        market.description,
        _json_dumps(outcomes),
        _json_dumps(market.outcome_prices),
        market.yes_price,
        market.no_price,
        raw.get("midpoint") or market.implied_probability,
        _float_or_none(raw.get("spread") or raw.get("bidAskSpread") or raw.get("bid_ask_spread")),
        market.volume_usd,
        market.liquidity_usd,
        _float_or_none(raw.get("open_interest") or raw.get("openInterest") or raw.get("open_interest_fp")),
        market.close_time.isoformat() if market.close_time else None,
        _bool_to_int(market.active),
        resolution_criteria,
        str(market.url) if market.url else None,
        _json_dumps(raw),
        timestamp,
        timestamp,
        search_text,
    )


def _catalog_search_text(market: NormalizedMarket, event_id: Any, resolution_criteria: Any) -> str:
    parts = [
        market.source,
        market.market_id,
        event_id,
        market.title,
        market.description,
        resolution_criteria,
        " ".join(market.outcome_prices.keys()),
    ]
    return " ".join(str(part).strip() for part in parts if part)


def _catalog_row_to_market(row: sqlite3.Row) -> NormalizedMarket:
    raw = _json_loads(row["raw_json"], {})
    if isinstance(raw, dict):
        raw.setdefault("event_id", row["event_id"])
        raw.setdefault("midpoint", row["midpoint"])
        raw.setdefault("spread", row["spread"])
        raw.setdefault("open_interest", row["open_interest"])
        raw.setdefault("catalog_fetched_at", row["fetched_at"])
        raw.setdefault("catalog_last_seen_at", row["last_seen_at"])
    outcome_prices = _json_loads(row["outcome_prices_json"], {})
    if not isinstance(outcome_prices, dict):
        outcome_prices = {}
    return NormalizedMarket(
        market_id=row["provider_market_id"],
        source=row["provider"],
        title=row["title"],
        description=row["description"],
        url=row["url"],
        outcome_name="Yes" if row["yes_price"] is not None else next(iter(outcome_prices), "Yes"),
        implied_probability=row["midpoint"] if row["midpoint"] is not None else row["yes_price"],
        yes_price=row["yes_price"],
        no_price=row["no_price"],
        outcome_prices={str(key): float(value) for key, value in outcome_prices.items()},
        volume_usd=row["volume"],
        liquidity_usd=row["liquidity"],
        close_time=_parse_datetime(row["close_date"]),
        active=_int_to_bool(row["active"]),
        closed=False if row["active"] == 1 else None,
        slug=row["provider_market_id"],
        raw=raw if isinstance(raw, dict) else {},
    )


def _decode_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    row["active"] = _int_to_bool(row["active"])
    for key in (
        "is_conditional",
        "is_compound",
        "has_scope_mismatch",
        "has_timeframe_conflict",
        "has_entity_conflict",
    ):
        row[key] = bool(row[key])
    row["warnings"] = _json_loads(row.pop("warnings_json"), [])
    row["penalty_reasons"] = _json_loads(row.pop("penalty_reasons_json"), [])
    row["raw_market"] = _json_loads(row.pop("raw_market_json"), {})
    return row


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _int_to_bool(value: int | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _warning_flag(warnings: list[str], marker: str) -> int:
    return 1 if any(marker in warning.lower() for warning in warnings) else 0


def _penalty_warnings(warnings: list[str]) -> list[str]:
    markers = (
        "conditional",
        "compound",
        "scope",
        "timeframe",
        "entity",
        "directly resolve",
        "different than",
        "differs",
    )
    return [warning for warning in warnings if any(marker in warning.lower() for marker in markers)]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    user_question TEXT NOT NULL,
    normalized_question TEXT,
    category TEXT,
    target_event TEXT,
    expected_outcome TEXT,
    geography TEXT,
    timeframe TEXT,
    estimated_probability REAL,
    confidence_score REAL,
    confidence_label TEXT,
    direct_market_probability REAL,
    related_signal_probability REAL,
    disagreement_score REAL,
    direct_markets_count INTEGER,
    related_markets_count INTEGER,
    markets_used_count INTEGER,
    key_warnings_json TEXT,
    interpreted_question_json TEXT NOT NULL,
    forecast_result_json TEXT NOT NULL,
    markdown_report TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    provider TEXT,
    provider_market_id TEXT,
    event_id TEXT,
    title TEXT,
    evidence_type TEXT,
    relevance_score REAL,
    target_outcome TEXT,
    implied_probability REAL,
    probability_source TEXT,
    market_weight REAL,
    yes_price REAL,
    no_price REAL,
    volume REAL,
    liquidity REAL,
    close_date TEXT,
    active INTEGER,
    is_conditional INTEGER,
    is_compound INTEGER,
    has_scope_mismatch INTEGER,
    has_timeframe_conflict INTEGER,
    has_entity_conflict INTEGER,
    warnings_json TEXT,
    penalty_reasons_json TEXT,
    raw_market_json TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS market_catalog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    provider_market_id TEXT NOT NULL,
    event_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    outcomes_json TEXT,
    outcome_prices_json TEXT,
    yes_price REAL,
    no_price REAL,
    midpoint REAL,
    spread REAL,
    volume REAL,
    liquidity REAL,
    open_interest REAL,
    close_date TEXT,
    active INTEGER,
    resolution_criteria TEXT,
    url TEXT,
    raw_json TEXT,
    fetched_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    search_text TEXT NOT NULL,
    UNIQUE(provider, provider_market_id)
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_analyses_category ON analyses(category);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_analysis_id ON market_snapshots(analysis_id);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_provider ON market_snapshots(provider);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_provider_market_id ON market_snapshots(provider_market_id);
CREATE INDEX IF NOT EXISTS idx_market_catalog_provider ON market_catalog(provider);
CREATE INDEX IF NOT EXISTS idx_market_catalog_provider_market_id ON market_catalog(provider_market_id);
CREATE INDEX IF NOT EXISTS idx_market_catalog_active ON market_catalog(active);
CREATE INDEX IF NOT EXISTS idx_market_catalog_fetched_at ON market_catalog(fetched_at);
CREATE INDEX IF NOT EXISTS idx_market_catalog_last_seen_at ON market_catalog(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_market_catalog_close_date ON market_catalog(close_date);
"""
