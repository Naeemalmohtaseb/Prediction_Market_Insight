"""Shared Pydantic schemas for market interpretation and forecasting."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

MarketSource = Literal["polymarket", "unknown"]


class InterpretedQuestion(BaseModel):
    """Structured representation of a user forecasting question."""

    original_question: str
    normalized_question: str
    core_event: str
    timeframe: str | None = None
    entities: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)


class NormalizedMarket(BaseModel):
    """Provider-neutral prediction market representation."""

    market_id: str
    source: MarketSource = "unknown"
    title: str
    description: str | None = None
    url: HttpUrl | None = None
    outcome_name: str = "Yes"
    implied_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    yes_price: float | None = Field(default=None, ge=0.0, le=1.0)
    no_price: float | None = Field(default=None, ge=0.0, le=1.0)
    outcome_prices: dict[str, float] = Field(default_factory=dict)
    volume_usd: float | None = Field(default=None, ge=0.0)
    liquidity_usd: float | None = Field(default=None, ge=0.0)
    close_time: datetime | None = None
    active: bool | None = None
    closed: bool | None = None
    slug: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class RankedMarket(BaseModel):
    """A normalized market with deterministic ranking signals."""

    market: NormalizedMarket
    relevance_score: float = Field(ge=0.0, le=1.0)
    directness_score: float = Field(ge=0.0, le=1.0)
    liquidity_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    rank_reasons: list[str] = Field(default_factory=list)


class ForecastResult(BaseModel):
    """Deterministic forecast output derived from ranked markets."""

    question: InterpretedQuestion
    ranked_markets: list[RankedMarket] = Field(default_factory=list)
    aggregate_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    method_notes: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ForecastReport(BaseModel):
    """User-facing report text plus its auditable deterministic result."""

    result: ForecastResult
    executive_summary: str
    market_evidence: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This report is for research and informational purposes only. "
        "It is not betting, trading, financial, legal, or investment advice."
    )
