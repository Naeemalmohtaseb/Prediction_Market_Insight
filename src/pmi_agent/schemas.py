"""Shared Pydantic schemas for market interpretation and forecasting."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

MarketSource = Literal["polymarket", "kalshi", "unknown"]
QuestionCategory = Literal[
    "politics",
    "macroeconomics",
    "finance",
    "technology",
    "consumer_products",
    "entertainment",
    "geopolitical_risk",
    "sports",
    "other",
]
EvidenceType = Literal["Direct", "Near-direct", "Related", "Weak", "Irrelevant"]


class InterpretedQuestion(BaseModel):
    """Structured representation of a user forecasting question."""

    original_question: str
    normalized_question: str
    category: QuestionCategory = "other"
    target_event: str
    expected_outcome: str
    timeframe: str | None = None
    entities: list[str] = Field(default_factory=list)
    geography: str | None = None
    resolution_criteria: str | None = None
    search_terms: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)

    @property
    def core_event(self) -> str:
        """Backward-compatible alias for earlier scaffold code."""

        return self.target_event


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
    evidence_type: EvidenceType
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    entity_overlap: float = Field(ge=0.0, le=1.0)
    timeframe_alignment: float = Field(ge=0.0, le=1.0)
    outcome_alignment: float = Field(ge=0.0, le=1.0)
    category_alignment: float = Field(ge=0.0, le=1.0)
    resolution_clarity: float = Field(ge=0.0, le=1.0)
    market_quality: float = Field(ge=0.0, le=1.0)
    rationale: str
    directness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    liquidity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rank_reasons: list[str] = Field(default_factory=list)
    is_conditional: bool = False
    is_compound: bool = False
    has_scope_mismatch: bool = False
    has_timeframe_conflict: bool = False
    has_entity_conflict: bool = False
    penalty_reasons: list[str] = Field(default_factory=list)


class MarketProbability(BaseModel):
    """A ranked market with an extracted target-outcome probability."""

    market: NormalizedMarket
    evidence_type: EvidenceType
    relevance_score: float = Field(ge=0.0, le=1.0)
    target_outcome: str | None = None
    implied_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_source: str | None = None
    market_weight: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    liquidity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    volume_score: float | None = Field(default=None, ge=0.0, le=1.0)
    spread_score: float | None = Field(default=None, ge=0.0, le=1.0)
    recency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    resolution_score: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_quality_note: str | None = None


class ContextItem(BaseModel):
    """Recent contextual item used for background, not probability estimation."""

    title: str
    source: str | None = None
    url: str | None = None
    published_at: str | None = None
    summary: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    query_term: str | None = None
    raw: dict[str, Any] | None = None


class ForecastResult(BaseModel):
    """Deterministic forecast output derived from ranked markets."""

    estimated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence_label: str = "Low"
    direct_market_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    related_signal_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    disagreement_score: float | None = Field(default=None, ge=0.0, le=1.0)
    direct_markets_count: int = Field(default=0, ge=0)
    related_markets_count: int = Field(default=0, ge=0)
    markets_used_count: int = Field(default=0, ge=0)
    key_warnings: list[str] = Field(default_factory=list)
    market_probabilities: list[MarketProbability] = Field(default_factory=list)
    provider_summary: dict[str, Any] | None = None
    provider_disagreement_score: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_notes: list[str] = Field(default_factory=list)
    context_items: list[ContextItem] = Field(default_factory=list)
    context_summary: str | None = None
    context_warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ForecastReport(BaseModel):
    """User-facing report text plus its auditable deterministic result."""

    question: InterpretedQuestion
    result: ForecastResult
    markdown_report: str
