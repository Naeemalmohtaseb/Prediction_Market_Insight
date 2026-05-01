"""Tests for deterministic market aggregation."""

from pmi_agent.engine.aggregation_engine import AggregationEngine
from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.schemas import NormalizedMarket, RankedMarket


def test_one_strong_direct_market_returns_its_probability() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    result = AggregationEngine().aggregate(question, [_ranked("Direct", 0.9, 0.7)])

    assert round(result.estimated_probability or 0, 3) == 0.700
    assert round(result.direct_market_probability or 0, 3) == 0.700


def test_direct_plus_related_uses_85_15_blend() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    result = AggregationEngine().aggregate(
        question,
        [_ranked("Direct", 0.9, 0.8), _ranked("Related", 0.6, 0.2, title="Will inflation fall?")],
    )

    assert round(result.estimated_probability or 0, 3) == 0.710


def test_weak_direct_plus_related_uses_65_35_blend() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    result = AggregationEngine().aggregate(
        question,
        [
            _ranked("Near-direct", 0.66, 0.8, liquidity=None, volume=None),
            _ranked("Related", 0.6, 0.2, title="Will inflation fall?"),
        ],
    )

    assert round(result.estimated_probability or 0, 3) == 0.590


def test_no_direct_market_uses_related_probability_and_warning() -> None:
    question = QueryInterpreter().interpret("Will gas prices rise this summer?")
    result = AggregationEngine().aggregate(
        question,
        [_ranked("Related", 0.6, 0.35, title="Will oil prices rise this summer?")],
    )

    assert result.estimated_probability == 0.35
    assert "No direct market found; estimate is based on related or weak market signals." in result.key_warnings


def test_no_usable_probabilities_returns_none() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    result = AggregationEngine().aggregate(
        question,
        [_ranked("Direct", 0.9, None, yes=None, no=None, outcome_prices={})],
    )

    assert result.estimated_probability is None
    assert result.confidence_label == "Low"
    assert "No usable market-implied probability found." in result.key_warnings


def test_demoted_conditional_markets_do_not_count_as_direct() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    conditional = _ranked("Near-direct", 0.66, 0.2).model_copy(
        update={
            "is_conditional": True,
            "is_compound": True,
            "has_scope_mismatch": True,
            "penalty_reasons": ["Conditional or compound market does not directly resolve the user question."],
        }
    )
    result = AggregationEngine().aggregate(question, [conditional])

    assert result.direct_markets_count == 0
    assert result.related_markets_count == 1
    assert result.direct_market_probability is None


def test_warning_when_multiple_high_similarity_markets_are_downweighted() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    markets = []
    for index in range(2):
        markets.append(
            _ranked("Related", 0.55, 0.2, title=f"Will the Fed cut rates before Person {index} is confirmed?").model_copy(
                update={
                    "is_conditional": True,
                    "is_compound": True,
                    "penalty_reasons": ["Conditional or compound market does not directly resolve the user question."],
                }
            )
        )

    result = AggregationEngine().aggregate(question, markets)

    assert "Several high-similarity markets were conditional or compound and were downweighted." in result.key_warnings


def test_provider_summary_and_disagreement_are_included() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    result = AggregationEngine().aggregate(
        question,
        [
            _ranked("Direct", 0.9, 0.30, source="polymarket"),
            _ranked("Direct", 0.9, 0.50, source="kalshi", liquidity=0, volume=5_000),
        ],
    )

    assert result.provider_summary is not None
    assert set(result.provider_summary) == {"polymarket", "kalshi"}
    assert result.provider_disagreement_score is not None
    assert result.provider_disagreement_score > 0.15
    assert "Providers show substantial disagreement." in result.provider_notes


def _ranked(
    evidence_type: str,
    relevance: float,
    probability: float | None,
    title: str = "Will the Fed cut interest rates by September?",
    yes: float | None = None,
    no: float | None = None,
    outcome_prices: dict[str, float] | None = None,
    liquidity: float | None = 100_000,
    volume: float | None = 1_000_000,
    source: str = "polymarket",
) -> RankedMarket:
    yes_price = probability if yes is None else yes
    no_price = (1 - probability) if probability is not None and no is None else no
    prices = outcome_prices if outcome_prices is not None else {"Yes": yes_price, "No": no_price}
    prices = {key: value for key, value in prices.items() if value is not None}
    return RankedMarket(
        market=NormalizedMarket(
            market_id=f"{evidence_type}-{title}",
            source=source,
            title=title,
            description="This market resolves Yes if the event occurs.",
            implied_probability=yes_price,
            yes_price=yes_price,
            no_price=no_price,
            outcome_prices=prices,
            volume_usd=volume,
            liquidity_usd=liquidity,
            active=True,
            closed=False,
            raw={"spread": 0.02},
        ),
        evidence_type=evidence_type,
        relevance_score=relevance,
        semantic_similarity=relevance,
        entity_overlap=relevance,
        timeframe_alignment=1.0,
        outcome_alignment=1.0,
        category_alignment=1.0,
        resolution_clarity=1.0,
        market_quality=1.0,
        rationale="test",
    )
