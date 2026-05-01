"""Tests for deterministic probability extraction."""

from pmi_agent.engine.probability_engine import ProbabilityEngine
from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.schemas import InterpretedQuestion, NormalizedMarket, RankedMarket


def test_binary_yes_market_maps_to_yes_price() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    probability = ProbabilityEngine().extract_probability(question, _ranked_market())

    assert probability.target_outcome == "Yes"
    assert probability.implied_probability == 0.64
    assert probability.probability_source == "yes_price"


def test_question_asking_not_maps_to_no_price_when_obvious() -> None:
    question = QueryInterpreter().interpret("Will Trump not be impeached?")
    ranked = _ranked_market(title="Will Trump be impeached in 2026?", yes=0.25, no=0.75)
    probability = ProbabilityEngine().extract_probability(question, ranked)

    assert probability.target_outcome == "No"
    assert probability.implied_probability == 0.75


def test_outcome_prices_mapping_works() -> None:
    question = InterpretedQuestion(
        original_question="Will OpenAI IPO before 2027?",
        normalized_question="Will OpenAI IPO before 2027?",
        category="finance",
        target_event="OpenAI IPO",
        expected_outcome="OpenAI IPO",
    )
    ranked = _ranked_market(outcome_prices={"OpenAI IPO": 0.42, "No IPO": 0.58}, yes=None, no=None)
    probability = ProbabilityEngine().extract_probability(question, ranked)

    assert probability.target_outcome == "OpenAI IPO"
    assert probability.implied_probability == 0.42


def test_missing_price_produces_warning_and_none() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = _ranked_market(yes=None, no=None, outcome_prices={})
    probability = ProbabilityEngine().extract_probability(question, ranked)

    assert probability.implied_probability is None
    assert "missing price" in probability.warnings
    assert "unclear outcome mapping" in probability.warnings


def test_low_liquidity_warning() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    probability = ProbabilityEngine().extract_probability(question, _ranked_market(liquidity=100))

    assert "low liquidity" in probability.warnings


def test_inactive_market_warning() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    probability = ProbabilityEngine().extract_probability(
        question,
        _ranked_market(active=False, closed=True),
    )

    assert "inactive or closed market" in probability.warnings


def test_unclear_outcome_mapping_warning() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = _ranked_market(outcome_prices={"Red": 0.5, "Blue": 0.5}, yes=None, no=None)
    probability = ProbabilityEngine().extract_probability(question, ranked)

    assert probability.implied_probability is None
    assert "unclear outcome mapping" in probability.warnings


def test_conditional_market_receives_lower_weight() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    clean = _ranked_market()
    conditional = clean.model_copy(
        update={
            "is_conditional": True,
            "is_compound": True,
            "has_scope_mismatch": True,
            "penalty_reasons": ["Conditional or compound market does not directly resolve the user question."],
        }
    )

    clean_probability = ProbabilityEngine().extract_probability(question, clean)
    conditional_probability = ProbabilityEngine().extract_probability(question, conditional)

    assert conditional_probability.market_weight < clean_probability.market_weight
    assert "conditional market" in conditional_probability.warnings
    assert "compound market" in conditional_probability.warnings


def test_penalty_reasons_propagate_to_market_probability_warnings() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = _ranked_market().model_copy(
        update={
            "penalty_reasons": ["Market scope is narrower or different than the interpreted event."],
            "has_scope_mismatch": True,
        }
    )
    probability = ProbabilityEngine().extract_probability(question, ranked)

    assert "Market scope is narrower or different than the interpreted event." in probability.warnings
    assert "scope mismatch" in probability.warnings


def test_kalshi_zero_liquidity_with_volume_and_spread_uses_fallback_quality() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = _ranked_market(source="kalshi", liquidity=0, volume=2_500, spread=0.04)

    probability = ProbabilityEngine().extract_probability(question, ranked)

    assert probability.liquidity_score == 0.55
    assert probability.market_weight > 0
    assert probability.provider_quality_note == "Liquidity unavailable/zero; using bid-ask/volume fallback."
    assert "Provider reported zero liquidity; weight uses fallback market-quality logic." in probability.warnings


def test_market_without_liquidity_volume_or_spread_gets_low_quality_scores() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = _ranked_market(source="kalshi", liquidity=None, volume=None, spread=None)

    probability = ProbabilityEngine().extract_probability(question, ranked)

    assert probability.liquidity_score == 0.25
    assert probability.volume_score == 0.35
    assert probability.spread_score == 0.75
    assert "Low volume or missing volume." in probability.warnings


def test_fallback_weight_remains_below_high_liquidity_equivalent() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    fallback = _ranked_market(source="kalshi", liquidity=0, volume=2_500, spread=0.04)
    high_quality = _ranked_market(source="kalshi", liquidity=100_000, volume=2_500, spread=0.04)

    fallback_probability = ProbabilityEngine().extract_probability(question, fallback)
    high_quality_probability = ProbabilityEngine().extract_probability(question, high_quality)

    assert fallback_probability.market_weight < high_quality_probability.market_weight


def _ranked_market(
    title: str = "Will the Fed cut interest rates by September?",
    yes: float | None = 0.64,
    no: float | None = 0.36,
    outcome_prices: dict[str, float] | None = None,
    liquidity: float | None = 100_000,
    volume: float | None = 500_000,
    spread: float | None = 0.02,
    source: str = "polymarket",
    active: bool = True,
    closed: bool = False,
) -> RankedMarket:
    prices = outcome_prices if outcome_prices is not None else {"Yes": yes, "No": no}
    prices = {key: value for key, value in prices.items() if value is not None}
    return RankedMarket(
        market=NormalizedMarket(
            market_id="m1",
            source=source,
            title=title,
            description="This market resolves Yes if the event occurs.",
            implied_probability=yes,
            yes_price=yes,
            no_price=no,
            outcome_prices=prices,
            volume_usd=volume,
            liquidity_usd=liquidity,
            active=active,
            closed=closed,
            raw={} if spread is None else {"spread": spread},
        ),
        evidence_type="Direct",
        relevance_score=0.9,
        semantic_similarity=0.9,
        entity_overlap=0.9,
        timeframe_alignment=1.0,
        outcome_alignment=1.0,
        category_alignment=1.0,
        resolution_clarity=1.0,
        market_quality=1.0,
        rationale="test",
    )
