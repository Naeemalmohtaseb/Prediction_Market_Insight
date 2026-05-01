"""Tests for deterministic confidence scoring."""

from pmi_agent.engine.confidence_engine import ConfidenceEngine
from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.schemas import MarketProbability, NormalizedMarket


def test_strong_direct_liquid_agreement_case_gives_higher_confidence() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    market_probabilities = [_prob("Direct", 0.7, 0.5), _prob("Near-direct", 0.72, 0.4)]

    score, label, warnings = ConfidenceEngine().score(question, market_probabilities, 0.71, None, 0.02)

    assert score >= 70
    assert label == "High"
    assert "No direct market found" not in warnings


def test_no_direct_market_lowers_confidence() -> None:
    question = QueryInterpreter().interpret("Will gas prices rise this summer?")
    score, label, warnings = ConfidenceEngine().score(
        question,
        [_prob("Related", 0.4, 0.25)],
        None,
        0.4,
        None,
    )

    assert score < 70
    assert label in {"Low", "Medium"}
    assert "No direct market found" in warnings


def test_high_disagreement_lowers_confidence() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    baseline_score, _baseline_label, _baseline_warnings = ConfidenceEngine().score(
        question,
        [_prob("Direct", 0.1, 0.4), _prob("Near-direct", 0.9, 0.4)],
        0.5,
        None,
        0.02,
    )
    score, _label, warnings = ConfidenceEngine().score(
        question,
        [_prob("Direct", 0.1, 0.4), _prob("Near-direct", 0.9, 0.4)],
        0.5,
        None,
        0.4,
    )

    assert score < baseline_score
    assert "High disagreement across markets" in warnings


def test_high_provider_disagreement_lowers_confidence() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    baseline_score, _label, _warnings = ConfidenceEngine().score(
        question,
        [_prob("Direct", 0.55, 0.4), _prob("Near-direct", 0.57, 0.4)],
        0.56,
        None,
        0.02,
        0.03,
    )
    score, _label, warnings = ConfidenceEngine().score(
        question,
        [_prob("Direct", 0.55, 0.4), _prob("Near-direct", 0.57, 0.4)],
        0.56,
        None,
        0.02,
        0.25,
    )

    assert score < baseline_score
    assert "Prediction market providers disagree materially." in warnings


def test_provider_agreement_does_not_overinflate_confidence() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    score_without_provider, _label, _warnings = ConfidenceEngine().score(
        question,
        [_prob("Direct", 0.55, 0.4), _prob("Near-direct", 0.57, 0.4)],
        0.56,
        None,
        0.02,
        None,
    )
    score_with_agreement, _label, _warnings = ConfidenceEngine().score(
        question,
        [_prob("Direct", 0.55, 0.4), _prob("Near-direct", 0.57, 0.4)],
        0.56,
        None,
        0.02,
        0.03,
    )

    assert score_with_agreement <= score_without_provider


def _prob(evidence_type: str, probability: float, weight: float) -> MarketProbability:
    return MarketProbability(
        market=NormalizedMarket(
            market_id=f"{evidence_type}-{probability}",
            source="polymarket",
            title="Will the Fed cut rates?",
            description="This market resolves Yes if the event occurs.",
            implied_probability=probability,
            yes_price=probability,
            no_price=1 - probability,
            outcome_prices={"Yes": probability, "No": 1 - probability},
            volume_usd=1_000_000,
            liquidity_usd=100_000,
            active=True,
            closed=False,
        ),
        evidence_type=evidence_type,
        relevance_score=0.9,
        target_outcome="Yes",
        implied_probability=probability,
        probability_source="yes_price",
        market_weight=weight,
        liquidity_score=1.0,
        volume_score=1.0,
        spread_score=0.9,
        recency_score=1.0,
        resolution_score=1.0,
    )
