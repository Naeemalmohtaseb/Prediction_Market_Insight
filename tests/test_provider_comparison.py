"""Tests for provider-level evidence comparison."""

from pmi_agent.engine.provider_comparison import compare_providers
from pmi_agent.schemas import MarketProbability, NormalizedMarket


def test_close_provider_probabilities_broadly_agree() -> None:
    summary, disagreement, notes = compare_providers(
        [_prob("polymarket", 0.50, 0.4), _prob("kalshi", 0.54, 0.3)]
    )

    assert round(disagreement or 0, 3) == 0.040
    assert "Providers broadly agree." in notes
    assert summary["polymarket"]["usable_markets_count"] == 1
    assert summary["kalshi"]["usable_markets_count"] == 1


def test_far_apart_provider_probabilities_show_substantial_disagreement() -> None:
    _summary, disagreement, notes = compare_providers(
        [_prob("polymarket", 0.25, 0.4), _prob("kalshi", 0.55, 0.3)]
    )

    assert round(disagreement or 0, 3) == 0.300
    assert "Providers show substantial disagreement." in notes


def test_one_provider_without_usable_markets_gets_limited_validation_note() -> None:
    _summary, disagreement, notes = compare_providers(
        [_prob("polymarket", 0.25, 0.4), _prob("kalshi", None, 0.0)]
    )

    assert disagreement is None
    assert "Only one provider produced usable probability evidence." in notes


def test_weaker_provider_quality_note_when_total_weight_is_much_lower() -> None:
    _summary, _disagreement, notes = compare_providers(
        [_prob("polymarket", 0.50, 1.0), _prob("kalshi", 0.52, 0.05)]
    )

    assert "Provider comparison is limited by weaker market quality from kalshi." in notes


def _prob(provider: str, probability: float | None, weight: float) -> MarketProbability:
    return MarketProbability(
        market=NormalizedMarket(
            market_id=f"{provider}-{probability}",
            source=provider,
            title="Will the Fed cut rates?",
            description="This market resolves Yes if the event occurs.",
            implied_probability=probability,
            yes_price=probability,
            no_price=None if probability is None else 1 - probability,
            outcome_prices={} if probability is None else {"Yes": probability, "No": 1 - probability},
            volume_usd=10_000,
            liquidity_usd=10_000,
            active=True,
            closed=False,
        ),
        evidence_type="Direct",
        relevance_score=0.9,
        target_outcome="Yes",
        implied_probability=probability,
        probability_source="yes_price" if probability is not None else None,
        market_weight=weight,
        liquidity_score=0.7 if weight else 0.0,
        volume_score=0.7 if weight else 0.0,
        spread_score=0.8 if weight else 0.0,
    )
