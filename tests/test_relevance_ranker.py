"""Tests for deterministic relevance ranking."""

from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.schemas import NormalizedMarket
from pmi_agent.search.relevance_ranker import RelevanceRanker


def test_exact_direct_market_gets_high_relevance() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = RelevanceRanker().rank(
        question,
        [
            _market(
                "1",
                "Will the Fed cut interest rates by September?",
                "This market resolves Yes if the Federal Reserve lowers the target rate by September.",
            )
        ],
    )[0]

    assert ranked.relevance_score >= 0.80
    assert ranked.evidence_type == "Direct"


def test_same_event_slightly_different_timeframe_gets_near_direct() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = RelevanceRanker().rank(
        question,
        [
            _market(
                "2",
                "Will the Fed cut interest rates by November?",
                "This market resolves Yes if the Federal Reserve lowers rates by November.",
            )
        ],
    )[0]

    assert ranked.evidence_type in {"Direct", "Near-direct"}
    assert ranked.relevance_score >= 0.65


def test_correlated_market_gets_related() -> None:
    question = QueryInterpreter().interpret("Will gas prices rise this summer?")
    ranked = RelevanceRanker().rank(
        question,
        [
            _market(
                "3",
                "Will oil prices rise this summer?",
                "This market resolves Yes if oil prices rise during the summer.",
            )
        ],
    )[0]

    assert 0.45 <= ranked.relevance_score < 0.80
    assert ranked.evidence_type in {"Related", "Near-direct"}


def test_unrelated_market_gets_irrelevant() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = RelevanceRanker().rank(
        question,
        [_market("4", "Will Air Jordans release this year?", "A sneaker release market.")],
    )[0]

    assert ranked.evidence_type == "Irrelevant"
    assert ranked.relevance_score < 0.25


def test_inactive_market_gets_capped() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = RelevanceRanker().rank(
        question,
        [
            _market(
                "5",
                "Will the Fed cut interest rates by September?",
                "This market resolves Yes if the Federal Reserve lowers the target rate by September.",
                active=False,
                closed=True,
            )
        ],
    )[0]

    assert ranked.relevance_score <= 0.30


def test_no_shared_entity_gets_capped() -> None:
    question = QueryInterpreter().interpret("Will OpenAI IPO before 2027?")
    ranked = RelevanceRanker().rank(
        question,
        [_market("6", "Will SpaceX IPO before 2027?", "This resolves Yes if SpaceX lists publicly.")],
    )[0]

    assert ranked.relevance_score <= 0.25


def test_conditional_market_is_demoted() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    ranked = RelevanceRanker().rank(
        question,
        [
            _market(
                "7",
                "Will the Fed cut rates before Kevin Warsh is confirmed?",
                "This market resolves Yes if the Fed cuts rates before Kevin Warsh is confirmed.",
            )
        ],
    )[0]

    assert ranked.is_conditional or ranked.is_compound
    assert ranked.evidence_type in {"Related", "Weak"}
    assert ranked.relevance_score <= 0.45
    assert "Conditional or compound market does not directly resolve the user question." in ranked.penalty_reasons


def test_normal_timeframe_is_not_conditional() -> None:
    question = QueryInterpreter().interpret("Will OpenAI IPO before 2027?")
    ranked = RelevanceRanker().rank(
        question,
        [_market("8", "Will OpenAI IPO before 2027?", "This resolves Yes if OpenAI IPOs before 2027.")],
    )[0]

    assert ranked.is_conditional is False
    assert ranked.is_compound is False
    assert ranked.evidence_type == "Direct"


def test_entity_conflict_is_capped() -> None:
    question = QueryInterpreter().interpret("Will Trump be impeached?")
    ranked = RelevanceRanker().rank(
        question,
        [_market("9", "Will Biden be impeached?", "This resolves Yes if Biden is impeached.")],
    )[0]

    assert ranked.has_entity_conflict is True
    assert ranked.relevance_score <= 0.40


def test_scope_mismatch_is_not_direct() -> None:
    question = QueryInterpreter().interpret("Will gas prices rise this summer?")
    ranked = RelevanceRanker().rank(
        question,
        [_market("10", "Will oil close above $100 this week?", "This resolves Yes if oil closes above $100 this week.")],
    )[0]

    assert ranked.has_scope_mismatch is True
    assert ranked.evidence_type in {"Related", "Weak", "Irrelevant"}
    assert ranked.evidence_type not in {"Direct", "Near-direct"}


def test_conditional_question_exception_can_be_direct() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates before Kevin Warsh is confirmed?")
    ranked = RelevanceRanker().rank(
        question,
        [
            _market(
                "11",
                "Will the Fed cut rates before Kevin Warsh is confirmed?",
                "This market resolves Yes if the Fed cuts rates before Kevin Warsh is confirmed.",
            )
        ],
    )[0]

    assert ranked.evidence_type == "Direct"
    assert "Conditional or compound market does not directly resolve the user question." not in ranked.penalty_reasons


def _market(
    market_id: str,
    title: str,
    description: str,
    active: bool = True,
    closed: bool = False,
) -> NormalizedMarket:
    return NormalizedMarket(
        market_id=market_id,
        source="polymarket",
        title=title,
        description=description,
        implied_probability=0.5,
        yes_price=0.5,
        no_price=0.5,
        outcome_prices={"Yes": 0.5, "No": 0.5},
        volume_usd=1_000_000,
        liquidity_usd=250_000,
        active=active,
        closed=closed,
    )
