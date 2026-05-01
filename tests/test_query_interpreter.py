"""Tests for deterministic forecast question interpretation."""

from pmi_agent.interpretation.query_interpreter import QueryInterpreter


def test_fed_rate_cut_question() -> None:
    result = QueryInterpreter().interpret("Will the Fed cut rates by September?")

    assert result.category == "macroeconomics"
    assert result.target_event == "Federal Reserve cuts interest rates"
    assert "Fed" in result.entities
    assert result.timeframe == "by September"
    assert "rate cut" in result.related_concepts


def test_gas_prices_question() -> None:
    result = QueryInterpreter().interpret("Will gas prices rise this summer?")

    assert result.category == "macroeconomics"
    assert result.expected_outcome == "gas prices rise"
    assert result.timeframe == "this summer"
    assert "OPEC" in result.related_concepts


def test_openai_ipo_question() -> None:
    result = QueryInterpreter().interpret("Will OpenAI IPO before 2027?")

    assert result.category == "finance"
    assert result.target_event == "OpenAI IPO"
    assert "OpenAI" in result.entities
    assert result.timeframe == "before 2027"


def test_air_jordans_question() -> None:
    result = QueryInterpreter().interpret("Will new Air Jordans release this year?")

    assert result.category == "consumer_products"
    assert result.target_event == "new Air Jordans release"
    assert "Air Jordans" in result.entities


def test_us_iran_conflict_question() -> None:
    result = QueryInterpreter().interpret("Will the U.S. enter a conflict with Iran?")

    assert result.category == "geopolitical_risk"
    assert result.target_event == "U.S. enters military conflict with Iran"
    assert "Iran" in result.entities
    assert result.geography == "Iran"


def test_trump_impeachment_question() -> None:
    result = QueryInterpreter().interpret("Will Trump be impeached?")

    assert result.category == "politics"
    assert result.target_event == "Trump is impeached"
    assert "Trump" in result.entities
    assert result.expected_outcome == "impeachment occurs"
