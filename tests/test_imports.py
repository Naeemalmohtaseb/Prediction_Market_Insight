"""Smoke tests for the initial scaffold."""

from pmi_agent.engine.probability_engine import ProbabilityEngine
from pmi_agent.interpretation.query_interpreter import QueryInterpreter


def test_query_interpreter_returns_structured_question() -> None:
    interpreted = QueryInterpreter().interpret("Will Example Corp launch a product in 2026?")

    assert interpreted.normalized_question
    assert interpreted.core_event


def test_probability_engine_normalizes_percentages() -> None:
    assert ProbabilityEngine().normalize_probability(62.5) == 0.625
