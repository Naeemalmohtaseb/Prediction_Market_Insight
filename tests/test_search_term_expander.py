"""Tests for deterministic search term expansion."""

from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.interpretation.search_term_expander import SearchTermExpander


def test_expander_deduplicates_terms() -> None:
    question = QueryInterpreter().interpret("Will the Fed cut rates by September?")
    terms = SearchTermExpander().expand(question)

    assert len(terms) == len({term.casefold() for term in terms})
    assert len(terms) <= 15


def test_expander_includes_entities_and_related_concepts() -> None:
    question = QueryInterpreter().interpret("Will gas prices rise this summer?")
    terms = SearchTermExpander().expand(question)

    joined = " | ".join(terms)
    assert "gas prices" in joined
    assert "oil prices" in joined
    assert "OPEC" in terms
