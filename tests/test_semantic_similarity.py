"""Tests for deterministic text similarity."""

from pmi_agent.search.semantic_similarity import text_similarity


def test_similar_text_scores_higher_than_unrelated_text() -> None:
    similar = text_similarity("Fed cut interest rates", "Federal Reserve rate cut in September")
    unrelated = text_similarity("Fed cut interest rates", "Air Jordans sneaker release")

    assert similar > unrelated
