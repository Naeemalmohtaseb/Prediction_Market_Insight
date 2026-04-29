"""Deterministic market relevance ranking."""

from pmi_agent.schemas import InterpretedQuestion, NormalizedMarket, RankedMarket
from pmi_agent.search.semantic_similarity import jaccard_similarity, token_set


class RelevanceRanker:
    """Rank normalized markets using deterministic relevance signals."""

    def rank(
        self,
        question: InterpretedQuestion,
        markets: list[NormalizedMarket],
        min_score: float = 0.0,
    ) -> list[RankedMarket]:
        """Return markets sorted by confidence-weighted relevance."""

        ranked = [self._rank_one(question, market) for market in markets]
        filtered = [market for market in ranked if market.relevance_score >= min_score]
        return sorted(
            filtered,
            key=lambda item: (
                item.confidence_score,
                item.relevance_score,
                item.liquidity_score,
            ),
            reverse=True,
        )

    def _rank_one(self, question: InterpretedQuestion, market: NormalizedMarket) -> RankedMarket:
        title_score = jaccard_similarity(question.core_event, market.title)
        description_score = jaccard_similarity(question.core_event, market.description or "")
        relevance_score = _clamp((0.75 * title_score) + (0.25 * description_score))
        directness_score = _directness(question.core_event, market.title)
        liquidity_score = _liquidity_score(market.liquidity_usd, market.volume_usd)
        confidence_score = _clamp(
            (0.55 * relevance_score) + (0.25 * directness_score) + (0.20 * liquidity_score)
        )

        reasons = [
            f"title_similarity={title_score:.2f}",
            f"description_similarity={description_score:.2f}",
            f"directness={directness_score:.2f}",
            f"liquidity={liquidity_score:.2f}",
        ]

        return RankedMarket(
            market=market,
            relevance_score=relevance_score,
            directness_score=directness_score,
            liquidity_score=liquidity_score,
            confidence_score=confidence_score,
            rank_reasons=reasons,
        )


def _directness(question_text: str, title: str) -> float:
    question_tokens = token_set(question_text)
    title_tokens = token_set(title)
    if not question_tokens:
        return 0.0
    return _clamp(len(question_tokens & title_tokens) / len(question_tokens))


def _liquidity_score(liquidity_usd: float | None, volume_usd: float | None) -> float:
    value = max(liquidity_usd or 0.0, volume_usd or 0.0)
    if value <= 0:
        return 0.0
    if value >= 1_000_000:
        return 1.0
    return _clamp(value / 1_000_000)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
