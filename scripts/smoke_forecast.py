"""Manual smoke test for full forecast pipeline."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.clients.kalshi_client import KalshiClient
from pmi_agent.clients.polymarket_client import PolymarketClient
from pmi_agent.context.news_context import NewsContextService
from pmi_agent.engine.aggregation_engine import AggregationEngine
from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.search.market_search import MarketSearchService
from pmi_agent.search.relevance_ranker import RelevanceRanker


QUESTIONS = [
    "Will the Fed cut rates by September?",
    "Will Trump be impeached?",
    "Will OpenAI IPO before 2027?",
    "Will the Fed cut rates before Kevin Warsh is confirmed?",
]
SEARCH_MODE = "hybrid"
INCLUDE_CONTEXT = True


def main() -> None:
    clients = [PolymarketClient(), KalshiClient()]
    interpreter = QueryInterpreter()
    search = MarketSearchService()
    ranker = RelevanceRanker()
    aggregator = AggregationEngine()
    context_service = NewsContextService()

    for question_text in QUESTIONS:
        print(f"\nQuestion: {question_text}")
        interpreted = interpreter.interpret(question_text)
        print(f"Interpreted event: {interpreted.target_event}")

        markets = search.search(interpreted, providers=clients, limit_per_term=8, search_mode=SEARCH_MODE)
        print(
            f"Search mode: {SEARCH_MODE} "
            f"(catalog={search.last_search_sources.get('catalog', 0)}, "
            f"live={search.last_search_sources.get('live', 0)})"
        )
        ranked = ranker.rank(interpreted, markets)
        result = aggregator.aggregate(interpreted, ranked)
        probability_before_context = result.estimated_probability
        if INCLUDE_CONTEXT:
            context_items = context_service.fetch_context(interpreted, max_items=3)
            result.context_items = context_items
            result.context_summary = context_service.summarize_context(interpreted, context_items)
            result.context_warnings = context_service.last_warnings
        probabilities_by_id = {
            item.market.market_id: item
            for item in result.market_probabilities
        }

        print("Top ranked markets:")
        for item in ranked[:10]:
            probability = probabilities_by_id.get(item.market.market_id)
            print(
                f"  {item.evidence_type:<11} "
                f"{item.market.source:<10} "
                f"rel={item.relevance_score:.3f} "
                f"flags={_flags(item):<18} "
                f"yes={_fmt_prob(item.market.yes_price):>6} "
                f"no={_fmt_prob(item.market.no_price):>6} "
                f"p={_fmt_prob(probability.implied_probability if probability else None):>6} "
                f"w={(probability.market_weight if probability else 0):.4f} "
                f"liq_s={(probability.liquidity_score if probability and probability.liquidity_score is not None else 0):.2f} "
                f"vol_s={(probability.volume_score if probability and probability.volume_score is not None else 0):.2f} "
                f"spr_s={(probability.spread_score if probability and probability.spread_score is not None else 0):.2f} "
                f"{item.market.title[:90]}"
            )
            if item.penalty_reasons:
                print(f"    penalties: {'; '.join(item.penalty_reasons)}")
            if probability and probability.provider_quality_note:
                print(f"    quality: {probability.provider_quality_note}")

        print(f"Estimated probability: {_fmt_prob(result.estimated_probability)}")
        if INCLUDE_CONTEXT:
            unchanged = probability_before_context == result.estimated_probability
            print(f"Context included: yes; probability unchanged by context: {unchanged}")
            print(f"Context summary: {result.context_summary}")
            for context_item in result.context_items[:3]:
                print(
                    f"  context rel={context_item.relevance_score:.3f} "
                    f"{_safe_console_text(context_item.source or '-')} | "
                    f"{_safe_console_text(context_item.title)[:90]}"
                )
        print(f"Confidence: {result.confidence_label} ({result.confidence_score:.1f}/100)")
        print("Provider summary:")
        for provider, data in (result.provider_summary or {}).items():
            print(
                f"  {provider:<10} "
                f"usable={data.get('usable_markets_count', 0):>2} "
                f"p={_fmt_prob(data.get('weighted_probability')):>6} "
                f"weight={(data.get('total_weight') or 0):.4f} "
                f"liq_s={_fmt_optional(data.get('average_liquidity_score'))} "
                f"vol_s={_fmt_optional(data.get('average_volume_score'))} "
                f"spr_s={_fmt_optional(data.get('average_spread_score'))}"
            )
        print(
            "Provider disagreement: "
            + ("-" if result.provider_disagreement_score is None else f"{result.provider_disagreement_score:.3f}")
        )
        if result.provider_notes:
            print("Provider notes:")
            for note in result.provider_notes:
                print(f"  - {note}")
        if result.key_warnings:
            print("Key warnings:")
            for warning in result.key_warnings[:5]:
                print(f"  - {warning}")
        else:
            print("Key warnings: none")


def _fmt_prob(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _safe_console_text(value: str) -> str:
    return value.encode("ascii", errors="replace").decode("ascii")


def _flags(item) -> str:
    flags = []
    if item.is_conditional:
        flags.append("cond")
    if item.is_compound:
        flags.append("comp")
    if item.has_scope_mismatch:
        flags.append("scope")
    if item.has_timeframe_conflict:
        flags.append("time")
    if item.has_entity_conflict:
        flags.append("entity")
    return ",".join(flags) if flags else "-"


if __name__ == "__main__":
    main()
