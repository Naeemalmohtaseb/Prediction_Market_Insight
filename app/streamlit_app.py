"""Streamlit entrypoint for the Prediction Market Intelligence Agent."""

from pathlib import Path
import sys
from typing import Any

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.clients.kalshi_client import KalshiClient
from pmi_agent.clients.polymarket_client import PolymarketClient
from pmi_agent.context.news_context import NewsContextService
from pmi_agent.engine.aggregation_engine import AggregationEngine
from pmi_agent.interpretation.query_interpreter import QueryInterpreter
from pmi_agent.reporting.report_generator import ReportGenerator
from pmi_agent.search.market_catalog import MarketCatalogService
from pmi_agent.search.market_search import MarketSearchService
from pmi_agent.search.relevance_ranker import RelevanceRanker
from pmi_agent.schemas import ForecastResult, InterpretedQuestion, MarketProbability, RankedMarket
from pmi_agent.storage.db import StorageError, StorageManager

EXAMPLES = [
    "Will the Fed cut rates by September?",
    "Will OpenAI IPO before 2027?",
    "Will gas prices rise this summer?",
    "Will Trump be impeached?",
]


def main() -> None:
    """Render the Streamlit dashboard."""

    st.set_page_config(page_title="Prediction Market Intelligence Agent", layout="wide")
    storage = StorageManager()

    _show_intro()
    controls = _show_question_card()
    loaded_analysis = _show_sidebar(storage, controls["provider_choice"])

    if controls["run_forecast"]:
        _run_forecast(storage, controls)

    _show_saved_analysis_history(loaded_analysis)


def _show_intro() -> None:
    with st.container(border=True):
        st.title("Prediction Market Intelligence Agent")
        st.subheader(
            "A market-implied forecasting dashboard that analyzes prediction markets and recent context "
            "to help interpret current-world questions."
        )
        st.write(
            "Ask a current-world question. The app searches Polymarket and Kalshi, finds relevant markets, "
            "estimates the market-implied probability, checks evidence quality, and produces a short explanation. "
            "Context/news is shown separately and does not change the market probability."
        )
        with st.expander("How to use this"):
            st.markdown(
                "- Enter a question with a clear event and timeframe.\n"
                "- Choose providers or leave as Both.\n"
                "- Use Hybrid search for normal use.\n"
                "- Click Run Forecast.\n"
                "- Read the summary first, then expand details if needed."
            )
        st.info("Research and market intelligence only. Not financial, investment, trading, or betting advice.")


def _show_question_card() -> dict[str, Any]:
    if "question_text" not in st.session_state:
        st.session_state.question_text = EXAMPLES[0]

    with st.container(border=True):
        st.header("Ask a question")
        st.caption("Clear event + timeframe works best. Example: “Will the Fed cut rates by September?”")

        example_cols = st.columns(len(EXAMPLES))
        for col, example in zip(example_cols, EXAMPLES):
            if col.button(example, use_container_width=True):
                st.session_state.question_text = example

        question_text = st.text_input(
            "What do you want to forecast?",
            key="question_text",
            placeholder="Will the Fed cut rates by September?",
        )

        with st.expander("Advanced options"):
            provider_choice = st.selectbox(
                "Providers",
                ["Both", "Polymarket", "Kalshi"],
                help="Choose which prediction-market platforms to search. Both is recommended unless you want to compare one source.",
            )
            search_label = st.selectbox(
                "Search mode",
                ["Hybrid", "Catalog only", "Live only"],
                help=(
                    "Hybrid searches the local market catalog first, then supplements with live API results. "
                    "Catalog only is faster but may be stale. Live only fetches directly from provider APIs."
                ),
            )
            include_context = st.checkbox(
                "Include recent context layer",
                value=True,
                help="Adds recent RSS/news context for background only. This does not change the market-implied probability.",
            )
            max_context_items = st.slider(
                "Max context items",
                min_value=3,
                max_value=10,
                value=6,
                help="Controls how many recent context items appear in the background section.",
            )
            save_analysis = st.checkbox(
                "Save this analysis locally",
                value=True,
                help="Saves the forecast and market evidence locally in SQLite so you can review it later.",
            )
            limit_per_term = st.slider(
                "Search breadth",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                help="Controls how many markets are requested for each search term before ranking.",
            )

        run_forecast = st.button("Run Forecast", type="primary", disabled=not question_text.strip())

    return {
        "question_text": question_text,
        "provider_choice": provider_choice,
        "search_mode": _search_mode_value(search_label),
        "include_context": include_context,
        "max_context_items": max_context_items,
        "save_analysis": save_analysis,
        "limit_per_term": limit_per_term,
        "run_forecast": run_forecast,
    }


def _show_sidebar(storage: StorageManager, provider_choice: str) -> dict[str, Any] | None:
    st.sidebar.header("Operations")
    st.sidebar.subheader("Market Catalog")
    st.sidebar.caption(
        "The catalog stores recently fetched active markets locally so searches are faster and broader. "
        "Refresh before serious analysis."
    )
    try:
        stats = storage.get_catalog_stats()
        st.sidebar.metric("Cached markets", stats.get("total_markets", 0))
        st.sidebar.caption(f"By provider: {stats.get('by_provider', {})}")
        latest = stats.get("latest_last_seen_at") or stats.get("latest_fetched_at")
        if latest:
            st.sidebar.caption(f"Latest refresh: {str(latest)[:19]}")
    except StorageError as exc:
        st.sidebar.warning(f"Could not load catalog stats: {exc}")

    if st.sidebar.button(
        "Refresh catalog",
        help="Fetches active markets from the selected provider set and stores them locally.",
    ):
        with st.spinner("Refreshing active market catalog..."):
            try:
                counts = MarketCatalogService(storage).refresh_all(
                    _selected_clients(provider_choice),
                    limit_per_provider=500,
                )
                st.sidebar.success(f"Catalog refreshed: {counts}")
            except StorageError as exc:
                st.sidebar.warning(f"Catalog refresh failed: {exc}")

    loaded_analysis = _show_recent_analyses_sidebar(storage)

    with st.sidebar.expander("Maintenance / advanced"):
        clear_provider = st.selectbox(
            "Clear catalog provider",
            ["All", "polymarket", "kalshi"],
            help="Removes cached market records. Saved analyses are not deleted.",
        )
        if st.button("Clear selected catalog entries"):
            try:
                provider = None if clear_provider == "All" else clear_provider
                deleted = storage.clear_market_catalog(provider=provider)
                st.warning(f"Cleared {deleted} cached market(s).")
            except StorageError as exc:
                st.warning(f"Could not clear catalog: {exc}")

    return loaded_analysis


def _run_forecast(storage: StorageManager, controls: dict[str, Any]) -> None:
    interpreted = QueryInterpreter().interpret(controls["question_text"])
    clients = _selected_clients(controls["provider_choice"])
    search_service = MarketSearchService(catalog_service=MarketCatalogService(storage))
    markets = search_service.search(
        interpreted,
        providers=clients,
        limit_per_term=controls["limit_per_term"],
        search_mode=controls["search_mode"],
    )

    for client in clients:
        if getattr(client, "last_error", None):
            st.warning(client.last_error)

    if not markets:
        st.info("No normalized markets were returned for this question.")
        return

    ranked_markets = RelevanceRanker().rank(interpreted, markets)
    result = AggregationEngine().aggregate(interpreted, ranked_markets)
    if controls["include_context"]:
        _attach_context(result, interpreted, controls["max_context_items"])
    report = ReportGenerator().generate(interpreted, result)

    if controls["save_analysis"]:
        try:
            analysis_id = storage.save_analysis(interpreted, result, report.markdown_report)
            st.success(f"Saved local analysis #{analysis_id}.")
        except StorageError as exc:
            st.warning(f"Could not save analysis locally: {exc}")

    st.caption(
        f"Search mode: {controls['search_mode']}. "
        f"Catalog results: {search_service.last_search_sources.get('catalog', 0)}; "
        f"live results: {search_service.last_search_sources.get('live', 0)}."
    )

    _show_forecast_snapshot(interpreted, result)
    _show_short_analysis(interpreted, result)
    _show_simple_evidence_summary(result)
    _show_top_market_evidence(result.market_probabilities)
    if controls["include_context"]:
        _show_context_layer(result)
    _show_detailed_sections(interpreted, ranked_markets, result, report.markdown_report)


def _show_forecast_snapshot(question: InterpretedQuestion, result: ForecastResult) -> None:
    with st.container(border=True):
        st.header("Forecast Snapshot")
        st.write(f"**Question:** {question.original_question}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market-implied probability", _fmt_probability(result.estimated_probability))
            st.caption(
                "Calculated from relevant prediction-market prices after filtering and weighting market evidence. "
                "This is not an AI guess."
            )
        with col2:
            st.metric("Confidence", result.confidence_label)
            st.caption(_confidence_explanation(result))
        with col3:
            st.metric("Evidence basis", _evidence_basis(result))
            with st.expander("Show numeric confidence score"):
                st.write(f"{result.confidence_score:.1f}/100")

        st.write(f"**Main reason:** {_main_reason_plain(result)}")
        if result.estimated_probability is None:
            st.warning("No usable market-implied probability was found.")


def _show_short_analysis(question: InterpretedQuestion, result: ForecastResult) -> None:
    with st.container(border=True):
        st.header("Short Analysis")
        st.write(_short_analysis(question, result))


def _show_simple_evidence_summary(result: ForecastResult) -> None:
    with st.container(border=True):
        st.header("What this is based on")
        providers = sorted({item.market.source.title() for item in result.market_probabilities}) or ["-"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Direct markets found", result.direct_markets_count, help="A market that closely answers the question.")
        col2.metric("Related markets found", result.related_markets_count, help="A market that does not answer the question directly but may provide useful context.")
        col3.metric("Providers used", ", ".join(providers), help="Prediction-market platforms searched for evidence.")
        col4.metric("Market activity", _market_activity_label(result), help="Plain-English read on reported volume and liquidity.")
        st.info(_provider_disagreement_text(result))


def _show_top_market_evidence(market_probabilities: list[MarketProbability]) -> None:
    st.header("Top Market Evidence")
    st.caption("Volume is reported market activity. Liquidity is reported market depth; thin liquidity can make prices noisy.")
    rows = [_simple_market_row(item) for item in _top_visible_market_probabilities(market_probabilities)]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No usable market evidence to show.")


def _show_context_layer(result: ForecastResult) -> None:
    with st.container(border=True):
        st.header("Recent Context")
        st.caption("Context is shown for background only. It does not change the market-implied probability.")
        st.write(
            result.context_summary
            or "No recent context items were retrieved. Forecast relies only on prediction-market evidence."
        )
        if result.context_warnings:
            st.warning("; ".join(result.context_warnings[:5]))
        if result.context_items:
            st.dataframe(
                [_context_row(item) for item in result.context_items],
                use_container_width=True,
                hide_index=True,
            )


def _show_detailed_sections(
    question: InterpretedQuestion,
    ranked_markets: list[RankedMarket],
    result: ForecastResult,
    markdown_report: str,
) -> None:
    with st.expander("Show full forecast report"):
        st.markdown(markdown_report)

    with st.expander("Show provider comparison details"):
        if result.provider_summary:
            st.dataframe(
                [_provider_summary_row(provider, data) for provider, data in sorted(result.provider_summary.items())],
                use_container_width=True,
                hide_index=True,
            )
            if result.provider_disagreement_score is not None:
                st.write(f"Provider disagreement score: {result.provider_disagreement_score:.3f}")
            if result.provider_notes:
                st.info("; ".join(result.provider_notes))
        else:
            st.write("No provider comparison available.")

    evidence = [item for item in ranked_markets if item.evidence_type != "Irrelevant"]
    irrelevant = [item for item in ranked_markets if item.evidence_type == "Irrelevant"]
    with st.expander("Show technical market scoring"):
        st.dataframe(
            [_market_probability_row(item) for item in result.market_probabilities],
            use_container_width=True,
            hide_index=True,
        )
        st.divider()
        st.dataframe([_ranked_row(item) for item in evidence], use_container_width=True, hide_index=True)

    with st.expander("Show interpreted question details"):
        st.json(question.model_dump(mode="json"))

    with st.expander("Show raw/debug data"):
        st.write(f"Irrelevant markets: {len(irrelevant)}")
        if irrelevant:
            st.dataframe([_ranked_row(item) for item in irrelevant], use_container_width=True, hide_index=True)
        st.json([item.market.model_dump(mode="json") for item in ranked_markets])


def _show_recent_analyses_sidebar(storage: StorageManager) -> dict[str, Any] | None:
    st.sidebar.subheader("Saved Analyses")
    st.sidebar.caption("These are historical snapshots. Loading one does not refresh market prices.")
    try:
        recent = storage.list_recent_analyses(limit=10)
    except StorageError as exc:
        st.sidebar.warning(f"Could not load local history: {exc}")
        return None

    if not recent:
        st.sidebar.caption("No saved analyses yet.")
        return None

    labels = {
        item["id"]: (
            f"#{item['id']} | {item['created_at'][:19]} | "
            f"{_fmt_probability(item['estimated_probability'])} | "
            f"{item.get('confidence_label') or '-'} | "
            f"{item['user_question'][:50]}"
        )
        for item in recent
    }
    selected_id = st.sidebar.selectbox(
        "Load previous analysis",
        options=[None, *labels.keys()],
        format_func=lambda value: "Select analysis" if value is None else labels[value],
        help="Loads a saved historical snapshot. It does not refresh market data.",
    )
    if selected_id is None:
        return None

    try:
        analysis = storage.get_analysis(int(selected_id))
    except StorageError as exc:
        st.sidebar.warning(f"Could not load analysis #{selected_id}: {exc}")
        return None
    if analysis is None:
        st.sidebar.warning(f"Analysis #{selected_id} was not found.")
    return analysis


def _show_saved_analysis_history(analysis: dict[str, Any] | None) -> None:
    with st.expander("Show saved analysis history"):
        if analysis is None:
            st.write("Select a saved analysis in the sidebar to view it here.")
            return
        st.subheader(f"Historical Snapshot #{analysis['id']}")
        st.info("Loaded analyses are saved local snapshots and are not live refreshed.")
        st.markdown(analysis["markdown_report"])
        snapshots = analysis.get("market_snapshots", [])
        if snapshots:
            st.dataframe([_snapshot_row(snapshot) for snapshot in snapshots], use_container_width=True, hide_index=True)


def _attach_context(result: ForecastResult, question: InterpretedQuestion, max_items: int) -> None:
    service = NewsContextService()
    try:
        items = service.fetch_context(question, max_items=max_items)
        result.context_items = items
        result.context_summary = service.summarize_context(question, items)
        result.context_warnings = service.last_warnings
    except Exception as exc:
        result.context_items = []
        result.context_summary = "No recent context items were retrieved. Forecast relies only on prediction-market evidence."
        result.context_warnings = [f"Context retrieval failed: {exc}"]


def _selected_clients(provider_choice: str) -> list[Any]:
    if provider_choice == "Polymarket":
        return [PolymarketClient()]
    if provider_choice == "Kalshi":
        return [KalshiClient()]
    return [PolymarketClient(), KalshiClient()]


def _search_mode_value(label: str) -> str:
    return {
        "Hybrid": "hybrid",
        "Catalog only": "catalog",
        "Live only": "live",
    }[label]


def _confidence_explanation(result: ForecastResult) -> str:
    if result.confidence_label == "High":
        return "High: multiple relevant markets or strong evidence quality support the estimate."
    if result.confidence_label == "Medium":
        return "Medium: some relevant markets exist, but evidence is mixed or liquidity is limited."
    return "Low: no direct market was found or the available markets are thin/noisy."


def _evidence_basis(result: ForecastResult) -> str:
    if result.estimated_probability is None:
        return "No usable evidence"
    if result.direct_markets_count > 0:
        return "Mostly direct markets"
    if result.related_markets_count > 0:
        return "Related markets"
    return "Limited evidence"


def _main_reason_plain(result: ForecastResult) -> str:
    if result.estimated_probability is None:
        return "No usable market-implied probability was found."
    if any("No direct market found" in warning for warning in result.key_warnings):
        return "No clean direct market was found, so the estimate relies on related market evidence."
    if any("Providers show substantial disagreement" in note for note in result.provider_notes):
        return "Relevant markets were found, but provider disagreement reduces confidence."
    if any("low liquidity" in warning.lower() or "thin liquidity" in warning.lower() for warning in result.key_warnings):
        return "Relevant markets were found, but some evidence has limited liquidity."
    if result.direct_markets_count > 0:
        return "Markets directly related to the question were found and carried the main aggregation weight."
    return "The forecast is based on the available market evidence after filtering for relevance and quality."


def _short_analysis(question: InterpretedQuestion, result: ForecastResult) -> str:
    probability = _fmt_probability(result.estimated_probability)
    evidence = _evidence_basis(result).lower()
    activity = _market_activity_label(result).lower()
    provider_text = _provider_disagreement_text(result)
    context_sentence = (
        "Recent context is shown below for background only and does not change the probability."
        if result.context_items or result.context_summary
        else "No recent context was attached, so this view relies on prediction-market evidence."
    )
    return (
        f"The market evidence suggests a market-implied probability of {probability} for: {question.target_event}. "
        f"The estimate is based on {evidence}, and reported market activity looks {activity}. "
        f"{provider_text} "
        f"{context_sentence}"
    )


def _market_activity_label(result: ForecastResult) -> str:
    usable = [item for item in result.market_probabilities if item.implied_probability is not None]
    if not usable:
        return "Thin"
    total_volume = sum(item.market.volume_usd or 0 for item in usable)
    total_liquidity = sum(item.market.liquidity_usd or 0 for item in usable)
    if total_volume >= 1_000_000 and total_liquidity >= 50_000:
        return "Strong"
    if total_volume >= 100_000 or total_liquidity >= 10_000:
        return "Moderate"
    return "Thin"


def _provider_disagreement_text(result: ForecastResult) -> str:
    notes = result.provider_notes
    if any("broadly agree" in note.lower() for note in notes):
        return "Providers broadly agree."
    if any("moderate disagreement" in note.lower() for note in notes):
        return "Providers differ somewhat."
    if any("substantial disagreement" in note.lower() for note in notes):
        return "Providers disagree substantially."
    if any("only one provider" in note.lower() for note in notes):
        return "Only one provider had usable evidence."
    return "Provider comparison is limited."


def _top_visible_market_probabilities(items: list[MarketProbability]) -> list[MarketProbability]:
    usable = [item for item in items if item.implied_probability is not None]
    usable.sort(key=lambda item: (item.evidence_type in {"Direct", "Near-direct"}, item.relevance_score), reverse=True)
    return usable[:10]


def _simple_market_row(item: MarketProbability) -> dict[str, Any]:
    market = item.market
    return {
        "Provider": market.source.title(),
        "Evidence Type": item.evidence_type,
        "Market": market.title,
        "Implied Probability": _fmt_probability(item.implied_probability),
        "Volume": _fmt_money(market.volume_usd),
        "Liquidity": _fmt_money(market.liquidity_usd),
        "Why it matters": _why_it_matters(item),
    }


def _why_it_matters(item: MarketProbability) -> str:
    warnings = " ".join(item.warnings).lower()
    if "conditional" in warnings or "compound" in warnings:
        return "Downweighted because it includes an extra condition."
    if "scope mismatch" in warnings or "timeframe conflict" in warnings or "entity conflict" in warnings:
        return "Useful context, but not the same exact question."
    if item.evidence_type in {"Direct", "Near-direct"}:
        if "low liquidity" in warnings or "thin liquidity" in warnings:
            return "Closely matches the question, but liquidity is limited."
        return "Directly matches the question."
    if item.evidence_type in {"Related", "Weak"}:
        return "Related signal, not a direct answer."
    return "Included for background evidence."


def _market_probability_row(item: MarketProbability) -> dict[str, Any]:
    market = item.market
    return {
        "provider": market.source,
        "evidence_type": item.evidence_type,
        "title": market.title,
        "target_outcome": item.target_outcome,
        "implied_probability": item.implied_probability,
        "probability_source": item.probability_source,
        "relevance_score": round(item.relevance_score, 3),
        "market_weight": round(item.market_weight, 4),
        "volume": market.volume_usd,
        "liquidity": market.liquidity_usd,
        "liquidity_score": item.liquidity_score,
        "volume_score": item.volume_score,
        "spread_score": item.spread_score,
        "recency_score": item.recency_score,
        "resolution_score": item.resolution_score,
        "provider_quality_note": item.provider_quality_note,
        "warnings": ", ".join(item.warnings),
        "penalty_reasons": ", ".join(_penalty_warnings(item.warnings)),
        "final_market_weight": round(item.market_weight, 4),
    }


def _context_row(item: Any) -> dict[str, Any]:
    return {
        "Title": item.title,
        "Source": item.source,
        "Date": item.published_at,
        "Relevance": round(item.relevance_score, 3),
    }


def _provider_summary_row(provider: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": provider,
        "markets_count": data.get("markets_count"),
        "usable_markets_count": data.get("usable_markets_count"),
        "direct_or_near_direct_count": data.get("direct_or_near_direct_count"),
        "weighted_probability": data.get("weighted_probability"),
        "average_relevance": data.get("average_relevance"),
        "total_weight": data.get("total_weight"),
        "average_liquidity_score": data.get("average_liquidity_score"),
        "average_volume_score": data.get("average_volume_score"),
        "average_spread_score": data.get("average_spread_score"),
        "warnings_count": data.get("warnings_count"),
    }


def _snapshot_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": snapshot.get("provider"),
        "provider_market_id": snapshot.get("provider_market_id"),
        "title": snapshot.get("title"),
        "evidence_type": snapshot.get("evidence_type"),
        "relevance_score": snapshot.get("relevance_score"),
        "target_outcome": snapshot.get("target_outcome"),
        "implied_probability": snapshot.get("implied_probability"),
        "market_weight": snapshot.get("market_weight"),
        "yes_price": snapshot.get("yes_price"),
        "no_price": snapshot.get("no_price"),
        "volume": snapshot.get("volume"),
        "liquidity": snapshot.get("liquidity"),
        "warnings": ", ".join(snapshot.get("warnings", [])),
        "penalty_reasons": ", ".join(snapshot.get("penalty_reasons", [])),
    }


def _ranked_row(ranked_market: RankedMarket) -> dict[str, Any]:
    market = ranked_market.market
    return {
        "provider": market.source,
        "evidence_type": ranked_market.evidence_type,
        "relevance_score": round(ranked_market.relevance_score, 3),
        "market_title": market.title,
        "yes_price": market.yes_price,
        "no_price": market.no_price,
        "volume": market.volume_usd,
        "liquidity": market.liquidity_usd,
        "close_date": market.close_time.isoformat() if market.close_time else None,
        "semantic_similarity": round(ranked_market.semantic_similarity, 3),
        "entity_overlap": round(ranked_market.entity_overlap, 3),
        "timeframe_alignment": round(ranked_market.timeframe_alignment, 3),
        "outcome_alignment": round(ranked_market.outcome_alignment, 3),
        "is_conditional": ranked_market.is_conditional,
        "is_compound": ranked_market.is_compound,
        "has_scope_mismatch": ranked_market.has_scope_mismatch,
        "has_timeframe_conflict": ranked_market.has_timeframe_conflict,
        "has_entity_conflict": ranked_market.has_entity_conflict,
        "penalty_reasons": ", ".join(ranked_market.penalty_reasons),
        "rationale": ranked_market.rationale,
    }


def _fmt_probability(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.0f}"


def _penalty_warnings(warnings: list[str]) -> list[str]:
    markers = (
        "conditional",
        "compound",
        "scope",
        "timeframe",
        "entity",
        "directly resolve",
        "different than",
        "differs",
    )
    return [warning for warning in warnings if any(marker in warning.lower() for marker in markers)]


if __name__ == "__main__":
    main()
