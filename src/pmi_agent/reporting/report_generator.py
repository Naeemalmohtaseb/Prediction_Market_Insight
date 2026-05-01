"""Deterministic forecast report generation."""

from __future__ import annotations

from pmi_agent.schemas import ContextItem, ForecastReport, ForecastResult, InterpretedQuestion, MarketProbability

DISALLOWED_ADVICE_PHRASES = ("you should bet", "you should trade", "profit from")


class ReportGenerator:
    """Generate template-based Markdown reports from deterministic results."""

    def generate_markdown(self, question: InterpretedQuestion, result: ForecastResult) -> str:
        """Create a deterministic Markdown report."""

        probability_text = _fmt_probability(result.estimated_probability)
        confidence_text = f"{result.confidence_label} ({result.confidence_score:.1f}/100)"
        main_reason = _main_reason(result)

        sections = [
            "# Forecast Summary",
            f"- User question: {question.original_question}",
            f"- Interpreted event: {question.target_event}",
            f"- Estimated probability: {probability_text}",
            f"- Confidence: {confidence_text}",
            f"- Main reason: {main_reason}",
            "",
            "# Market Evidence",
            _evidence_table(result.market_probabilities),
            "",
            "# Evidence Quality Notes",
            _evidence_quality_notes(result.market_probabilities),
            "",
            "# Provider Comparison",
            _provider_comparison(result),
            "",
            "# Context Layer",
            _context_layer(result),
            "",
            "# Related Market Signals",
            _related_signals(result.market_probabilities),
            "",
            "# Uncertainty Drivers",
            _uncertainty_drivers(result),
            "",
            "# Caveats",
            "- Market-implied probabilities are not guarantees.",
            "- Context items provide background only and are not used as direct probability inputs.",
            "- Thin liquidity can distort prices.",
            "- Related markets may be noisy.",
            "- This is not financial, investment, trading, or betting advice.",
        ]
        report = "\n".join(sections)
        _assert_no_advice_language(report)
        return report

    def generate(self, question: InterpretedQuestion, result: ForecastResult) -> ForecastReport:
        """Create a report model."""

        return ForecastReport(
            question=question,
            result=result,
            markdown_report=self.generate_markdown(question, result),
        )


def _main_reason(result: ForecastResult) -> str:
    if result.estimated_probability is None:
        return "No usable market-implied probability was found."
    if result.direct_market_probability is not None:
        return "Direct or near-direct markets carried the main aggregation weight."
    if result.related_signal_probability is not None:
        return "No direct market was available, so related market signals were used."
    return "Market evidence was insufficient for aggregation."


def _evidence_table(market_probabilities: list[MarketProbability]) -> str:
    rows = [
        "| Evidence type | Market title | Outcome used | Implied probability | Relevance | Weight | Volume | Liquidity | Warnings |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not market_probabilities:
        rows.append("| None | No market evidence | - | - | - | - | - | - | No usable market evidence |")
        return "\n".join(rows)

    for item in market_probabilities[:12]:
        rows.append(
            "| "
            + " | ".join(
                [
                    item.evidence_type,
                    _escape(item.market.title),
                    item.target_outcome or "-",
                    _fmt_probability(item.implied_probability),
                    f"{item.relevance_score:.2f}",
                    f"{item.market_weight:.3f}",
                    _fmt_money(item.market.volume_usd),
                    _fmt_money(item.market.liquidity_usd),
                    ", ".join(item.warnings) if item.warnings else "-",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _provider_comparison(result: ForecastResult) -> str:
    summary = result.provider_summary or {}
    if not summary:
        return "No provider comparison was available."

    rows = [
        "| Provider | Weighted probability | Usable markets | Total weight | Avg liquidity score | Avg volume score | Avg spread score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for provider, data in sorted(summary.items()):
        rows.append(
            "| "
            + " | ".join(
                [
                    provider,
                    _fmt_probability(data.get("weighted_probability")),
                    str(data.get("usable_markets_count", 0)),
                    _fmt_decimal(data.get("total_weight")),
                    _fmt_decimal(data.get("average_liquidity_score")),
                    _fmt_decimal(data.get("average_volume_score")),
                    _fmt_decimal(data.get("average_spread_score")),
                ]
            )
            + " |"
        )

    lines = rows
    if result.provider_disagreement_score is not None:
        lines.append(f"\nProvider disagreement score: {result.provider_disagreement_score:.3f}.")
    if result.provider_notes:
        lines.append("\n" + "\n".join(f"- {note}" for note in result.provider_notes))
    lines.append(
        "\nProvider disagreement does not prove one provider is right. Differences may reflect liquidity, "
        "participant base, contract wording, or market quality."
    )
    return "\n".join(lines)


def _related_signals(market_probabilities: list[MarketProbability]) -> str:
    related = [item for item in market_probabilities if item.evidence_type in {"Related", "Weak"}]
    if not related:
        return "No related or weak market signals were used."
    lines = []
    for item in related[:5]:
        lines.append(
            f"- {item.evidence_type}: {item.market.title} contributed "
            f"{_fmt_probability(item.implied_probability)} with weight {item.market_weight:.3f}."
        )
    return "\n".join(lines)


def _context_layer(result: ForecastResult) -> str:
    lines = [
        result.context_summary
        or "No recent context items were retrieved. Forecast relies only on prediction-market evidence.",
        "",
        "Context items provide background only and are not used as direct probability inputs.",
    ]
    if result.context_warnings:
        lines.append("")
        lines.extend(f"- Warning: {warning}" for warning in result.context_warnings[:5])
    if not result.context_items:
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "| Title | Source | Published | Relevance | URL |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for item in result.context_items[:8]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(item.title),
                    _escape(item.source or "-"),
                    _escape(item.published_at or "-"),
                    f"{item.relevance_score:.2f}",
                    item.url or "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _evidence_quality_notes(market_probabilities: list[MarketProbability]) -> str:
    downweighted = [
        item
        for item in market_probabilities
        if any(
            warning in item.warnings
            for warning in (
                "conditional market",
                "compound market",
                "scope mismatch",
                "timeframe conflict",
                "entity conflict",
                "Conditional or compound market does not directly resolve the user question.",
                "Market scope is narrower or different than the interpreted event.",
                "Market centers on a different entity or sub-event.",
                "Market timeframe differs from the user's timeframe.",
            )
        )
    ]
    if not downweighted:
        return "No conditional, compound, or scope-mismatch penalties were applied to the included markets."

    lines = [
        "Some high-similarity markets were downweighted because they do not cleanly resolve the interpreted event. "
        "These markets can still provide context, but they are not treated as direct evidence when mismatch flags are present."
    ]
    for item in downweighted[:6]:
        penalty_text = ", ".join(item.warnings[:6])
        lines.append(f"- {item.market.title}: {penalty_text}. Final weight {item.market_weight:.3f}.")
    return "\n".join(lines)


def _uncertainty_drivers(result: ForecastResult) -> str:
    warnings = result.key_warnings or ["No major deterministic warnings were generated."]
    lines = [f"- {warning}" for warning in warnings[:10]]
    if result.disagreement_score is not None:
        lines.append(f"- Weighted disagreement score: {result.disagreement_score:.3f}.")
    return "\n".join(lines)


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


def _fmt_decimal(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def _assert_no_advice_language(report: str) -> None:
    lowered = report.lower()
    for phrase in DISALLOWED_ADVICE_PHRASES:
        if phrase in lowered:
            raise ValueError(f"Report contains disallowed advice phrase: {phrase}")
