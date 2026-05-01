"""Lightweight RSS-based context retrieval."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import logging
from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests

from pmi_agent.config import get_config
from pmi_agent.schemas import ContextItem, InterpretedQuestion
from pmi_agent.search.semantic_similarity import text_similarity, token_set

try:
    import feedparser
except ImportError:  # pragma: no cover - exercised when optional dependency is absent.
    feedparser = None

logger = logging.getLogger(__name__)

GENERAL_FEEDS = (
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
)


class NewsContextService:
    """Fetch and summarize recent RSS context without changing probabilities."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update(
                {
                    "Accept": "application/rss+xml, application/xml, text/xml, */*",
                    "User-Agent": "prediction-market-intelligence-agent/0.1 context research",
                }
            )
        self.last_warnings: list[str] = []

    def fetch_context(self, question: InterpretedQuestion, max_items: int = 8) -> list[ContextItem]:
        """Fetch, deduplicate, and rank recent context items."""

        self.last_warnings = []
        items: list[ContextItem] = []
        for term in _context_terms(question):
            url = _google_news_rss_url(term)
            items.extend(self._fetch_feed(url, query_term=term))

        for feed_url in GENERAL_FEEDS[:1]:
            items.extend(self._fetch_feed(feed_url, query_term=None))

        ranked = self.rank_context_items(question, items)
        if not ranked:
            self.last_warnings.append("No recent context items were retrieved.")
        return ranked[:max_items]

    def summarize_context(self, question: InterpretedQuestion, items: list[ContextItem]) -> str:
        """Create a deterministic context summary."""

        if not items:
            return "No recent context items were retrieved. Forecast relies only on prediction-market evidence."

        themes = _top_themes(question, items)
        theme_text = ", ".join(themes[:3]) if themes else question.target_event
        return (
            f"Recent context surfaced items related to {theme_text}. "
            "These items provide background for the forecast but do not modify the market-implied probability."
        )

    def rank_context_items(
        self,
        question: InterpretedQuestion,
        items: list[ContextItem],
    ) -> list[ContextItem]:
        """Rank context items with deterministic lexical and recency signals."""

        deduped = _dedupe_items(items)
        ranked: list[ContextItem] = []
        question_text = _question_text(question)
        question_entities = token_set(" ".join([*question.entities, *question.related_concepts, question.target_event]))
        for item in deduped:
            item_text = f"{item.title} {item.summary or ''}"
            semantic = text_similarity(question_text, item_text)
            item_tokens = token_set(item_text)
            entity_overlap = len(question_entities & item_tokens) / len(question_entities) if question_entities else 0.0
            score = (
                0.45 * semantic
                + 0.30 * entity_overlap
                + 0.15 * _recency_score(item.published_at)
                + 0.10 * _source_quality_score(item)
            )
            ranked.append(item.model_copy(update={"relevance_score": _clamp(score)}))

        ranked.sort(key=lambda item: item.relevance_score, reverse=True)
        return ranked

    def _fetch_feed(self, url: str, query_term: str | None) -> list[ContextItem]:
        try:
            response = self.session.get(url, timeout=get_config().request_timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            warning = f"Context RSS fetch failed: {exc}"
            logger.warning("%s url=%s", warning, url)
            self.last_warnings.append(warning)
            return []

        try:
            entries = _parse_feed(response.text)
        except Exception as exc:
            warning = f"Context RSS parsing failed: {exc}"
            logger.warning("%s url=%s", warning, url)
            self.last_warnings.append(warning)
            return []

        return [
            ContextItem(
                title=entry.get("title") or "Untitled context item",
                source=entry.get("source"),
                url=entry.get("url"),
                published_at=entry.get("published_at"),
                summary=entry.get("summary"),
                relevance_score=0.0,
                query_term=query_term,
                raw=entry.get("raw"),
            )
            for entry in entries
            if entry.get("title")
        ]


def _context_terms(question: InterpretedQuestion) -> list[str]:
    terms: list[str] = []
    if question.category == "macroeconomics" and any("fed" in text.lower() for text in [question.target_event, *question.entities]):
        terms.extend(["Fed rate cut September", "FOMC interest rates", "Federal Reserve rate cut"])
    elif "openai" in question.normalized_question.lower():
        terms.extend(["OpenAI IPO", "OpenAI public offering", "OpenAI valuation"])
    elif "gas" in question.normalized_question.lower():
        terms.extend(["gas prices summer", "gasoline prices", "oil prices OPEC"])

    terms.extend(
        [
            question.normalized_question,
            question.target_event,
            *question.search_terms[:3],
        ]
    )

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = " ".join(str(term).split())
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        deduped.append(clean)
        if len(deduped) >= 5:
            break
    return deduped


def _google_news_rss_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"


def _parse_feed(text: str) -> list[dict[str, Any]]:
    if feedparser is not None:
        parsed = feedparser.parse(text)
        entries = []
        for entry in parsed.entries:
            source = None
            if isinstance(entry.get("source"), dict):
                source = entry.get("source", {}).get("title")
            entries.append(
                {
                    "title": entry.get("title"),
                    "source": source or entry.get("author") or parsed.feed.get("title"),
                    "url": entry.get("link"),
                    "published_at": entry.get("published") or entry.get("updated"),
                    "summary": _strip_markup(entry.get("summary") or entry.get("description")),
                    "raw": dict(entry),
                }
            )
        return entries
    return _parse_feed_xml(text)


def _parse_feed_xml(text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(text)
    channel_title = _find_text(root, "./channel/title")
    entries: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        source = _find_text(item, "source") or channel_title
        entries.append(
            {
                "title": _find_text(item, "title"),
                "source": source,
                "url": _find_text(item, "link"),
                "published_at": _find_text(item, "pubDate"),
                "summary": _strip_markup(_find_text(item, "description")),
                "raw": {},
            }
        )
    return entries


def _find_text(element: ET.Element, path: str) -> str | None:
    found = element.find(path)
    if found is None or found.text is None:
        return None
    text = found.text.strip()
    return text or None


def _dedupe_items(items: list[ContextItem]) -> list[ContextItem]:
    deduped: list[ContextItem] = []
    seen: set[str] = set()
    for item in items:
        key = (item.url or item.title).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _question_text(question: InterpretedQuestion) -> str:
    return " ".join(
        [
            question.normalized_question,
            question.target_event,
            question.expected_outcome,
            " ".join(question.entities),
            " ".join(question.related_concepts),
        ]
    )


def _recency_score(published_at: str | None) -> float:
    published = _parse_datetime(published_at)
    if published is None:
        return 0.5
    age_days = max(0, (datetime.now(UTC) - published).days)
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.8
    if age_days <= 90:
        return 0.55
    return 0.25


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _source_quality_score(item: ContextItem) -> float:
    score = 0.4
    if item.source:
        score += 0.25
    if item.url:
        score += 0.15
    if len(item.title) >= 20:
        score += 0.20
    return _clamp(score)


def _top_themes(question: InterpretedQuestion, items: list[ContextItem]) -> list[str]:
    protected = [
        entity
        for entity in [*question.entities, *question.related_concepts, question.target_event]
        if entity and len(entity) > 2
    ]
    counts: Counter[str] = Counter()
    for phrase in protected:
        phrase_tokens = token_set(phrase)
        for item in items[:5]:
            if phrase_tokens & token_set(f"{item.title} {item.summary or ''}"):
                counts[phrase] += 1
    if counts:
        return [theme for theme, _count in counts.most_common(3)]

    tokens = Counter()
    for item in items[:5]:
        tokens.update(token_set(item.title))
    return [token for token, _count in tokens.most_common(3)]


def _strip_markup(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(value.replace("<", " <").split())
    while "<" in text and ">" in text:
        start = text.find("<")
        end = text.find(">", start)
        if end == -1:
            break
        text = text[:start] + text[end + 1 :]
    return " ".join(text.split()) or None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
