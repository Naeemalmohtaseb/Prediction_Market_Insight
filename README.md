# Prediction Market Intelligence Agent

Prediction Market Intelligence Agent is a Streamlit-based market-implied forecasting dashboard. It lets a user enter a current-world question, searches prediction market data, ranks direct and related markets, normalizes implied probabilities, computes confidence scores, and generates an uncertainty-aware forecast report.

The core design rule is strict: the LLM must not invent probabilities. All probabilities, relevance scores, confidence scores, and aggregations must be computed deterministically in Python. Language models may only interpret user questions, expand search terms, and generate natural-language summaries.

## Disclaimer

This project is for research and informational purposes only. It is not betting, trading, financial, legal, or investment advice.

## Architecture Overview

- `app/streamlit_app.py` provides the Streamlit interface and runs the MVP pipeline.
- `src/pmi_agent/schemas.py` defines the shared Pydantic models.
- `src/pmi_agent/clients/` contains provider adapters. Polymarket uses public Gamma market discovery endpoints, and Kalshi uses public read-only market data endpoints.
- `src/pmi_agent/interpretation/` parses questions and expands search terms.
- `src/pmi_agent/search/` handles market search, semantic similarity, evidence classification, and deterministic relevance ranking.
- `src/pmi_agent/engine/` extracts market-implied probabilities, computes deterministic weights, aggregates markets, and scores confidence.
- `src/pmi_agent/reporting/` turns deterministic forecast results into template-based Markdown reports.
- `src/pmi_agent/storage/` manages local SQLite persistence for historical analysis snapshots.

## MVP Roadmap

1. Expand Polymarket and Kalshi discovery coverage and pagination.
2. Improve deterministic related-market ranking with better semantic search and calibration.
3. Add market resolution and close-time handling.
4. Add confidence calibration based on relevance, liquidity, market count, and disagreement.
5. Add optional LLM-assisted question interpretation and report wording without allowing model-generated probabilities.
6. Add persistent local caching and tests around every deterministic scoring function.

## Local Development

```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Local Setup on Windows

From PowerShell:

```powershell
cd "c:\Users\priva\OneDrive\Documents\Projects\Betting Markets\prediction-market-intelligence-agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest
python scripts\smoke_polymarket.py
python scripts\smoke_kalshi.py
python scripts\smoke_forecast.py
streamlit run app\streamlit_app.py
```

Script-based setup:

```powershell
.\scripts\setup_windows.ps1
.\scripts\run_tests.ps1
.\scripts\run_app.ps1
```

If PowerShell blocks local scripts, run this once for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Manual market data smoke tests:

```powershell
python scripts/smoke_polymarket.py
python scripts/smoke_kalshi.py
python scripts/refresh_market_catalog.py
python scripts/search_catalog.py "Fed rate cut"
python scripts/smoke_forecast.py
```

Run tests:

```powershell
python -m pytest
```

## Local Persistence

Saved analyses are stored locally in `data/pmi_agent.sqlite`. Each saved analysis is a historical snapshot: it preserves the interpreted question, deterministic forecast result, Markdown report, and the normalized market evidence used at the time it was run. Loading a saved analysis does not refresh live market data.

Deleting `data/pmi_agent.sqlite` clears local history. The app does not store betting, trading, order, private key, or authenticated account data. Public API responses are stored only as analysis/debugging snapshots.

Initialize the database and inspect recent history:

```powershell
python scripts\init_db.py
python scripts\show_history.py
```

## Market Catalog / Local Search Cache

Active markets can be cached locally in the same SQLite database at `data/pmi_agent.sqlite`. The catalog stores normalized public market records from Polymarket and Kalshi, then lets the app search across recently fetched markets before falling back to live provider search. This improves discovery, reduces repeated API calls, and makes the project behave more like a small data engineering pipeline.

The catalog is not a trading database and does not store orders, private keys, or authenticated account data. Cached markets may become stale, so refresh the catalog before serious analysis. Streamlit includes a sidebar “Market Catalog” section with stats, refresh controls, clear controls, and search mode selection.

Refresh and search the local catalog from PowerShell:

```powershell
python scripts\refresh_market_catalog.py
python scripts\search_catalog.py "Fed rate cut"
python scripts\smoke_context.py
```

Search modes:

- `Hybrid`: search the local catalog first, then supplement with smaller live provider searches.
- `Catalog only`: use cached markets only.
- `Live only`: use provider APIs directly, matching the earlier behavior.

## Data Sources

The app currently supports two read-only providers:

- Polymarket, via public Gamma market discovery endpoints.
- Kalshi, via public unauthenticated market data endpoints at `https://api.elections.kalshi.com/trade-api/v2`.

No trading, order placement, private key, or authenticated account endpoints are used. Provider-specific fields are normalized into the shared `NormalizedMarket` schema before relevance ranking, probability extraction, aggregation, reporting, and storage.

In Streamlit, use the sidebar provider selector to search Polymarket, Kalshi, or both. When both are selected, the same deterministic ranking and aggregation pipeline runs across the combined normalized market set.

Kalshi search currently uses bounded `/markets` pagination plus client-side text similarity filtering. This keeps the integration conservative, but it may miss relevant markets if they are not present in the fetched public pages.

## Provider Quality and Comparison

The aggregation engine computes provider-aware market quality diagnostics without changing provider-reported probabilities. Liquidity, volume, spread, recency, and resolution scores are stored with each extracted market probability. If a Kalshi market reports zero liquidity but has valid bid/ask and volume data, the app applies a conservative fallback quality score and labels that choice in the market warnings.

When multiple providers contribute usable probabilities, the app also reports per-provider weighted probabilities, total weight, average quality scores, and provider disagreement. Provider disagreement can reduce confidence, but it does not override or modify the deterministic aggregate probability.

## Context Layer

The app can retrieve recent RSS/news context for an interpreted question. Context retrieval uses lightweight unauthenticated RSS feeds, including query-based Google News RSS URLs and a small set of general current-event feeds. No paid news API key is required.

Context is separated from the market-implied probability. It is used for background, explanation, and uncertainty framing only; it does not override market prices or directly change the estimated probability. Retrieved items may be incomplete or noisy, especially for broad queries, and missing context never blocks the forecast pipeline.

Manual context smoke test:

```powershell
python scripts\smoke_context.py
```
