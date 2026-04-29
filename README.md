# Prediction Market Intelligence Agent

Prediction Market Intelligence Agent is a Streamlit-based market-implied forecasting dashboard. It lets a user enter a current-world question, searches prediction market data, ranks direct and related markets, normalizes implied probabilities, computes confidence scores, and generates an uncertainty-aware forecast report.

The core design rule is strict: the LLM must not invent probabilities. All probabilities, relevance scores, confidence scores, and aggregations must be computed deterministically in Python. Language models may only interpret user questions, expand search terms, and generate natural-language summaries.

## Disclaimer

This project is for research and informational purposes only. It is not betting, trading, financial, legal, or investment advice.

## Architecture Overview

- `app/streamlit_app.py` provides the Streamlit interface and runs the MVP pipeline.
- `src/pmi_agent/schemas.py` defines the shared Pydantic models.
- `src/pmi_agent/clients/` contains provider adapters. The Polymarket client uses public Gamma market discovery endpoints only.
- `src/pmi_agent/interpretation/` parses questions and expands search terms.
- `src/pmi_agent/search/` handles market search, semantic similarity, and deterministic ranking.
- `src/pmi_agent/engine/` computes probability normalization, aggregation, and confidence scores.
- `src/pmi_agent/reporting/` turns deterministic forecast results into readable reports.
- `src/pmi_agent/storage/` is reserved for local cache and persistence utilities.

## MVP Roadmap

1. Expand Polymarket discovery coverage and pagination.
2. Add deterministic related-market ranking with stronger semantic search.
3. Add market resolution and close-time handling.
4. Add confidence calibration based on relevance, liquidity, market count, and disagreement.
5. Add optional LLM-assisted question interpretation and report wording without allowing model-generated probabilities.
6. Add persistent local caching and tests around every deterministic scoring function.

## Local Development

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Manual Polymarket smoke test:

```bash
python scripts/smoke_polymarket.py
```

Run tests:

```bash
python -m pytest
```
