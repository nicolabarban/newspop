# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**newspop** is an automated news monitoring system for Italian demographic research (fertility, birth rates, demographic decline). It collects articles from GDELT and NewsData.io, extracts full text, and generates weekly AI-powered digests.

## Architecture

### Three-pipeline architecture:

1. **`gdelt_pipeline.py`** — Queries GDELT Global Knowledge Graph via Google BigQuery. Filters by demographic themes (`FERTILITY`, `POPULATION_DECLINE`, `ECON_BIRTHRATE`) and Italian keywords. Extracts full article text with `trafilatura` using multithreaded web scraping. Outputs `data/gdelt_YYYYMMDD_HHMMSS.parquet`.

2. **`newsdata_pipeline.py`** — Fetches Italian articles from NewsData.io REST API. Normalizes to the GDELT schema for downstream compatibility. Outputs `data/newsdata_YYYYMMDD_HHMMSS.parquet`.

3. **`generate_digest.py`** — Merges latest GDELT + NewsData parquet files, deduplicates by URL, sends articles to Claude Sonnet API, saves Markdown digest to `posts/YYYY-MM-DD_digest.md`, optionally emails and publishes to `gh-pages`.

### Shared parquet schema:
```
date_str, source, url, themes, locations, persons, organizations, tone, translation_info, full_text
```

### Automation:
- **Daily** (05:00 UTC): GDELT 3-day fetch → `daily_fetch.yml`
- **Weekly** (Monday 06:00 UTC): GDELT 7-day + NewsData + Claude digest → `weekly_digest.yml`

## Commands

### Setup
```bash
pip install -r requirements.txt
gcloud auth application-default login   # for BigQuery/GDELT
cp config.example.json config.json      # then edit project_id
```

### Run pipelines
```bash
# GDELT: last 3 days, auto date range
python gdelt_pipeline.py --config config.json --auto-dates --days 3

# GDELT: skip full text download (faster, for testing)
python gdelt_pipeline.py --config config.json --no-fulltext

# NewsData.io
python newsdata_pipeline.py --output-dir data

# Generate weekly digest
python generate_digest.py --data-dir data --output-dir posts

# Digest from specific parquet file
python generate_digest.py --parquet data/gdelt_20260219_151632.parquet

# Add --send-email to any pipeline to trigger email notification
```

### Environment variables
```
ANTHROPIC_API_KEY      # required for generate_digest.py
NEWSDATA_API_KEY       # required for newsdata_pipeline.py
MAIL_USERNAME          # Gmail address for email notifications
MAIL_PASSWORD          # Gmail app password (not account password)
```

`config.json` holds Google Cloud `project_id` and pipeline parameters (keywords, themes, date range, language filters). This file is gitignored — `config.actions.json` is the committed version used by GitHub Actions.

## Key Implementation Notes

- `trafilatura` is used for article body extraction; it runs in a `ThreadPoolExecutor` with configurable workers (`full_text_workers` in config, default 8, reduced to 4 in CI).
- Date ranges can be specified in `config.json` or computed at runtime via `--auto-dates --days N`.
- `generate_digest.py` merges GDELT and NewsData by concatenating DataFrames and dropping URL duplicates — only articles with non-empty `full_text` are sent to the LLM.
- Digest posts are committed to both `main` (under `posts/`) and `gh-pages` (Jekyll blog) by the weekly workflow.
- GitHub Actions secrets: `GCP_SA_KEY` (service account JSON), `ANTHROPIC_API_KEY`, `NEWSDATA_API_KEY`, `MAIL_USERNAME`, `MAIL_PASSWORD`.
