"""
GDELT BigQuery pipeline for fertility/demographic news research.

Steps:
  1. Query GDELT GKG (Global Knowledge Graph) on BigQuery
  2. Filter by themes and/or keywords
  3. Download full text with trafilatura
  4. Save results to CSV/Parquet

Usage:
  cp config.example.json config.json   # edit with your GCP project id
  python gdelt_pipeline.py --config config.json
  python gdelt_pipeline.py --config config.json --no-fulltext  # skip text download
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import trafilatura
from google.cloud import bigquery
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BigQuery query builder
# ---------------------------------------------------------------------------

GDELT_GKG_TABLE = "gdelt-bq.gdeltv2.gkg"

def build_query(config: dict) -> str:
    """Build a BigQuery SQL query from config."""

    date_from = datetime.strptime(config["date_from"], "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    date_to   = datetime.strptime(config["date_to"],   "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

    # Theme filter (GDELT Themes column is semicolon-separated)
    theme_clauses = " OR ".join(
        f"Themes LIKE '%{t}%'" for t in config.get("gdelt_themes", [])
    )

    # Keyword filter on SourceCommonName + DocumentIdentifier (fast proxy)
    # Full-text keyword search requires the GKG extras or post-filtering
    keyword_clauses = " OR ".join(
        f"LOWER(DocumentIdentifier) LIKE '%{k.lower()}%'"
        for k in config.get("keywords", [])
    )

    where_parts = [f"DATE >= {date_from}", f"DATE <= {date_to}"]
    filters = []
    if theme_clauses:
        filters.append(f"({theme_clauses})")
    if keyword_clauses:
        filters.append(f"({keyword_clauses})")
    if filters:
        where_parts.append("(" + " OR ".join(filters) + ")")

    where_sql = "\n  AND ".join(where_parts)
    limit = config.get("max_articles", 5000)

    query = f"""
SELECT
  CAST(DATE AS STRING)                         AS date_str,
  SourceCommonName                             AS source,
  DocumentIdentifier                           AS url,
  Themes                                       AS themes,
  Locations                                    AS locations,
  Persons                                      AS persons,
  Organizations                                AS organizations,
  SPLIT(V2Tone, ',')[SAFE_OFFSET(0)]           AS tone,
  TranslationInfo                              AS translation_info
FROM `{GDELT_GKG_TABLE}`
WHERE {where_sql}
LIMIT {limit}
"""
    return query.strip()


def run_bigquery(config: dict) -> pd.DataFrame:
    """Run the BigQuery query and return a DataFrame."""
    client = bigquery.Client(project=config["project_id"])
    query = build_query(config)
    log.info("Running BigQuery query:\n%s", query)
    job = client.query(query)
    log.info("Query job id: %s — waiting for results...", job.job_id)
    df = job.to_dataframe(progress_bar_type="tqdm")
    log.info("Retrieved %d rows from GDELT.", len(df))
    return df


# ---------------------------------------------------------------------------
# Full text extraction
# ---------------------------------------------------------------------------

def fetch_full_text(url: str, timeout: int = 15) -> str | None:
    """Download and extract main text from a URL using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if downloaded is None:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return text
    except Exception as exc:
        log.debug("Failed to fetch %s: %s", url, exc)
        return None


def add_full_text(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    """Download full text for all URLs in parallel."""
    urls = df["url"].tolist()
    texts = [None] * len(urls)

    log.info("Downloading full text for %d articles (%d workers)...", len(urls), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(fetch_full_text, url): i for i, url in enumerate(urls)}
        with tqdm(total=len(urls), desc="Full text") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                texts[idx] = future.result()
                pbar.update(1)

    df = df.copy()
    df["full_text"] = texts
    n_ok = sum(1 for t in texts if t)
    log.info("Full text retrieved for %d / %d articles.", n_ok, len(urls))
    return df


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, output_dir: str, tag: str = "") -> None:
    """Save DataFrame to CSV and Parquet."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"gdelt_{tag}_{ts}" if tag else f"gdelt_{ts}"
    csv_path     = Path(output_dir) / f"{stem}.csv"
    parquet_path = Path(output_dir) / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    log.info("Saved %d rows → %s", len(df), csv_path)
    log.info("Saved %d rows → %s", len(df), parquet_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GDELT fertility news pipeline")
    parser.add_argument("--config",      default="config.json", help="Path to config JSON")
    parser.add_argument("--no-fulltext", action="store_true",   help="Skip full text download")
    parser.add_argument("--output-dir",  default=None,          help="Override output directory")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    output_dir = args.output_dir or config.get("output_dir", "data")

    # 1. Query GDELT
    df = run_bigquery(config)
    if df.empty:
        log.warning("No results returned. Check your themes/keywords/date range.")
        return

    # 2. Optional: full text
    if not args.no_fulltext and config.get("full_text", True):
        workers = config.get("full_text_workers", 8)
        df = add_full_text(df, workers=workers)
    else:
        df["full_text"] = None

    # 3. Save
    save_results(df, output_dir)


if __name__ == "__main__":
    main()
