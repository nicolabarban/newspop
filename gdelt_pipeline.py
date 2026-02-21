"""
GDELT DOC 2.0 pipeline for fertility/demographic news research.

Uses the free GDELT DOC 2.0 API (no BigQuery, no GCP account needed).

Steps:
  1. Query GDELT DOC 2.0 API by keywords and/or themes
  2. Download full text with trafilatura
  3. Save results to CSV/Parquet
  4. (optional) Send email digest via SMTP

Usage:
  python gdelt_pipeline.py --config config.json
  python gdelt_pipeline.py --config config.json --no-fulltext
  python gdelt_pipeline.py --config config.actions.json --auto-dates --send-email
"""

import argparse
import json
import logging
import os
import smtplib
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# gdeltdoc recurses when paginating large result sets; give it enough headroom
sys.setrecursionlimit(5000)

import pandas as pd
import trafilatura
from gdeltdoc import GdeltDoc, Filters
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GDELT DOC 2.0 query
# ---------------------------------------------------------------------------

def run_gdeltdoc(config: dict) -> pd.DataFrame:
    """Query GDELT DOC 2.0 API and return a normalised DataFrame."""
    gd = GdeltDoc()

    date_from    = config["date_from"]
    date_to      = config["date_to"]
    keywords     = config.get("keywords", [])
    themes       = config.get("gdelt_themes", [])
    languages    = config.get("languages", []) or []
    max_articles = config.get("max_articles", 500)

    # Map language names to FIPS-2 country codes for theme queries
    # (country filter is more reliable than language filter in GDELT DOC 2.0)
    LANGUAGE_TO_COUNTRY = {
        "Italian": "IT", "German": "GM", "French": "FR",
        "Spanish": "SP", "Portuguese": "PO",
    }
    country_list = [LANGUAGE_TO_COUNTRY[l] for l in languages if l in LANGUAGE_TO_COUNTRY]
    country_filter_list = country_list if country_list else [None]

    # Only keep ASCII-safe keywords — GDELT API rejects accented characters
    ascii_keywords = [k for k in keywords if k.isascii()]
    skipped = set(keywords) - set(ascii_keywords)
    if skipped:
        log.info("Skipping non-ASCII keywords (not supported by API): %s", skipped)

    all_dfs = []

    def _search(filters_kwargs: dict) -> pd.DataFrame:
        try:
            f  = Filters(**filters_kwargs)
            df = gd.article_search(f)
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()
            return df
        except RecursionError:
            log.warning("GDELT DOC: recursion limit hit for %s — skipping", filters_kwargs)
            return pd.DataFrame()
        except Exception as exc:
            log.warning("GDELT DOC query failed (%s): %s", filters_kwargs, exc)
            return pd.DataFrame()

    # Keyword queries — GDELT API accepts at most 3 keyphrases per request
    BATCH = 3
    keyword_batches = [ascii_keywords[i:i+BATCH] for i in range(0, len(ascii_keywords), BATCH)]
    for batch in keyword_batches:
        keyword_query = " OR ".join(f'"{k}"' for k in batch)
        for country in country_filter_list:
            kwargs = dict(keyword=keyword_query, start_date=date_from, end_date=date_to)
            if country:
                kwargs["country"] = country
            log.info("Keyword query [country=%s]: %s", country or "any", keyword_query[:80])
            df = _search(kwargs)
            if not df.empty:
                log.info("  → %d results", len(df))
                all_dfs.append(df)
            time.sleep(0.5)  # be polite to the free API

    # Theme queries — use country filter (more reliable than language for GDELT DOC 2.0)
    for theme in themes:
        for country in country_filter_list:
            kwargs = dict(theme=theme, start_date=date_from, end_date=date_to)
            if country:
                kwargs["country"] = country
            log.info("Theme query [%s | country=%s]", theme, country or "any")
            df = _search(kwargs)
            if not df.empty:
                log.info("  → %d results", len(df))
                all_dfs.append(df)
            time.sleep(0.5)

    if not all_dfs:
        log.warning("No results from GDELT DOC 2.0. Check keywords/themes/date range.")
        return pd.DataFrame()

    # Combine, deduplicate, cap
    combined = (
        pd.concat(all_dfs, ignore_index=True)
        .drop_duplicates(subset="url")
        .head(max_articles)
        .reset_index(drop=True)
    )

    # Normalise column names to match downstream expectations
    combined = combined.rename(columns={
        "seendate":      "date_str",
        "domain":        "source",
        "language":      "language",
        "sourcecountry": "country",
    })

    # Add placeholder columns that BigQuery used to provide
    for col in ["themes", "locations", "persons", "organizations", "tone"]:
        if col not in combined.columns:
            combined[col] = None

    log.info("Total unique articles retrieved: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Full text extraction
# ---------------------------------------------------------------------------

def fetch_full_text(url: str, timeout: int = 15) -> str | None:
    """Download and extract main text from a URL using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if downloaded is None:
            return None
        return trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
    except Exception as exc:
        log.debug("Failed to fetch %s: %s", url, exc)
        return None


def add_full_text(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    """Download full text for all URLs in parallel."""
    urls  = df["url"].tolist()
    texts = [None] * len(urls)

    log.info("Downloading full text for %d articles (%d workers)...", len(urls), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(fetch_full_text, url): i for i, url in enumerate(urls)}
        with tqdm(total=len(urls), desc="Full text") as pbar:
            for future in as_completed(future_to_idx):
                texts[future_to_idx[future]] = future.result()
                pbar.update(1)

    df = df.copy()
    df["full_text"] = texts
    log.info("Full text retrieved for %d / %d articles.", sum(1 for t in texts if t), len(urls))
    return df


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, output_dir: str, tag: str = "") -> None:
    """Save DataFrame to CSV and Parquet."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"gdelt_{tag}_{ts}" if tag else f"gdelt_{ts}"
    csv_path     = Path(output_dir) / f"{stem}.csv"
    parquet_path = Path(output_dir) / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    log.info("Saved %d rows → %s", len(df), csv_path)
    log.info("Saved %d rows → %s", len(df), parquet_path)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def build_email_summary(df: pd.DataFrame, date_from: str, date_to: str) -> tuple[str, str, str]:
    """Build plain-text + HTML email summary of the daily fetch results."""
    df_ok     = df[df["full_text"].notna()] if "full_text" in df.columns else df
    total     = len(df)
    with_text = len(df_ok)

    lines_plain = [
        f"GDELT Daily Digest — {date_from} → {date_to}",
        f"Articoli trovati: {total}  |  Con full text: {with_text}",
        "=" * 60,
    ]
    lines_html = [
        f"<h2>GDELT Daily Digest — {date_from} → {date_to}</h2>",
        f"<p><b>Articoli trovati:</b> {total} &nbsp;|&nbsp; <b>Con full text:</b> {with_text}</p>",
        "<hr>",
    ]

    for _, r in df_ok.iterrows():
        text    = str(r.get("full_text") or r.get("title") or "")
        snippet = text[:300].replace("\n", " ").strip()
        source  = r.get("source", "")
        url     = r.get("url", "")
        lines_plain.append(f"\n[{source}]\n{url}\n{snippet}...\n")
        lines_html.append(
            f"<p><b>[{source}]</b><br>"
            f"<a href='{url}'>{url[:80]}</a><br>"
            f"{snippet}...</p><hr>"
        )

    subject    = f"[newspop] {total} articoli su fertilità — {date_from}"
    body_plain = "\n".join(lines_plain)
    body_html  = "\n".join(lines_html)
    return subject, body_plain, body_html


def write_summary_file(subject: str, body_plain: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "latest_subject.txt").write_text(subject)
    (Path(output_dir) / "latest_body.txt").write_text(body_plain)
    log.info("Email summary written to %s/latest_*.txt", output_dir)


def send_email(subject: str, body_plain: str, to_addr: str) -> None:
    """Send email via Gmail SMTP. Needs MAIL_USERNAME and MAIL_PASSWORD env vars."""
    username = os.environ.get("MAIL_USERNAME")
    password = os.environ.get("MAIL_PASSWORD")
    if not username or not password:
        log.warning("MAIL_USERNAME / MAIL_PASSWORD not set — skipping email.")
        return

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"newspop-bot <{username}>"
    msg["To"]      = to_addr
    msg.attach(MIMEText(body_plain, "plain", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(username, password)
        smtp.sendmail(username, to_addr, msg.as_string())
    log.info("Email sent to %s", to_addr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GDELT DOC 2.0 fertility news pipeline (free, no BigQuery)")
    parser.add_argument("--config",      default="config.json")
    parser.add_argument("--no-fulltext", action="store_true", help="Skip full text download")
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--auto-dates",  action="store_true", help="Auto set dates to last N days")
    parser.add_argument("--days",        type=int, default=1)
    parser.add_argument("--send-email",  action="store_true")
    parser.add_argument("--email-to",    default="n.barban@unibo.it")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.auto_dates:
        from datetime import timedelta
        today               = datetime.utcnow().date()
        config["date_to"]   = today.strftime("%Y-%m-%d")
        config["date_from"] = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
        log.info("Auto dates: %s → %s", config["date_from"], config["date_to"])

    output_dir = args.output_dir or config.get("output_dir", "data")

    # 1. Query GDELT DOC 2.0
    df = run_gdeltdoc(config)
    if df.empty:
        log.warning("No results. Exiting.")
        return

    # 2. Full text
    if not args.no_fulltext and config.get("full_text", True):
        df = add_full_text(df, workers=config.get("full_text_workers", 8))
    else:
        df["full_text"] = None

    # 3. Save
    save_results(df, output_dir)

    # 4. Email
    subject, body_plain, _ = build_email_summary(df, config["date_from"], config["date_to"])
    write_summary_file(subject, body_plain, output_dir)
    if args.send_email:
        send_email(subject, body_plain, args.email_to)


if __name__ == "__main__":
    main()
