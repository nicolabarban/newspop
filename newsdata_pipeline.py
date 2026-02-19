"""
newsdata_pipeline.py — Fetch Italian fertility/demographics news from NewsData.io.

Queries NewsData.io API for articles in Italian, saves results to data/
in the same parquet format as gdelt_pipeline.py so generate_digest.py
can merge both sources transparently.

Usage:
    python newsdata_pipeline.py
    python newsdata_pipeline.py --output-dir data --timeframe 48
    python newsdata_pipeline.py --output-dir data --timeframe 48 --send-email --email-to you@example.com

Requires:
    NEWSDATA_API_KEY environment variable (free key from https://newsdata.io)
"""

import argparse
import logging
import os
import smtplib
import ssl
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

NEWSDATA_URL = "https://newsdata.io/api/1/news"

# Query inviata a NewsData.io — operatori booleani supportati
QUERY = (
    "fertilità OR natalità OR denatalità OR "
    "\"calo demografico\" OR \"invecchiamento popolazione\" OR "
    "\"tasso di natalità\" OR infertilità OR fecondità"
)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_articles(api_key: str, timeframe: int = 48, max_pages: int = 5) -> list[dict]:
    """Fetch articles from NewsData.io (latest news endpoint, free plan)."""
    # Free plan supports: q, language, country, category, domain, page
    # timeframe requires a paid plan — not used here
    params = {
        "apikey":   api_key,
        "q":        QUERY,
        "language": "it",
    }

    articles   = []
    page_token = None

    for i in range(max_pages):
        if page_token:
            params["page"] = page_token

        log.info("Fetching page %d from NewsData.io...", i + 1)
        try:
            resp = requests.get(NEWSDATA_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            log.error("Request failed: %s", exc)
            break

        if resp.status_code == 429:
            log.warning("Rate limit — waiting 60 s...")
            time.sleep(60)
            continue

        if not resp.ok:
            log.error("NewsData HTTP %s: %s", resp.status_code, resp.text[:500])
            break

        data = resp.json()

        if data.get("status") != "success":
            log.error("NewsData error: %s", data.get("message", data))
            break

        results = data.get("results", [])
        articles.extend(results)
        log.info("  Page %d: %d articles (running total: %d)", i + 1, len(results), len(articles))

        page_token = data.get("nextPage")
        if not page_token:
            break

        time.sleep(1)   # cortesia al server

    return articles


# ---------------------------------------------------------------------------
# Normalize to shared schema
# ---------------------------------------------------------------------------

def normalize(raw: list[dict]) -> pd.DataFrame:
    """Convert NewsData.io results to the shared pipeline schema."""
    rows = []
    for a in raw:
        # Preferisce content (testo completo), poi description (sommario)
        text = (a.get("content") or a.get("description") or "").strip()

        # pubDate formato "YYYY-MM-DD HH:MM:SS" → YYYYMMDDHHMMSS
        pub = (a.get("pubDate") or "").replace("-", "").replace(" ", "").replace(":", "")

        rows.append({
            "date_str":         pub[:14],
            "source":           (a.get("source_name") or a.get("source_id") or "").strip(),
            "url":              a.get("link") or "",
            "themes":           "",
            "locations":        "",
            "persons":          "",
            "organizations":    "",
            "tone":             "",
            "translation_info": "srclc:ita",
            "full_text":        text if text else None,
        })

    df = pd.DataFrame(rows)
    df = df[df["url"] != ""].drop_duplicates(subset="url").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, output_dir: str) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"newsdata_{ts}"
    csv_path     = Path(output_dir) / f"{stem}.csv"
    parquet_path = Path(output_dir) / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    log.info("Saved %d rows → %s", len(df), parquet_path)
    return parquet_path


# ---------------------------------------------------------------------------
# Optional email summary
# ---------------------------------------------------------------------------

def send_summary_email(df: pd.DataFrame, to_addr: str) -> None:
    username = os.environ.get("MAIL_USERNAME")
    password = os.environ.get("MAIL_PASSWORD")
    if not username or not password:
        log.warning("MAIL_USERNAME / MAIL_PASSWORD not set — skip email.")
        return

    df_ok   = df[df["full_text"].notna()]
    subject = f"[newspop/newsdata] {len(df)} articoli italiani — {datetime.utcnow().date()}"

    lines = [
        f"NewsData.io fetch — {datetime.utcnow().date()}",
        f"Articoli trovati: {len(df)}  |  Con full text: {len(df_ok)}",
        "=" * 60,
    ]
    for _, r in df_ok.iterrows():
        snippet = str(r["full_text"])[:300].replace("\n", " ").strip()
        lines.append(f"\n[{r['source']}]\n{r['url']}\n{snippet}...\n")

    body = "\n".join(lines)
    msg  = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = f"newspop-bot <{username}>"
    msg["To"]      = to_addr
    msg.attach(MIMEText(body, "plain", "utf-8"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as smtp:
        smtp.login(username, password)
        smtp.sendmail(username, to_addr, msg.as_string())
    log.info("Summary email sent to %s", to_addr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Italian news from NewsData.io")
    parser.add_argument("--output-dir", default="data",              help="Output directory")
    parser.add_argument("--timeframe",  type=int, default=48,        help="Hours to look back (1-48, default: 48)")
    parser.add_argument("--max-pages",  type=int, default=5,         help="Max API pages to fetch")
    parser.add_argument("--send-email", action="store_true",         help="Send summary email")
    parser.add_argument("--email-to",   default="n.barban@unibo.it", help="Recipient address")
    args = parser.parse_args()

    api_key = os.environ.get("NEWSDATA_API_KEY")
    if not api_key:
        log.error("NEWSDATA_API_KEY not set — skipping NewsData fetch.")
        return

    raw = fetch_articles(api_key, timeframe=args.timeframe, max_pages=args.max_pages)
    if not raw:
        log.warning("No articles returned from NewsData.io")
        return

    df = normalize(raw)
    log.info("Unique articles: %d  |  with full text: %d",
             len(df), df["full_text"].notna().sum())

    save_results(df, args.output_dir)

    if args.send_email:
        send_summary_email(df, args.email_to)


if __name__ == "__main__":
    main()
