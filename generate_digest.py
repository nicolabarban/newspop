"""
generate_digest.py — Generate a blog-style weekly digest using Claude API.

Reads the latest GDELT parquet file, sends articles to Claude Sonnet,
and saves the output as a dated Markdown file in posts/.

Usage:
    python generate_digest.py
    python generate_digest.py --data-dir data --output-dir posts
    python generate_digest.py --parquet data/gdelt_20260218_103109.parquet
"""

import argparse
import logging
import os
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import time

import anthropic
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL    = "claude-haiku-4-5-20251001"
MAX_CHARS_PER_ARTICLE = 2000   # truncate long articles before sending to API


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Sei un ricercatore esperto di demografia e politiche familiari.
Il tuo compito è leggere una lista di articoli giornalistici italiani raccolti
automaticamente e produrre una rassegna stampa settimanale in italiano.

Regole:
- Scarta gli articoli non pertinenti (sport, cronaca locale, necrologi, beauty, ecc.)
- Raggruppa gli articoli pertinenti per tema
- Scrivi in stile giornalistico chiaro e preciso
- Cita sempre la fonte e la data tra parentesi
- Il tono deve essere informativo e neutro
- Output: solo il testo Markdown della rassegna, senza commenti aggiuntivi
"""

def build_user_prompt(articles: list[dict], date_from: str, date_to: str) -> str:
    lines = [
        f"Ecco {len(articles)} articoli raccolti dal {date_from} al {date_to}.",
        "Scrivi una rassegna stampa settimanale in italiano in formato Markdown.",
        "Titolo: **Fertilità e Calo Demografico — Rassegna Stampa {date_from} / {date_to}**",
        "",
        "--- ARTICOLI ---",
        "",
    ]
    for i, a in enumerate(articles, 1):
        text = (a.get("full_text") or "").strip()[:MAX_CHARS_PER_ARTICLE]
        lines += [
            f"### Articolo {i}",
            f"**Fonte**: {a['source']}  |  **Data**: {a['date_str']}  |  **URL**: {a['url']}",
            "",
            text,
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Call Claude API via Batch (50% cheaper than synchronous requests)
# ---------------------------------------------------------------------------

def generate_digest(articles: list[dict], date_from: str, date_to: str) -> str:
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    user_prompt = build_user_prompt(articles, date_from, date_to)
    log.info("Submitting batch request: %d articles → Claude %s...", len(articles), MODEL)

    batch = client.messages.batches.create(
        requests=[{
            "custom_id": "weekly-digest",
            "params": {
                "model":      MODEL,
                "max_tokens": 4096,
                "system":     SYSTEM_PROMPT,
                "messages":   [{"role": "user", "content": user_prompt}],
            },
        }]
    )
    log.info("Batch submitted: %s — polling for results...", batch.id)

    # Poll until complete (max 20 minutes; batches usually finish in 1-3 min)
    for attempt in range(40):
        time.sleep(30)
        batch = client.messages.batches.retrieve(batch.id)
        log.info("Batch status [%d/40]: %s", attempt + 1, batch.processing_status)
        if batch.processing_status == "ended":
            break
    else:
        raise TimeoutError("Batch did not complete within 20 minutes")

    # Retrieve result
    for result in client.messages.batches.results(batch.id):
        if result.custom_id == "weekly-digest":
            if result.result.type == "succeeded":
                return result.result.message.content[0].text
            raise RuntimeError(f"Batch request failed: {result.result}")

    raise RuntimeError("No result found in batch response")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_latest_parquet(data_dir: str) -> pd.DataFrame:
    """Load and merge the latest GDELT parquet + any NewsData parquets from today."""
    data_path = Path(data_dir)

    # Latest GDELT file
    gdelt_files = sorted(data_path.glob("gdelt_*.parquet"), reverse=True)
    if not gdelt_files:
        raise FileNotFoundError(f"No gdelt parquet files found in {data_dir}")

    dfs = [pd.read_parquet(gdelt_files[0])]
    log.info("GDELT: %s  (%d rows)", gdelt_files[0].name, len(dfs[0]))

    # NewsData files from today (produced in the same workflow run)
    today = datetime.now().strftime("%Y%m%d")
    for f in sorted(data_path.glob(f"newsdata_{today}*.parquet"), reverse=True):
        nd = pd.read_parquet(f)
        log.info("NewsData: %s  (%d rows)", f.name, len(nd))
        dfs.append(nd)

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="url").reset_index(drop=True)
    log.info("Merged: %d unique articles from %d source(s)", len(df), len(dfs))
    return df, gdelt_files[0]


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_digest(text: str, output_dir: str, date_from: str) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    date_tag = date_from.replace("-", "")[:8]
    filename = f"{datetime.now().strftime('%Y-%m-%d')}_digest.md"
    out_path = Path(output_dir) / filename
    out_path.write_text(text, encoding="utf-8")
    log.info("Digest saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Send email
# ---------------------------------------------------------------------------

def send_digest_email(digest_text: str, subject: str, to_addr: str) -> None:
    """Send the generated digest via email (plain text + HTML).
    Reads MAIL_USERNAME and MAIL_PASSWORD from environment variables.
    """
    username = os.environ.get("MAIL_USERNAME")
    password = os.environ.get("MAIL_PASSWORD")
    if not username or not password:
        log.warning("MAIL_USERNAME / MAIL_PASSWORD not set — skipping email.")
        return

    # Simple markdown → HTML: wrap in <pre> for readability
    body_html = (
        "<html><body>"
        "<pre style='font-family:Arial,sans-serif;font-size:14px;white-space:pre-wrap'>"
        + digest_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre></body></html>"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"newspop-bot <{username}>"
    msg["To"]      = to_addr
    msg.attach(MIMEText(digest_text, "plain", "utf-8"))
    msg.attach(MIMEText(body_html,   "html",  "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(username, password)
        smtp.sendmail(username, to_addr, msg.as_string())
    log.info("Digest email sent to %s", to_addr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate digest with Claude API")
    parser.add_argument("--data-dir",   default="data",  help="Directory with parquet files")
    parser.add_argument("--output-dir", default="posts", help="Output directory for markdown")
    parser.add_argument("--parquet",    default=None,    help="Use a specific parquet file")
    parser.add_argument("--send-email", action="store_true", help="Send digest via email")
    parser.add_argument("--email-to",   default="n.barban@unibo.it", help="Recipient email address")
    args = parser.parse_args()

    # Load data
    if args.parquet:
        df = pd.read_parquet(args.parquet)
        path = Path(args.parquet)
    else:
        df, path = load_latest_parquet(args.data_dir)

    # Filter articles with full text
    df_ok = df[df["full_text"].notna()].reset_index(drop=True)
    log.info("Articles with full text: %d / %d", len(df_ok), len(df))

    if df_ok.empty:
        log.warning("No articles with full text — skipping digest generation.")
        return

    # Infer date range from data
    dates = df_ok["date_str"].dropna().sort_values()
    date_from = dates.iloc[0][:8]
    date_to   = dates.iloc[-1][:8]
    date_from_fmt = f"{date_from[:4]}-{date_from[4:6]}-{date_from[6:8]}"
    date_to_fmt   = f"{date_to[:4]}-{date_to[4:6]}-{date_to[6:8]}"

    articles = df_ok[["source", "date_str", "url", "full_text"]].to_dict("records")

    # Generate digest
    digest_text = generate_digest(articles, date_from_fmt, date_to_fmt)

    # Save
    save_digest(digest_text, args.output_dir, date_from_fmt)

    # Optionally send by email
    if args.send_email:
        subject = f"[newspop] Rassegna stampa fertilità — {date_from_fmt} / {date_to_fmt}"
        send_digest_email(digest_text, subject, args.email_to)


if __name__ == "__main__":
    main()
