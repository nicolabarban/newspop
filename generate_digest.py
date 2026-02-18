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
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL    = "claude-sonnet-4-6"
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
# Call Claude API
# ---------------------------------------------------------------------------

def generate_digest(articles: list[dict], date_from: str, date_to: str) -> str:
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    user_prompt = build_user_prompt(articles, date_from, date_to)
    log.info("Sending %d articles to Claude (%s)...", len(articles), MODEL)

    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_prompt}],
        system=SYSTEM_PROMPT,
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_latest_parquet(data_dir: str) -> pd.DataFrame:
    parquet_files = sorted(Path(data_dir).glob("gdelt_*.parquet"), reverse=True)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    path = parquet_files[0]
    log.info("Loading %s", path)
    return pd.read_parquet(path), path


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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate digest with Claude API")
    parser.add_argument("--data-dir",   default="data",  help="Directory with parquet files")
    parser.add_argument("--output-dir", default="posts", help="Output directory for markdown")
    parser.add_argument("--parquet",    default=None,    help="Use a specific parquet file")
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


if __name__ == "__main__":
    main()
