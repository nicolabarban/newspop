"""Tests for generate_digest.build_user_prompt — verifying the title carries
the consultation window dates, not arbitrary article dates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_digest import _to_it_date, build_user_prompt


# ---------------------------------------------------------------------------
# _to_it_date
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    ("iso", "expected"),
    [
        ("2026-06-08", "08/06/2026"),
        ("2026-01-01", "01/01/2026"),
        ("2026-12-31", "31/12/2026"),
    ],
)
def test_to_it_date(iso: str, expected: str) -> None:
    assert _to_it_date(iso) == expected


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------

def _sample_article() -> dict:
    return {
        "source":    "Corriere",
        "date_str":  "20260607",
        "url":       "https://example.com/a",
        "full_text": "Lorem ipsum dolor sit amet.",
    }


@pytest.mark.unit
def test_prompt_title_uses_consultation_window() -> None:
    prompt = build_user_prompt([_sample_article()], "2026-06-01", "2026-06-08")
    # Italian DD/MM/YYYY in title
    assert "01/06/2026 / 08/06/2026" in prompt
    # Header sentence keeps ISO format
    assert "dal 2026-06-01 al 2026-06-08" in prompt


@pytest.mark.unit
def test_prompt_has_no_unsubstituted_placeholders() -> None:
    """Regression: the old code shipped a literal '{date_from}' to Claude."""
    prompt = build_user_prompt([_sample_article()], "2026-06-01", "2026-06-08")
    assert "{date_from}" not in prompt
    assert "{date_to}" not in prompt


@pytest.mark.unit
def test_prompt_includes_explicit_instruction_about_title_dates() -> None:
    """The model must be told to honour the window, not the article dates."""
    prompt = build_user_prompt([_sample_article()], "2026-06-01", "2026-06-08")
    assert "intervallo di consultazione" in prompt


@pytest.mark.unit
def test_prompt_includes_article_metadata() -> None:
    prompt = build_user_prompt([_sample_article()], "2026-06-01", "2026-06-08")
    assert "Corriere" in prompt
    assert "https://example.com/a" in prompt
    assert "Lorem ipsum" in prompt


@pytest.mark.unit
def test_prompt_handles_multiple_articles() -> None:
    articles = [_sample_article(), {**_sample_article(), "source": "Avvenire"}]
    prompt = build_user_prompt(articles, "2026-06-01", "2026-06-08")
    assert "Ecco 2 articoli" in prompt
    assert "### Articolo 1" in prompt
    assert "### Articolo 2" in prompt
