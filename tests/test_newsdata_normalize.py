"""Tests for newsdata_pipeline.normalize and the paywall-stub fix.

NewsData.io's free plan returns "ONLY AVAILABLE IN PAID PLANS" in the
`content` field. These tests pin the behaviour that we (a) never store
that stub as full_text, (b) fall back to `description`, and (c) hydration
fills `full_text` from `description` when trafilatura returns nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newsdata_pipeline import PAYWALL_STUB, _clean_content, hydrate_full_text, normalize


# ---------------------------------------------------------------------------
# _clean_content
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (PAYWALL_STUB, ""),
        (f"  {PAYWALL_STUB}  ", ""),
        (f"prefix {PAYWALL_STUB} suffix", ""),  # stub anywhere inside → drop
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("real article body", "real article body"),
        ("  padded body  ", "padded body"),
    ],
)
def test_clean_content(raw: str | None, expected: str) -> None:
    assert _clean_content(raw) == expected


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def _make_article(**overrides: object) -> dict:
    """Build a NewsData.io-shaped article with sensible defaults."""
    base = {
        "link":        "https://example.com/a",
        "content":     None,
        "description": "",
        "pubDate":     "2026-06-08 10:00:00",
        "source_name": "Example",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_normalize_strips_paywall_stub_and_uses_description() -> None:
    raw = [_make_article(
        link="https://a.example/1",
        content=PAYWALL_STUB,
        description="Short summary A",
    )]
    df = normalize(raw)
    assert len(df) == 1
    assert df["full_text"].iloc[0] == "Short summary A"
    assert PAYWALL_STUB not in df["full_text"].fillna("").str.cat()


@pytest.mark.unit
def test_normalize_prefers_real_content_over_description() -> None:
    raw = [_make_article(
        link="https://b.example/2",
        content="Real body B",
        description="ignored summary",
    )]
    df = normalize(raw)
    assert df["full_text"].iloc[0] == "Real body B"


@pytest.mark.unit
def test_normalize_returns_none_when_no_text_available() -> None:
    raw = [_make_article(
        link="https://c.example/3",
        content=None,
        description="",
    )]
    df = normalize(raw)
    assert df["full_text"].isna().iloc[0]


@pytest.mark.unit
def test_normalize_drops_rows_without_url() -> None:
    raw = [
        _make_article(link="",     content="x"),
        _make_article(link="https://ok.example", content="y"),
    ]
    df = normalize(raw)
    assert list(df["url"]) == ["https://ok.example"]


@pytest.mark.unit
def test_normalize_deduplicates_by_url() -> None:
    raw = [
        _make_article(link="https://dup.example", content="first"),
        _make_article(link="https://dup.example", content="second"),
    ]
    df = normalize(raw)
    assert len(df) == 1


@pytest.mark.unit
def test_normalize_formats_date_str() -> None:
    raw = [_make_article(pubDate="2026-06-08 10:15:30", content="x")]
    df = normalize(raw)
    assert df["date_str"].iloc[0] == "20260608101530"


@pytest.mark.unit
def test_normalize_carries_description_for_hydration() -> None:
    raw = [_make_article(content=PAYWALL_STUB, description="summary")]
    df = normalize(raw)
    # description column is kept until hydrate_full_text drops it
    assert "description" in df.columns
    assert df["description"].iloc[0] == "summary"


@pytest.mark.unit
def test_normalize_schema_matches_shared_format() -> None:
    raw = [_make_article(content="body")]
    df = normalize(raw)
    expected = {
        "date_str", "source", "url", "themes", "locations", "persons",
        "organizations", "tone", "translation_info", "description", "full_text",
    }
    assert expected.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# hydrate_full_text fallback (without trafilatura)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hydrate_falls_back_to_description(monkeypatch: pytest.MonkeyPatch) -> None:
    """When trafilatura returns nothing, we fall back to the API description."""

    def fake_add_full_text(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
        out = df.copy()
        # Simulate: row 0 fails, row 1 extracted, row 2 fails (no description either)
        out["full_text"] = [None, "extracted body", None]
        return out

    # Inject a stub gdelt_pipeline so the lazy import inside hydrate_full_text
    # resolves without requiring trafilatura.
    import types
    stub = types.ModuleType("gdelt_pipeline")
    stub.add_full_text = fake_add_full_text
    monkeypatch.setitem(sys.modules, "gdelt_pipeline", stub)

    df = pd.DataFrame({
        "url":         ["u1", "u2", "u3"],
        "description": ["desc1", "desc2", ""],
        "full_text":   [None, None, None],
    })
    result = hydrate_full_text(df)

    assert "description" not in result.columns
    assert result["full_text"].tolist() == ["desc1", "extracted body", None]


@pytest.mark.unit
def test_hydrate_full_text_handles_empty_df() -> None:
    df = pd.DataFrame(columns=["url", "description", "full_text"])
    result = hydrate_full_text(df)
    assert result.empty
