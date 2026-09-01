"""Tests for issuer ticker normalisation.

`issuerTradingSymbol` on a Form 4 is free text, and filers treat it as such.
Every case below was taken from the live event store, not invented: 50 of 4,740
distinct symbols were unusable as written, and they failed the price backfill
with a vendor 400 that was parked as a "failure" -- which reads as absent data
when it was absent parsing.

The bias is the reason this matters more than 1.1% suggests. A company with two
share classes has two symbols to write in one field, so the malformed set skews
toward established mid and large caps (SIRI, LEN, HEI, GEF, PARA, NYCB) -- the
opposite of the direction a microcap study can afford to lose names in.
"""

from __future__ import annotations

import pytest

from tradezbotz.research.edgar import normalise_symbol


@pytest.mark.parametrize("raw,expected", [
    # Exchange prefixes and suffixes.
    ("NYSE:NYCB", "NYCB"),
    ("NYSE: KRC", "KRC"),
    ("NYSE: ZETA", "ZETA"),
    ("NASDAQ:OPI", "OPI"),
    ("NASDAQ:SVC", "SVC"),
    ("ASX:CRN", "CRN"),
    ("NTIP-NYSE", "NTIP"),
    ("NYSE/TRN", "TRN"),
    ("(NYSE:FBC)", "FBC"),
    # Punctuation wrappers.
    ("(SIRI)", "SIRI"),
    ("(CALX)", "CALX"),
    ("(LUMO)", "LUMO"),
    ("CHEA]", "CHEA"),
    ("QSAM)", "QSAM"),
    ("\"OMEX\"", "OMEX"),
    ("\"'LTRX\"", "LTRX"),
    # Two share classes in one field: the first wins.
    ("GEF,GEF.B", "GEF"),
    ("LEN, LEN.B", "LEN"),
    ("HEI, HEI.A", "HEI"),
    ("BFA, BFB", "BFA"),
    ("MOGA/MOGB", "MOGA"),
    ("GTII/GTBIF", "GTII"),
    ("WLY, WLYB", "WLY"),
    ("BBXIA/B", "BBXIA"),
    ("BCDA;BCDAW", "BCDA"),
    ("BFRG,BFRGW", "BFRG"),
    ("Z AND ZG", "Z"),
    ("BIO BIOB", "BIO"),
    ("CRDA CRDB", "CRDA"),
    ("CRDA -CRDB", "CRDA"),
    # A ticker written with spaces between its letters.
    ("N O G", "NOG"),
    # Already clean.
    ("AAPL", "AAPL"),
    ("BRK.B", "BRK.B"),
    ("RDS-A", "RDS-A"),
    ("aapl", "AAPL"),
    ("  MSFT  ", "MSFT"),
])
def test_real_symbols_from_the_store(raw, expected):
    assert normalise_symbol(raw) == expected


@pytest.mark.parametrize("raw", [
    "N/A", "NONE", "-", "--", "---", "", "   ", None,
    "1314152",          # a CIK in the ticker field
    "*H6ZMFDX",         # internal identifier
    "9QGNC@RY",
    "AB-LEND",          # not a ticker shape
])
def test_junk_is_dropped_rather_than_guessed_at(raw):
    """A wrong ticker attaches an insider's trade to another company's price
    series. That is far worse than dropping the filing, so anything not
    recognisably a ticker is discarded."""
    assert normalise_symbol(raw) == ""


def test_a_dual_class_filing_produces_one_symbol_not_two():
    """Splitting into two events would double-count a single filing -- one
    transaction becoming two observations in the backtest."""
    assert normalise_symbol("GEF, GEF-B") == "GEF"


def test_normalisation_is_idempotent():
    """It runs at ingest, and a re-ingest must not shift a symbol again."""
    for raw in ("NYSE: KRC", "(SIRI)", "GEF,GEF.B", "N O G", "AAPL"):
        once = normalise_symbol(raw)
        assert normalise_symbol(once) == once


def test_a_recovered_symbol_is_actually_usable():
    """The point of the exercise: what comes out has to be something a price
    vendor will accept. The parked failures were 400s from Alpaca on strings
    like '(SIRI)'."""
    import re
    tradeable = re.compile(r"^[A-Z][A-Z0-9]{0,6}(?:[.\-][A-Z0-9]{1,4})?$")
    for raw in ("NYSE:NYCB", "(SIRI)", "GEF,GEF.B", "N O G", "\"OMEX\""):
        out = normalise_symbol(raw)
        assert tradeable.match(out), f"{raw!r} -> {out!r} is still not tradeable"
