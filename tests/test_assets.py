"""Tests for symbol classification.

The bug this module was written to expose, and then nearly reproduced itself:
a symbol returning no price data was indistinguishable from one that never
existed, so every coverage number was a guess.

The sharpest test here is `test_the_bulk_list_alone_understates_delistings`.
Alpaca's bulk asset list silently omits recently delisted names while the
per-symbol endpoint still answers for them, and reading only the bulk list put
1,001 real companies into `unknown` and reported 99.5% survivorship. The failure
direction is what makes it dangerous: it reports a universe as healthier than it
is, which is the same direction survivorship bias itself runs.
"""

from __future__ import annotations

import pytest

from tradezbotz.research.assets import (
    DELISTED,
    LISTED,
    OTC,
    UNKNOWN,
    Asset,
    AssetCatalog,
    describe,
    resolve_unknown,
)


def catalog(tmp_path, assets=()) -> AssetCatalog:
    c = AssetCatalog(tmp_path / "a.db")
    if assets:
        c.put_many(assets)
    return c


def listed(sym, exchange="NASDAQ"):
    return Asset(sym, exchange, "active", True)


def delisted(sym, exchange="NYSE"):
    return Asset(sym, exchange, "inactive", False)


def otc(sym):
    return Asset(sym, "OTC", "active", True)


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("asset,expected", [
    (Asset("AAPL", "NASDAQ", "active", True), LISTED),
    (Asset("F", "NYSE", "active", True), LISTED),
    (Asset("SPY", "ARCA", "active", True), LISTED),
    (Asset("X", "BATS", "active", True), LISTED),
    (Asset("Y", "AMEX", "active", True), LISTED),
    (Asset("MDSO", "NASDAQ", "inactive", False), DELISTED),
    (Asset("GGP", "NYSE", "inactive", False), DELISTED),
    (Asset("AECX", "OTC", "inactive", False), OTC),
    (Asset("SBIG", "OTC", "active", True), OTC),
])
def test_classification(asset, expected):
    assert asset.classification == expected


def test_a_symbol_not_in_the_catalog_is_unknown(tmp_path):
    c = catalog(tmp_path, [listed("AAPL")])
    assert c.classify("AAPL") == LISTED
    assert c.classify("NOPE") == UNKNOWN
    c.close()


def test_classification_is_case_insensitive(tmp_path):
    c = catalog(tmp_path, [listed("AAPL")])
    assert c.classify("aapl") == LISTED
    c.close()


def test_breakdown_counts_every_bucket(tmp_path):
    c = catalog(tmp_path, [listed("A"), listed("B"), delisted("C"), otc("D")])

    out = c.breakdown(["A", "B", "C", "D", "E"])

    assert out == {LISTED: 2, DELISTED: 1, OTC: 1, UNKNOWN: 1}
    c.close()


# --- the vendor gap ---------------------------------------------------------

class FakeSession:
    """Stands in for the per-symbol endpoint the bulk list omits."""

    def __init__(self, known):
        self.known = known
        self.calls = []

    def get(self, url, **kw):
        symbol = url.rsplit("/", 1)[-1]
        self.calls.append(symbol)
        return FakeResponse(self.known.get(symbol))


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200 if payload else 404

    def raise_for_status(self):
        if self.status_code == 404:
            raise AssertionError("caller should check status_code first")

    def json(self):
        return self._payload


def test_the_bulk_list_alone_understates_delistings(tmp_path):
    """The measured failure. Reading only the bulk list left 1,001 real
    companies in `unknown` and reported 99.5% survivorship; resolving them moved
    641 into `delisted` and brought it to 84.4%.

    The direction is what matters: the error makes a universe look healthier
    than it is, which is the same direction survivorship bias runs, so it would
    never have looked wrong.
    """
    universe = ["AAPL", "LEG", "CMA", "JUNK"]
    c = catalog(tmp_path, [listed("AAPL")])

    before = c.breakdown(universe)
    assert before[UNKNOWN] == 3
    assert before[DELISTED] == 0

    session = FakeSession({
        "LEG": {"symbol": "LEG", "exchange": "NYSE", "status": "inactive",
                "tradable": False, "name": "Leggett & Platt"},
        "CMA": {"symbol": "CMA", "exchange": "NYSE", "status": "inactive",
                "tradable": False, "name": "Comerica"},
    })
    stats = resolve_unknown(c, universe, session=session)

    after = c.breakdown(universe)
    assert stats["resolved"] == 2
    assert stats["absent"] == 1
    assert after[DELISTED] == 2
    assert after[UNKNOWN] == 1
    c.close()


def test_a_confirmed_absence_is_not_retried(tmp_path):
    """A 404 is a stable fact; "not in the bulk list" is not. Retrying every run
    would spend the rate limit re-learning the same answer."""
    c = catalog(tmp_path)
    session = FakeSession({})

    resolve_unknown(c, ["JUNK"], session=session)
    first = list(session.calls)
    resolve_unknown(c, ["JUNK"], session=session)

    assert first == ["JUNK"]
    assert session.calls == ["JUNK"], "second pass made no request"
    c.close()


def test_resolution_does_not_relookup_what_the_bulk_list_had(tmp_path):
    c = catalog(tmp_path, [listed("AAPL")])
    session = FakeSession({})

    resolve_unknown(c, ["AAPL"], session=session)

    assert session.calls == []
    c.close()


def test_a_failing_lookup_does_not_end_the_pass(tmp_path):
    class Exploding(FakeSession):
        def get(self, url, **kw):
            if url.endswith("BOOM"):
                raise ConnectionError("network")
            return super().get(url, **kw)

    c = catalog(tmp_path)
    session = Exploding({"OK": {"symbol": "OK", "exchange": "NYSE",
                                "status": "active", "tradable": True}})

    stats = resolve_unknown(c, ["BOOM", "OK"], session=session)

    assert stats["failed"] == 1
    assert stats["resolved"] == 1
    c.close()


# --- the survivorship report ------------------------------------------------

def test_an_all_surviving_universe_is_flagged():
    text = describe({LISTED: 1000, DELISTED: 2, OTC: 0, UNKNOWN: 0})

    assert "WARNING" in text
    assert "selected for survival" in text


def test_a_universe_with_real_attrition_is_not_flagged():
    text = describe({LISTED: 3479, DELISTED: 641, OTC: 316, UNKNOWN: 304})

    assert "WARNING" not in text
    assert "84.4%" in text


def test_survivorship_ignores_otc_and_unknown():
    """Both are excluded from the ratio on purpose: an OTC name did not
    necessarily delist, and an unknown one may never have been listed. Counting
    either would move the number without meaning anything."""
    a = describe({LISTED: 80, DELISTED: 20, OTC: 0, UNKNOWN: 0})
    b = describe({LISTED: 80, DELISTED: 20, OTC: 500, UNKNOWN: 500})

    assert "80.0%" in a and "80.0%" in b
