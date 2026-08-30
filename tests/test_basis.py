"""Tests for the price-basis dimension.

Two things carry the weight: that a cache written before `basis` existed
migrates without losing its bars (the CI copy holds ~2,200 symbols that cost
many hours to fetch), and that the two bases cannot be silently mixed.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from tradezbotz.research.prices import (
    ALPACA_ADJUSTMENT,
    BASES,
    BASIS_PRICE,
    BASIS_TOTAL,
    Bar,
    DualBasisSource,
    PriceCache,
    PriceError,
    Series,
)

START, END = date(2025, 3, 3), date(2025, 3, 21)


def series(symbol="T", closes=(10.0, 11.0, 12.0), basis=BASIS_PRICE):
    bars, day = [], START
    for c in closes:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        bars.append(Bar(day, c, c * 1.01, c * 0.99, c, 1_000))
        day += timedelta(days=1)
    return Series(symbol=symbol, bars=tuple(bars), requested_start=START,
                  requested_end=END, basis=basis)


# --- the two bases are kept apart ---------------------------------------------

def test_bases_are_stored_independently(tmp_path):
    cache = PriceCache(tmp_path / "bars.db")

    cache.put(series(closes=(10.0, 11.0, 12.0), basis=BASIS_PRICE))
    cache.put(series(closes=(9.0, 10.0, 12.0), basis=BASIS_TOTAL))

    price = cache.get("T", START, END, BASIS_PRICE)
    total = cache.get("T", START, END, BASIS_TOTAL)

    assert [b.close for b in price.bars] == [10.0, 11.0, 12.0]
    assert [b.close for b in total.bars] == [9.0, 10.0, 12.0]
    assert price.basis == BASIS_PRICE and total.basis == BASIS_TOTAL
    cache.close()


def test_writing_one_basis_does_not_satisfy_the_other(tmp_path):
    """Coverage is per basis. Otherwise a half-filled symbol reads as complete
    and the missing basis surfaces later as a silent hole."""
    cache = PriceCache(tmp_path / "bars.db")
    cache.put(series(basis=BASIS_PRICE))

    assert cache.covered("T", START, END, BASIS_PRICE) is True
    assert cache.covered("T", START, END, BASIS_TOTAL) is False
    cache.close()


def test_series_defaults_to_price_basis():
    """The safe default: price-only is what everything stored so far is."""
    assert Series("T", (), START, END).basis == BASIS_PRICE


def test_bases_helper_reports_what_is_held(tmp_path):
    cache = PriceCache(tmp_path / "bars.db")
    cache.put(series(basis=BASIS_PRICE))

    assert cache.bases("T") == [BASIS_PRICE]
    cache.put(series(basis=BASIS_TOTAL))
    assert sorted(cache.bases("T")) == [BASIS_PRICE, BASIS_TOTAL]
    cache.close()


def test_symbols_can_be_filtered_by_basis(tmp_path):
    cache = PriceCache(tmp_path / "bars.db")
    cache.put(series("AAA", basis=BASIS_PRICE))
    cache.put(series("BBB", basis=BASIS_PRICE))
    cache.put(series("AAA", basis=BASIS_TOTAL))

    assert cache.symbols() == ["AAA", "BBB"], "any basis counts"
    assert cache.symbols(BASIS_TOTAL) == ["AAA"]
    cache.close()


# --- migrating a cache written before basis existed ---------------------------

OLD_SCHEMA = """
CREATE TABLE bars (
    symbol TEXT NOT NULL, day TEXT NOT NULL, open REAL NOT NULL,
    high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume REAL NOT NULL, PRIMARY KEY (symbol, day)
);
CREATE TABLE fetches (
    symbol TEXT PRIMARY KEY, start_day TEXT NOT NULL, end_day TEXT NOT NULL,
    is_active INTEGER, fetched_at TEXT NOT NULL
);
"""


def make_legacy_cache(path):
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.executemany(
        "INSERT INTO bars VALUES (?,?,?,?,?,?,?)",
        [("XELB", "2025-03-03", 7.0, 7.1, 6.9, 7.03, 1000),
         ("XELB", "2025-03-04", 7.03, 7.2, 7.0, 7.15, 1200)],
    )
    conn.execute("INSERT INTO fetches VALUES (?,?,?,?,?)",
                 ("XELB", "2025-03-03", "2025-03-21", 1, "2025-03-22T00:00:00+00:00"))
    conn.commit()
    conn.close()


def test_legacy_cache_migrates_without_losing_bars(tmp_path):
    """The CI cache holds ~2,200 symbols that cost many hours at 5 req/min.
    Losing them to a schema change would be the expensive kind of mistake."""
    path = tmp_path / "bars.db"
    make_legacy_cache(path)

    cache = PriceCache(path)
    got = cache.get("XELB", START, END, BASIS_PRICE)

    assert [b.close for b in got.bars] == [7.03, 7.15]
    assert got.is_active is True, "metadata survives too"
    cache.close()


def test_legacy_rows_are_stamped_price_not_total(tmp_path):
    """Everything cached so far came from Massive, which is price-only. Stamping
    it `total` would silently mix bases in every subsequent return."""
    path = tmp_path / "bars.db"
    make_legacy_cache(path)

    cache = PriceCache(path)

    assert cache.bases("XELB") == [BASIS_PRICE]
    assert cache.get("XELB", START, END, BASIS_TOTAL).bars == ()
    cache.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "bars.db"
    make_legacy_cache(path)

    PriceCache(path).close()
    cache = PriceCache(path)

    assert len(cache.get("XELB", START, END, BASIS_PRICE).bars) == 2
    cache.close()


def test_migration_drops_the_scratch_table(tmp_path):
    path = tmp_path / "bars.db"
    make_legacy_cache(path)
    PriceCache(path).close()

    conn = sqlite3.connect(path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    conn.close()

    assert not any(n.endswith("_pre_basis") for n in names)


# --- dual-basis source ---------------------------------------------------------

class FakeSource:
    def __init__(self, basis, offset=0.0, fail=False):
        self.basis, self.offset, self.fail = basis, offset, fail
        self.calls = []

    def daily_bars(self, symbol, start, end):
        self.calls.append(symbol)
        if self.fail:
            raise PriceError("vendor said no")
        return series(symbol, closes=(10.0 + self.offset,), basis=self.basis)


def test_dual_source_fetches_every_basis():
    price = FakeSource(BASIS_PRICE)
    total = FakeSource(BASIS_TOTAL, offset=-1.0)

    got = DualBasisSource({BASIS_PRICE: price, BASIS_TOTAL: total}).daily_bars(
        "T", START, END)

    assert price.calls == ["T"] and total.calls == ["T"]
    assert got.basis == BASIS_PRICE, "the primary is returned"


def test_dual_source_propagates_a_failure_on_either_basis():
    """Half a symbol is worse than none: the runner would mark it done and the
    missing basis would show up later as a silent coverage hole."""
    src = DualBasisSource({BASIS_PRICE: FakeSource(BASIS_PRICE),
                           BASIS_TOTAL: FakeSource(BASIS_TOTAL, fail=True)})

    with pytest.raises(PriceError):
        src.daily_bars("T", START, END)


def test_dual_source_requires_all_bases():
    with pytest.raises(PriceError, match="no source supplied"):
        DualBasisSource({BASIS_PRICE: FakeSource(BASIS_PRICE)})


def test_every_basis_maps_to_an_alpaca_adjustment():
    assert set(ALPACA_ADJUSTMENT) == set(BASES)
    assert ALPACA_ADJUSTMENT[BASIS_PRICE] == "split", "price-only"
    assert ALPACA_ADJUSTMENT[BASIS_TOTAL] == "all", "split + dividend"


def test_unknown_basis_is_rejected_at_construction(monkeypatch):
    from tradezbotz.research.prices import AlpacaPriceSource

    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "k")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "s")

    with pytest.raises(PriceError, match="unknown basis"):
        AlpacaPriceSource(basis="adjusted-somehow")
