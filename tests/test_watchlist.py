"""Tests for the watchlist.

The firewall carries the weight. A watchlist that could reach the backtest
universe would be the most effective way to fool ourselves available here, and
unlike every other bias in this repository nothing in the data would reveal it.
"""

from __future__ import annotations

from datetime import date

import pytest

from tradezbotz.research.watchlist import (
    WatchedSymbol,
    Watchlist,
    WatchlistError,
    guard_universe,
    prioritise,
)


@pytest.fixture
def wl(tmp_path):
    return Watchlist(tmp_path / "watchlist.yml")


# --- storage --------------------------------------------------------------------

def test_missing_file_is_an_empty_list_not_an_error(wl):
    assert wl.load() == []
    assert wl.symbols() == []


def test_add_and_read_back(wl):
    wl.add("nvda", "mentions move markets")

    entries = wl.load()

    assert len(entries) == 1
    assert entries[0].symbol == "NVDA", "normalised to upper case"
    assert entries[0].reason == "mentions move markets"


def test_a_reason_is_required(wl):
    """An undocumented entry becomes superstition: nobody remembers why it is
    there, so nobody is willing to remove it."""
    with pytest.raises(WatchlistError, match="reason is required"):
        wl.add("NVDA", "")
    with pytest.raises(WatchlistError, match="reason is required"):
        wl.add("NVDA", "   ")


def test_empty_symbol_is_rejected(wl):
    with pytest.raises(WatchlistError, match="cannot be empty"):
        wl.add("  ", "some reason")


def test_adding_twice_updates_rather_than_duplicates(wl):
    wl.add("NVDA", "first reason")
    wl.add("nvda", "better reason")

    entries = wl.load()

    assert len(entries) == 1
    assert entries[0].reason == "better reason"


def test_remove(wl):
    wl.add("NVDA", "r")
    wl.add("XELB", "r")

    assert wl.remove("nvda") is True
    assert wl.symbols() == ["XELB"]
    assert wl.remove("NOTTHERE") is False


def test_contains_is_case_insensitive(wl):
    wl.add("NVDA", "r")

    assert wl.contains("nvda") is True
    assert wl.contains("TSLA") is False


def test_the_file_carries_the_warning(wl):
    """The caution belongs in the file, because the file is what gets edited
    six months from now."""
    wl.add("NVDA", "r")

    text = wl.path.read_text(encoding="utf-8")

    assert "never the backtest" in text
    assert "lookahead" in text


def test_malformed_entries_are_skipped_not_fatal(wl):
    wl.path.write_text("symbols:\n- notadict\n- symbol: ''\n- symbol: OK\n",
                       encoding="utf-8")

    entries = wl.load()

    assert [e.symbol for e in entries] == ["OK"]


def test_entry_without_a_date_still_loads(wl):
    wl.path.write_text("symbols:\n- symbol: NVDA\n  reason: r\n", encoding="utf-8")

    assert wl.load()[0].added == date.today()


# --- prioritisation ---------------------------------------------------------------

def test_watched_symbols_come_first():
    out = prioritise(["AAA", "NVDA", "BBB", "XELB"], ["XELB", "NVDA"])

    assert out[:2] == ["NVDA", "XELB"], "watched first, original order kept"
    assert out[2:] == ["AAA", "BBB"]


def test_prioritise_never_adds_or_drops():
    """Reordering only. A watchlist that could remove symbols would quietly
    shrink the measured universe -- the failure this module exists to prevent."""
    pending = ["AAA", "BBB", "CCC"]

    out = prioritise(pending, ["ZZZ", "AAA"])

    assert sorted(out) == sorted(pending)
    assert len(out) == len(pending)


def test_prioritise_with_no_watchlist_is_identity():
    pending = ["AAA", "BBB"]

    assert prioritise(pending, []) == pending


def test_prioritise_is_case_insensitive():
    assert prioritise(["nvda", "aaa"], ["NVDA"])[0] == "nvda"


# --- the firewall -------------------------------------------------------------------

def test_a_universe_of_watched_names_is_refused():
    """The whole point. Measuring only hand-picked symbols is lookahead through
    the researcher's own memory, and no guardrail here can detect it."""
    watched = ["NVDA", "XELB", "ARI"]

    with pytest.raises(WatchlistError, match="refusing to measure"):
        guard_universe(["NVDA", "XELB", "ARI"], watched)


def test_a_broad_universe_containing_watched_names_is_fine():
    """Watched symbols are not excluded from research -- they simply cannot BE
    the research population."""
    watched = ["NVDA", "XELB"]
    universe = ["NVDA", "XELB"] + [f"S{i}" for i in range(200)]

    guard_universe(universe, watched)   # must not raise


def test_guard_is_inert_without_a_watchlist():
    guard_universe(["A", "B"], [])
    guard_universe([], ["NVDA"])


def test_guard_allows_a_small_universe_that_is_mostly_unwatched():
    guard_universe([f"S{i}" for i in range(20)] + ["NVDA"], ["NVDA"])
