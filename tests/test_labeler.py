"""Tests for forward-return labelling.

The entry-timing tests are the important ones. Every other bug in this repo
loses money slowly; getting entry timing wrong invents returns that never
existed and makes a worthless strategy look excellent.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tradezbotz.research.labeler import (
    Coverage,
    Labeller,
    coverage_report,
    label_event,
)
from tradezbotz.research.prices import Bar, Series

ET = ZoneInfo("America/New_York")


def make_series(symbol="TEST", start=date(2025, 3, 3), n=40, is_active=True, step=1.0):
    """Consecutive weekday bars where close = open + step, open rises by 1/day."""
    bars, day, i = [], start, 0
    while len(bars) < n:
        if day.weekday() < 5:
            o = 100.0 + i
            bars.append(Bar(day=day, open=o, high=o + 2, low=o - 2, close=o + step, volume=1e6))
            i += 1
        day += timedelta(days=1)
    return Series(
        symbol=symbol, bars=tuple(bars),
        requested_start=start, requested_end=bars[-1].day, is_active=is_active,
    )


def et(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


# --- entry timing ------------------------------------------------------------

def test_after_hours_event_enters_next_session_open():
    """A Form 4 hitting at 18:40 ET cannot be traded at that day's close."""
    s = make_series(start=date(2025, 3, 3))
    lab = label_event(s, et(2025, 3, 4, 18, 40))

    assert lab.entry_day == date(2025, 3, 5), "must skip to the next session"
    assert lab.entry_price == pytest.approx(102.0)


def test_premarket_event_can_take_the_same_session_open():
    """An event disseminated at 06:00 ET is actionable at that day's 09:30 open."""
    s = make_series(start=date(2025, 3, 3))
    lab = label_event(s, et(2025, 3, 5, 6, 0))

    assert lab.entry_day == date(2025, 3, 5)


def test_event_exactly_at_open_is_treated_as_too_late():
    s = make_series(start=date(2025, 3, 3))
    lab = label_event(s, et(2025, 3, 5, 9, 30))

    assert lab.entry_day == date(2025, 3, 6)


def test_friday_evening_event_enters_monday():
    s = make_series(start=date(2025, 3, 3))
    lab = label_event(s, et(2025, 3, 7, 20, 0))  # Friday night

    assert lab.entry_day == date(2025, 3, 10), "weekend must be skipped"


def test_entry_never_precedes_the_event():
    s = make_series(start=date(2025, 3, 3))
    for hour in (1, 6, 9, 10, 16, 23):
        lab = label_event(s, et(2025, 3, 5, hour))
        assert lab.entry_day >= date(2025, 3, 5)


# --- return computation ------------------------------------------------------

def test_returns_measured_from_entry_open_not_event_close():
    s = make_series(start=date(2025, 3, 3), step=1.0)
    lab = label_event(s, et(2025, 3, 4, 18, 0), horizons=(0,))

    # Entry 2025-03-05 open = 102.0, same-session close = 103.0
    assert lab.entry_price == pytest.approx(102.0)
    assert lab.returns[0] == pytest.approx(103.0 / 102.0 - 1.0)


def test_multiple_horizons_resolve():
    s = make_series(start=date(2025, 3, 3), n=40)
    lab = label_event(s, et(2025, 3, 4, 18, 0), horizons=(0, 1, 5))

    assert set(lab.returns) == {0, 1, 5}
    assert lab.coverage is Coverage.COMPLETE
    # Prices rise monotonically, so longer horizons return more.
    assert lab.returns[5] > lab.returns[1] > lab.returns[0]


# --- coverage ----------------------------------------------------------------

def test_no_bars_is_no_data():
    s = Series("GONE", (), date(2025, 3, 1), date(2025, 4, 1), is_active=False)
    lab = label_event(s, et(2025, 3, 4, 18, 0))

    assert lab.coverage is Coverage.NO_DATA
    assert lab.usable is False


def test_event_after_last_bar_has_no_entry():
    s = make_series(start=date(2025, 3, 3), n=5)
    lab = label_event(s, et(2025, 6, 1, 18, 0))

    assert lab.coverage is Coverage.NO_ENTRY_BAR
    assert lab.last_available_day is not None


def test_delisting_is_recorded_not_dropped():
    """The failure case that survivorship bias hides. It must survive labelling
    as a distinct, countable outcome."""
    s = make_series(start=date(2025, 3, 3), n=6, is_active=False)
    lab = label_event(s, et(2025, 3, 4, 18, 0), horizons=(0, 1, 20))

    assert lab.coverage is Coverage.DELISTED_DURING_WINDOW
    assert 0 in lab.returns and 20 not in lab.returns
    assert lab.usable is True, "short horizons still carry information"


def test_still_active_but_short_window_is_partial_not_delisted():
    """Running out of bars because the horizon extends past today is a very
    different thing from the company ceasing to exist."""
    s = make_series(start=date(2025, 3, 3), n=6, is_active=True)
    lab = label_event(s, et(2025, 3, 4, 18, 0), horizons=(0, 20))

    assert lab.coverage is Coverage.PARTIAL


def test_coverage_report_surfaces_the_missing_fraction():
    complete = [label_event(make_series(n=40), et(2025, 3, 4, 18, 0)) for _ in range(7)]
    dead = [
        label_event(make_series(n=6, is_active=False), et(2025, 3, 4, 18, 0))
        for _ in range(2)
    ]
    empty = [
        label_event(Series("X", (), date(2025, 3, 1), date(2025, 4, 1)), et(2025, 3, 4, 18, 0))
    ]

    rep = coverage_report(complete + dead + empty)

    assert rep["total"] == 10
    assert rep["complete"] == 7
    assert rep["delisted_during_window"] == 2
    assert rep["complete_rate"] == pytest.approx(0.7)
    assert rep["delisting_rate"] == pytest.approx(0.2)
    assert rep["missing_rate"] == pytest.approx(0.1)


def test_coverage_report_handles_empty():
    assert coverage_report([]) == {"total": 0}


# --- Labeller batching -------------------------------------------------------

class FakeSource:
    def __init__(self):
        self.calls: list[str] = []

    def daily_bars(self, symbol, start, end):
        self.calls.append(symbol)
        return make_series(symbol=symbol, start=date(2025, 3, 3), n=40)


def test_each_symbol_is_fetched_once_regardless_of_event_count():
    """At 5 requests/min, refetching per event is the difference between a
    ten-hour backfill and a ten-day one."""
    src = FakeSource()
    events = [
        {"symbol": "AAPL", "observed_at": et(2025, 3, 4, 18, 0)},
        {"symbol": "AAPL", "observed_at": et(2025, 3, 6, 18, 0)},
        {"symbol": "AAPL", "observed_at": et(2025, 3, 7, 18, 0)},
        {"symbol": "MSFT", "observed_at": et(2025, 3, 4, 18, 0)},
    ]

    labels = Labeller(src).label(events)

    assert len(labels) == 4
    assert sorted(src.calls) == ["AAPL", "MSFT"]


def test_events_without_a_symbol_are_labelled_not_dropped():
    """Changed deliberately. Dropping them shortened the output and broke
    positional alignment with the input, which silently mispaired every
    downstream payload. An unlabellable event is NO_DATA, not absent."""
    src = FakeSource()
    labels = Labeller(src).label(
        [{"symbol": None, "observed_at": et(2025, 3, 4, 18, 0)},
         {"symbol": "AAPL", "observed_at": et(2025, 3, 4, 18, 0)}]
    )

    assert len(labels) == 2
    assert labels[0].coverage is Coverage.NO_DATA
    assert labels[1].symbol == "AAPL"


def test_iso_string_timestamps_are_accepted():
    src = FakeSource()
    labels = Labeller(src).label(
        [{"symbol": "AAPL", "observed_at": et(2025, 3, 4, 18, 0).isoformat()}]
    )

    assert labels[0].entry_day == date(2025, 3, 5)


# --- input alignment ---------------------------------------------------------

def test_labels_are_aligned_to_input_order():
    """Regression: labels were returned grouped by symbol, so zipping them with
    the input paired every label to the wrong event. Downstream this made every
    backtest selector return identical results -- silently, because both lists
    were the right length."""
    src = FakeSource()
    events = [
        {"symbol": "AAA", "observed_at": et(2025, 3, 4, 18)},
        {"symbol": "BBB", "observed_at": et(2025, 3, 4, 18)},
        {"symbol": "AAA", "observed_at": et(2025, 3, 5, 18)},
        {"symbol": "CCC", "observed_at": et(2025, 3, 4, 18)},
        {"symbol": "BBB", "observed_at": et(2025, 3, 6, 18)},
    ]

    labels = Labeller(src).label(events)

    assert [e["symbol"] for e in events] == [lab.symbol for lab in labels]


def test_alignment_holds_when_a_symbol_is_missing():
    """Events without a symbol are labelled NO_DATA, not dropped, so index
    correspondence survives."""
    src = FakeSource()
    events = [
        {"symbol": "AAA", "observed_at": et(2025, 3, 4, 18)},
        {"symbol": None, "observed_at": et(2025, 3, 4, 18)},
        {"symbol": "BBB", "observed_at": et(2025, 3, 4, 18)},
    ]

    labels = Labeller(src).label(events)

    assert len(labels) == 3
    assert labels[1].coverage is Coverage.NO_DATA
    assert labels[0].symbol == "AAA" and labels[2].symbol == "BBB"


def test_each_symbol_still_fetched_once_after_the_ordering_fix():
    src = FakeSource()
    events = [
        {"symbol": "AAA", "observed_at": et(2025, 3, 4, 18)},
        {"symbol": "BBB", "observed_at": et(2025, 3, 4, 18)},
        {"symbol": "AAA", "observed_at": et(2025, 3, 5, 18)},
    ]

    Labeller(src).label(events)

    assert sorted(src.calls) == ["AAA", "BBB"]
