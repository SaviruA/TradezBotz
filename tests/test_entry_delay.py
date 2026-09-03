"""Tests for the entry-delay diagnostic.

Conrad, Gultekin & Kaul (1997) found short-term reversal profits are
PREDOMINANTLY driven by bid-ask bounce. Our first surviving candidate was
`buy + rsi_oversold` -- buy after a fall, in illiquid microcaps, measured to a
close -- which is precisely that setup. An oversold name's opening print sits
toward the bid, and its drift back to mid is booked as profit.

Skipping a session between signal and entry is the standard separation: a real
information effect survives it, a bounce artefact does not. These tests fix the
mechanics of the skip, because a delay that silently clamps, shifts horizons,
or truncates the fetch window would answer a different question than the one
asked and still look like a clean result.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.labeler import Coverage, Labeller, label_event
from tradezbotz.research.prices import Bar, Series


def _series(symbol="AAA", n=40, start=date(2020, 1, 1), closes=None):
    bars = []
    for i in range(n):
        close = closes[i] if closes else 100.0 + i
        bars.append(Bar(day=start + timedelta(days=i), open=close, high=close + 1,
                        low=close - 1, close=close, volume=10_000))
    return Series(symbol=symbol, bars=tuple(bars), requested_start=start,
                  requested_end=start + timedelta(days=n - 1))


BEFORE_OPEN = datetime(2020, 1, 1, 5, 0, tzinfo=UTC)


# --- the skip does what it says ---------------------------------------------

def test_a_delay_moves_the_entry_forward_by_whole_sessions():
    s = _series()

    base = label_event(s, BEFORE_OPEN, horizons=[5])
    skipped = label_event(s, BEFORE_OPEN, horizons=[5], entry_delay=1)

    assert base.entry_day == date(2020, 1, 1)
    assert skipped.entry_day == date(2020, 1, 2)
    assert skipped.entry_price == base.entry_price + 1


def test_zero_delay_is_byte_for_byte_the_undelayed_label():
    """The diagnostic must not perturb the primary measurement."""
    s = _series()

    assert label_event(s, BEFORE_OPEN, horizons=[1, 5, 20], entry_delay=0) == \
        label_event(s, BEFORE_OPEN, horizons=[1, 5, 20])


def test_the_horizon_is_measured_from_the_delayed_entry_not_the_original():
    """Holding five sessions from the skipped entry, not four. Measuring to the
    original exit would shorten the horizon and confound the two effects."""
    s = _series()

    skipped = label_event(s, BEFORE_OPEN, horizons=[5], entry_delay=2)

    # entry on day index 2 (open 102), exit five sessions later (close 107)
    assert skipped.entry_price == 102.0
    assert skipped.returns[5] == pytest.approx(107.0 / 102.0 - 1.0)


# --- the failure modes that would fake a clean answer -----------------------

def test_a_delay_past_the_last_bar_is_no_entry_bar_not_a_clamp():
    """Clamping to the final bar would enter on a different session from the
    one asked for, and the delayed run would stop being comparable to the
    undelayed one -- which is the whole point of running it."""
    s = _series(n=3)

    out = label_event(s, BEFORE_OPEN, horizons=[1], entry_delay=10)

    assert out.coverage is Coverage.NO_ENTRY_BAR
    assert out.entry_day is None
    assert out.entry_price is None


def test_a_delay_that_runs_the_horizon_off_the_end_is_partial_not_complete():
    """Otherwise the skip silently drops the losers that ran out of bars."""
    s = _series(n=10)

    out = label_event(s, BEFORE_OPEN, horizons=[1, 20], entry_delay=5)

    assert out.coverage is not Coverage.COMPLETE
    assert 20 not in out.returns


def test_the_bounce_artefact_is_what_the_skip_removes():
    """A one-session rebound followed by flat prices: profitable entering at
    the depressed open, worthless a session later. This is the shape the
    diagnostic exists to expose."""
    closes = [100.0] + [110.0] * 39  # one jump, then nothing
    s = _series(closes=closes)

    base = label_event(s, BEFORE_OPEN, horizons=[5])
    skipped = label_event(s, BEFORE_OPEN, horizons=[5], entry_delay=1)

    assert base.returns[5] == pytest.approx(0.10)
    assert skipped.returns[5] == pytest.approx(0.0)


def test_a_genuine_drift_survives_the_skip():
    """The other half of the diagnostic: a signal carrying information keeps
    most of its edge, so the skip separates rather than simply deflating."""
    s = _series()  # +1 a session, indefinitely

    base = label_event(s, BEFORE_OPEN, horizons=[20])
    skipped = label_event(s, BEFORE_OPEN, horizons=[20], entry_delay=1)

    assert skipped.returns[20] > 0.8 * base.returns[20]


# --- through the Labeller ---------------------------------------------------

class _Source:
    def __init__(self, series):
        self.series = series
        self.requested_end = None

    def daily_bars(self, symbol, start, end):
        self.requested_end = end
        return self.series


def test_the_labeller_threads_the_delay_through():
    src = _Source(_series())
    events = [{"symbol": "AAA", "observed_at": BEFORE_OPEN.isoformat()}]

    base = Labeller(src, horizons=[5]).label(events)[0]
    skipped = Labeller(src, horizons=[5], entry_delay=1).label(events)[0]

    assert skipped.entry_day > base.entry_day


def test_the_fetch_window_is_padded_for_the_delay():
    """A delayed entry needs bars further out. Padding only for the horizon
    would truncate the longest holding period on the delayed run and report it
    as PARTIAL -- an artefact of the fetch, read as an artefact of the signal."""
    src = _Source(_series())
    events = [{"symbol": "AAA", "observed_at": BEFORE_OPEN.isoformat()}]

    Labeller(src, horizons=[20]).label(events)
    undelayed_end = src.requested_end
    Labeller(src, horizons=[20], entry_delay=5).label(events)

    assert src.requested_end > undelayed_end
