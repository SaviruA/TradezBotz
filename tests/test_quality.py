"""Tests for bar-level data validation."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradezbotz.research.prices import Bar, Series
from tradezbotz.research.quality import (
    Issue,
    compare_coverage,
    expected_sessions,
    modified_zscores,
    summarise,
    validate,
)


def series(bars, symbol="TEST"):
    return Series(symbol, tuple(bars), bars[0].day, bars[-1].day) if bars else \
        Series(symbol, (), date(2025, 1, 1), date(2025, 1, 2))


def clean_bars(n=40, start=date(2025, 3, 3), price=100.0, seed=7):
    """A plausible price path: small pseudo-random walk, no repeating pattern.

    A sawtooth fixture (i % 5) looks harmless but makes the return series almost
    constant, which drives MAD to zero and turns every ordinary step into an
    apparent outlier. Test data has to be as realistic as the detector is
    sensitive.
    """
    import random

    rng = random.Random(seed)
    out, day, p = [], start, price
    while len(out) < n:
        if day.weekday() < 5:
            p *= 1 + rng.gauss(0, 0.008)
            hi, lo = p * (1 + abs(rng.gauss(0, 0.004))), p * (1 - abs(rng.gauss(0, 0.004)))
            close = min(max(p * (1 + rng.gauss(0, 0.003)), lo), hi)
            out.append(Bar(day, p, hi, lo, close, 1_000_000 + rng.randint(0, 50_000)))
        day += timedelta(days=1)
    return out


# --- modified Z-score --------------------------------------------------------

def test_modified_zscore_flags_an_outlier():
    vals = [10.0] * 20 + [80.0]

    assert abs(modified_zscores(vals)[-1]) > 3.5


def test_modified_zscore_is_robust_where_stdev_is_not():
    """An outlier inflates the standard deviation used to detect it. MAD does
    not, which is why the cleaning literature uses it."""
    import statistics
    vals = [10.0, 10.1, 9.9, 10.2, 9.8, 200.0]

    sd_z = abs(vals[-1] - statistics.fmean(vals)) / statistics.pstdev(vals)
    mad_z = abs(modified_zscores(vals)[-1])

    assert sd_z < 2.5, "standard deviation is contaminated by the outlier"
    assert mad_z > 3.5, "MAD still catches it"


def test_constant_window_yields_no_scores():
    assert modified_zscores([5.0] * 10) == [0.0] * 10


def test_short_input_is_safe():
    assert modified_zscores([1.0, 2.0]) == [0.0, 0.0]


# --- OHLC validity -----------------------------------------------------------

def test_clean_series_is_clean():
    rep = validate(series(clean_bars()))

    assert rep.clean_rate == 1.0
    assert rep.issues == ()


def test_high_below_low_is_invalid():
    bars = clean_bars()
    bars[10] = Bar(bars[10].day, 100, 95.0, 105.0, 100, 1e6)

    rep = validate(series(bars))

    assert any(i.issue is Issue.INVALID_OHLC for i in rep.issues)


def test_close_outside_range_is_invalid():
    bars = clean_bars()
    bars[5] = Bar(bars[5].day, 100, 101, 99, 150.0, 1e6)

    assert any(i.issue is Issue.INVALID_OHLC for i in validate(series(bars)).issues)


def test_non_positive_price_is_invalid():
    bars = clean_bars()
    bars[7] = Bar(bars[7].day, 0.0, 0.0, 0.0, 0.0, 1e6)

    assert any(i.issue is Issue.INVALID_OHLC for i in validate(series(bars)).issues)


# --- stale bars --------------------------------------------------------------

def test_flat_bar_is_stale():
    """o==h==l==c with negligible volume is a carried-forward price, not a
    traded one. It depresses measured volatility, widening Bollinger signals
    and shrinking ATR stops."""
    bars = clean_bars()
    d = bars[12].day
    bars[12] = Bar(d, 9.95, 9.95, 9.95, 9.95, 144.0)

    rep = validate(series(bars))

    assert any(i.issue is Issue.STALE for i in rep.issues)


def test_zero_volume_is_flagged():
    bars = clean_bars()
    b = bars[8]
    bars[8] = Bar(b.day, b.open, b.high, b.low, b.close, 0.0)

    assert any(i.issue is Issue.ZERO_VOLUME for i in validate(series(bars)).issues)


# --- spikes ------------------------------------------------------------------

def test_phantom_high_is_flagged():
    """The failure that triggers false Donchian breakouts and false stops --
    and nothing was checking highs or lows before this."""
    bars = clean_bars()
    b = bars[20]
    bars[20] = Bar(b.day, b.open, b.high * 4, b.low, b.close, b.volume)

    rep = validate(series(bars))

    assert any(i.issue is Issue.SPIKE_HIGH and i.day == b.day for i in rep.issues)


def test_phantom_low_is_flagged():
    bars = clean_bars()
    b = bars[18]
    bars[18] = Bar(b.day, b.open, b.high, b.low * 0.2, b.close, b.volume)

    rep = validate(series(bars))

    assert any(i.issue is Issue.SPIKE_LOW and i.day == b.day for i in rep.issues)


def test_ordinary_volatility_is_not_flagged():
    """A wide but plausible range must not be called a spike, or the filter
    removes exactly the moves a strategy exists to trade."""
    bars = clean_bars()
    for i in range(len(bars)):
        b = bars[i]
        bars[i] = Bar(b.day, b.open, b.open + 3.0, b.open - 3.0, b.close, b.volume)

    rep = validate(series(bars))

    assert not any(i.issue in (Issue.SPIKE_HIGH, Issue.SPIKE_LOW) for i in rep.issues)


# --- missing sessions --------------------------------------------------------

def test_missing_weekday_is_detected():
    """Alpaca emits no bar when an interval has no trades or only one trade, so
    thin names lose whole sessions and windows silently misalign."""
    bars = clean_bars(20)
    gone = bars.pop(10).day

    assert gone in expected_sessions(bars)


def test_weekends_are_not_missing_sessions():
    assert all(d.weekday() < 5 for d in expected_sessions(clean_bars(30)))


def test_no_bars_no_sessions():
    assert expected_sessions([]) == []


# --- coverage comparison -----------------------------------------------------

def test_compare_coverage_counts_both_directions():
    a = clean_bars(20)
    b = [x for i, x in enumerate(a) if i != 5]

    out = compare_coverage(series(a), series(b))

    assert out["primary_bars"] == 20
    assert out["secondary_bars"] == 19
    assert out["primary_only"] == 1
    assert out["secondary_only"] == 0
    assert out["shared"] == 19


# --- reporting ---------------------------------------------------------------

def test_empty_series_reports_nothing():
    rep = validate(Series("X", (), date(2025, 1, 1), date(2025, 2, 1)))

    assert rep.n_bars == 0 and rep.clean_rate == 0.0


def test_summarise_aggregates():
    good = validate(series(clean_bars()))
    bars = clean_bars()
    bars[9] = Bar(bars[9].day, 9.95, 9.95, 9.95, 9.95, 0.0)
    bad = validate(series(bars, "BAD"))

    s = summarise([good, bad])

    assert s["symbols"] == 2
    assert s["stale"] >= 1
    assert 0.0 < s["clean_rate"] <= 1.0


def test_summarise_handles_empty():
    assert summarise([]) == {"symbols": 0}


def test_report_summary_is_readable():
    bars = clean_bars()
    bars[9] = Bar(bars[9].day, 9.95, 9.95, 9.95, 9.95, 0.0)

    assert "stale" in validate(series(bars)).summary()


def test_mad_zero_fallback_still_detects_outliers():
    """Regression: when >50% of values are identical, MAD is exactly zero and a
    naive implementation goes blind -- on precisely the case that matters, a
    flat thinly-traded series with one genuine spike."""
    vals = [10.0] * 20 + [80.0]

    assert abs(modified_zscores(vals)[-1]) > 3.5


def test_truly_constant_window_yields_no_scores():
    """Only when the mean absolute deviation is also zero is the window really
    constant, and then no deviation is meaningful."""
    assert modified_zscores([5.0] * 10) == [0.0] * 10
