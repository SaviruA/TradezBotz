"""Tests for two-source price comparison."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradezbotz.research.crosscheck import (
    Agreement,
    compare,
    summarise,
)
from tradezbotz.research.prices import Bar, Series


def series(symbol="TEST", closes=None, start=date(2025, 3, 3)):
    closes = closes or [100.0] * 10
    bars, day = [], start
    for c in closes:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        bars.append(Bar(day, c, c * 1.01, c * 0.99, c, 1e6))
        day += timedelta(days=1)
    return Series(symbol, tuple(bars), start, bars[-1].day)


def test_matching_sources_agree():
    a = series(closes=[100.0] * 10)
    b = series(closes=[100.0] * 10)

    assert compare(a, b).verdict is Agreement.AGREE


def test_tick_level_noise_still_agrees():
    """Rounding and minor adjustment differences must not read as disagreement."""
    a = series(closes=[100.00] * 10)
    b = series(closes=[100.02] * 10)  # 0.02%

    assert compare(a, b).verdict is Agreement.AGREE


def test_moderate_divergence_is_suspect():
    a = series(closes=[100.0] * 10)
    b = series(closes=[101.0] * 10)  # 1%

    assert compare(a, b).verdict is Agreement.SUSPECT


def test_ghost_prices_are_flagged_as_disagreement():
    """The documented IEX failure: a small cap printing at 4.85 elsewhere while
    IEX shows 4.20 -- about 13% off."""
    a = series(closes=[4.85] * 10)
    b = series(closes=[4.20] * 10)

    result = compare(a, b)

    assert result.verdict is Agreement.DISAGREE
    assert result.median_rel_diff > 0.10
    assert result.trustworthy is False


def test_thin_overlap_reports_insufficient_rather_than_agreement():
    """Two sources that barely overlap have not verified anything."""
    a = series(closes=[100.0] * 3)
    b = series(closes=[100.0] * 3)

    result = compare(a, b)

    assert result.verdict is Agreement.INSUFFICIENT_OVERLAP
    assert result.trustworthy is False, "unverified is not the same as verified-good"


def test_missing_days_are_not_counted_as_disagreement():
    """A coverage gap is a different problem from a price conflict, and
    coverage_report already tracks gaps."""
    a = series(closes=[100.0] * 10)
    b = Series("TEST", a.bars[:6], a.requested_start, a.requested_end)

    result = compare(a, b)

    assert result.overlapping_days == 6
    assert result.verdict is Agreement.AGREE


def test_no_overlap_at_all():
    a = series(start=date(2025, 3, 3))
    b = series(start=date(2026, 3, 3))

    assert compare(a, b).verdict is Agreement.INSUFFICIENT_OVERLAP


def test_max_diff_surfaces_a_single_bad_bar():
    closes = [100.0] * 10
    other = list(closes)
    other[4] = 130.0

    result = compare(series(closes=closes), series(closes=other))

    assert result.max_rel_diff == pytest.approx(0.30)
    assert result.verdict is Agreement.AGREE, "one bad bar should not condemn the series"


# --- summary -----------------------------------------------------------------

def test_summarise_reports_the_trustworthy_share():
    results = [
        compare(series(closes=[100.0] * 10), series(closes=[100.0] * 10)),
        compare(series(closes=[100.0] * 10), series(closes=[100.0] * 10)),
        compare(series(closes=[4.85] * 10), series(closes=[4.20] * 10)),
        compare(series(closes=[100.0] * 3), series(closes=[100.0] * 3)),
    ]

    s = summarise(results)

    assert s["total"] == 4
    assert s["agree"] == 2
    assert s["disagree"] == 1
    assert s["insufficient_overlap"] == 1
    assert s["trustworthy_rate"] == pytest.approx(0.5)


def test_summarise_handles_empty():
    assert summarise([]) == {"total": 0}


# --- high/low agreement ------------------------------------------------------

def hl_series(symbol="TEST", n=10, close=100.0, high=101.0, low=99.0,
              start=date(2025, 3, 3)):
    bars, day = [], start
    for _ in range(n):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        bars.append(Bar(day, close, high, low, close, 1e6))
        day += timedelta(days=1)
    return Series(symbol, tuple(bars), start, bars[-1].day)


def test_matching_highs_and_lows_are_range_trustworthy():
    d = compare(hl_series(), hl_series())

    assert d.range_trustworthy is True
    assert d.median_high_diff == pytest.approx(0.0)


def test_phantom_high_breaks_range_trust_while_closes_still_agree():
    """The failure Concretum names: a source can agree on closes and still carry
    a bad high that manufactures a Donchian breakout or trips a stop."""
    a = hl_series(high=101.0)
    b = hl_series(high=140.0)

    d = compare(a, b)

    assert d.verdict is Agreement.AGREE, "closes are identical"
    assert d.range_trustworthy is False, "but highs are not"


def test_bad_low_also_breaks_range_trust():
    d = compare(hl_series(low=99.0), hl_series(low=60.0))

    assert d.range_trustworthy is False


def test_range_trust_is_false_without_overlap():
    """Unverified is not verified-good, for ranges as for closes."""
    a = hl_series(n=3)
    b = hl_series(n=3)

    assert compare(a, b).range_trustworthy is False


def test_summarise_reports_range_trust_separately():
    good = compare(hl_series(), hl_series())
    bad = compare(hl_series(high=101.0), hl_series(high=140.0))

    s = summarise([good, bad])

    assert s["range_trustworthy"] == 1
    assert s["agree"] == 2, "both agree on closes"
