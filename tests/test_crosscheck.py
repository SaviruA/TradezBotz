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
