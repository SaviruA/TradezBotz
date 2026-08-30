"""Tests for three-way price adjudication.

The point of a third source is to convert "one of you is wrong" into "this one
is wrong". These tests pin the majority rule, and in particular that proximity
never substitutes for a majority -- a source that is merely less wrong has not
been vindicated by anything.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradezbotz.research.crosscheck import (
    Adjudication,
    Verdict,
    adjudicate,
    summarise_adjudications,
)
from tradezbotz.research.prices import Bar, Series


def series(symbol: str, closes, start=date(2025, 3, 3)):
    bars, day = [], start
    for c in closes:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        bars.append(Bar(day, c, c * 1.01, c * 0.99, c, 1_000_000))
        day += timedelta(days=1)
    return Series(symbol=symbol, bars=tuple(bars),
                  requested_start=start, requested_end=day)


BASE = [10.0 + i * 0.1 for i in range(20)]


def scaled(factor):
    return [c * factor for c in BASE]


def test_all_three_agreeing():
    a = adjudicate(series("T", BASE), series("T", BASE), series("T", BASE))

    assert a.verdict is Verdict.ALL_AGREE
    assert a.trustworthy_source == "both"


def test_primary_is_the_outlier():
    """The XELB-inverted case: secondary and referee agree, primary does not."""
    a = adjudicate(series("T", scaled(3.0)), series("T", BASE), series("T", BASE))

    assert a.verdict is Verdict.PRIMARY_OUTLIER
    assert a.trustworthy_source == "secondary"


def test_secondary_is_the_outlier():
    """The real XELB case: Massive 7.03 and Yahoo 7.027 agree, Alpaca says 21.11."""
    a = adjudicate(series("XELB", BASE), series("XELB", scaled(3.004)),
                   series("XELB", BASE))

    assert a.verdict is Verdict.SECONDARY_OUTLIER
    assert a.trustworthy_source == "primary"


def test_referee_outlier_yields_no_usable_verdict():
    """If the two vendors agree and only the referee differs, nothing about the
    vendors has been learned -- and we do not conclude the referee is 'wrong'."""
    a = adjudicate(series("T", BASE), series("T", BASE), series("T", scaled(2.0)))

    assert a.verdict is Verdict.REFEREE_OUTLIER
    assert a.trustworthy_source is None


def test_three_way_disagreement_gives_no_verdict():
    """The BDX shape: Massive 240.97, Alpaca 181.20, Yahoo 189.44. Yahoo is
    nearer Alpaca, but nearer is not a majority and must not be reported as one."""
    a = adjudicate(series("BDX", scaled(1.33)), series("BDX", BASE),
                   series("BDX", scaled(1.045)))

    assert a.verdict is Verdict.NO_MAJORITY
    assert a.trustworthy_source is None


def test_proximity_does_not_win_a_vote():
    """Explicitly: being closer than the third source is not being vindicated."""
    a = adjudicate(series("T", scaled(1.10)), series("T", BASE),
                   series("T", scaled(1.06)))

    assert a.verdict is not Verdict.SECONDARY_OUTLIER
    assert a.trustworthy_source is None


def test_insufficient_overlap_is_not_a_verdict():
    short = series("T", [10.0, 10.1])

    a = adjudicate(short, short, short)

    assert a.verdict is Verdict.INSUFFICIENT_OVERLAP
    assert a.trustworthy_source is None


def test_overlap_is_the_smallest_pairwise_overlap():
    """A referee covering only part of the window must not be reported as if it
    covered all of it."""
    full = series("T", BASE)
    partial = Series(symbol="T", bars=full.bars[:8],
                     requested_start=full.requested_start,
                     requested_end=full.requested_end)

    a = adjudicate(full, full, partial)

    assert a.overlapping_days == 8


def test_summary_counts_what_was_resolved():
    results = [
        Adjudication("A", 0.0, 0.0, 0.0, Verdict.ALL_AGREE, 20),
        Adjudication("B", 0.5, 0.5, 0.0, Verdict.PRIMARY_OUTLIER, 20),
        Adjudication("C", 0.5, 0.0, 0.5, Verdict.SECONDARY_OUTLIER, 20),
        Adjudication("D", 0.5, 0.5, 0.5, Verdict.NO_MAJORITY, 20),
    ]

    out = summarise_adjudications(results)

    assert out["resolved"] == 2, "only the two with a majority"
    assert out["usable"] == 3, "resolved plus the unanimous one"
    assert out["no_majority"] == 1


def test_summary_of_nothing():
    assert summarise_adjudications([]) == {"total": 0}


# --- total-return vs price-only ------------------------------------------------

def dividend_adjusted(closes, yield_total=0.20):
    """A total-return series: marked down in the past by dividends paid since,
    converging to the price-only series at the present."""
    n = len(closes)
    return [c * (1 - yield_total * (n - 1 - i) / (n - 1)) for i, c in enumerate(closes)]


def test_dividend_basis_is_not_called_an_error():
    """The real case: 20 of 24 disputed symbols were this. Alpaca sat below
    Massive and Yahoo by exactly the accumulated distribution -- ARI 5.663 against
    10.430 with 5.50 of dividends. Calling that 'wrong' retires a good source
    over a definitional difference."""
    price_only = series("ARI", BASE * 3)
    total_return = series("ARI", dividend_adjusted(BASE * 3))

    a = adjudicate(price_only, total_return, series("ARI", BASE * 3))

    assert a.verdict is Verdict.ADJUSTMENT_BASIS
    assert a.trustworthy_source == "both", "neither is wrong"


def test_a_constant_ratio_error_is_still_an_error():
    """XELB: a clean 3.004 throughout that never converges. Unlike a dividend
    basis this is a genuine split-adjustment fault and must stay flagged."""
    a = adjudicate(series("XELB", BASE * 3), series("XELB", scaled(3.004) * 3),
                   series("XELB", BASE * 3))

    assert a.verdict is Verdict.SECONDARY_OUTLIER
    assert a.trustworthy_source == "primary"


def test_dividend_signature_requires_one_sidedness():
    """A series that is sometimes above and sometimes below is not a total-return
    basis, whatever its magnitude."""
    noisy = [c * (1.1 if i % 2 else 0.9) for i, c in enumerate(BASE * 3)]

    a = adjudicate(series("T", BASE * 3), series("T", noisy), series("T", BASE * 3))

    assert a.verdict is not Verdict.ADJUSTMENT_BASIS


def test_dividend_signature_requires_convergence():
    """Consistently below but never converging is a scale error, not dividends."""
    below = [c * 0.8 for c in BASE * 3]

    a = adjudicate(series("T", BASE * 3), series("T", below), series("T", BASE * 3))

    assert a.verdict is Verdict.SECONDARY_OUTLIER


def test_summary_separates_benign_from_resolved():
    results = [
        Adjudication("A", 0.0, 0.0, 0.0, Verdict.ALL_AGREE, 20),
        Adjudication("B", 0.2, 0.0, 0.2, Verdict.ADJUSTMENT_BASIS, 20),
        Adjudication("C", 0.5, 0.0, 0.5, Verdict.SECONDARY_OUTLIER, 20),
    ]

    out = summarise_adjudications(results)

    assert out["adjustment_basis"] == 1
    assert out["resolved"] == 1, "only genuine faults count as resolved"
    assert out["usable"] == 3
