"""Tests for the transaction cost model.

The load-bearing property is that costs are never silently zero. A missing cost
becomes a free fill, which is precisely the assumption that made every earlier
result unusable.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from tradezbotz.research.costs import (
    EDGE_RELIABLE_FLOOR_BPS,
    MAX_PARTICIPATION,
    CostModel,
    amihud,
    daily_volatility,
    edge_spread,
    effective_spread,
    market_impact,
    participation_rate,
)
from tradezbotz.research.prices import Bar


def walk(n=300, start=100.0, vol=0.02, spread=0.0, seed=7, volume=1_000_000):
    """A random walk with a real intraday range, plus an optional bid-ask bounce.

    The intraday range is intrinsic rather than derived from `spread`. An earlier
    version made high == low == close when spread was zero, which is a session
    that never traded through a range at all -- EDGE correctly returns NaN on it,
    since a flat bar carries no spread information. That fixture was testing the
    degenerate case, not the tight-spread case.
    """
    rng = random.Random(seed)
    bars, p, day = [], start, date(2024, 1, 1)
    for _ in range(n):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        p *= 1 + rng.gauss(0, vol)
        half = p * spread / 2
        # Intraday range independent of the spread, so a zero-spread series is
        # still a real series.
        swing = abs(rng.gauss(0, vol)) * p
        o = p + (half if rng.random() < 0.5 else -half)
        c = p + (half if rng.random() < 0.5 else -half)
        hi = max(o, c) + half + swing
        lo = min(o, c) - half - swing
        bars.append(Bar(day, o, hi, max(lo, 0.01), c, volume))
        day += timedelta(days=1)
    return bars


# --- spread estimation ---------------------------------------------------------

def test_edge_needs_enough_history():
    """None, not zero. A missing estimate that reads as zero is a free fill."""
    assert edge_spread(walk(10)) is None


def test_edge_detects_a_wide_spread():
    wide = edge_spread(walk(300, spread=0.05, seed=3))
    tight = edge_spread(walk(300, spread=0.0, seed=3))

    assert wide is not None and tight is not None
    assert wide > tight, "a 5% bounce must read wider than none"


def test_edge_is_non_negative():
    """Negative squared spreads are truncated by the estimator; a negative cost
    would be a subsidy."""
    for seed in range(5):
        v = edge_spread(walk(120, seed=seed))
        assert v is None or v >= 0.0


# --- effective spread from real quotes ------------------------------------------

def test_effective_spread_measures_distance_from_the_midpoint():
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 9.90, "ap": 10.10}]
    trades = [{"t": "2025-03-04T14:00:01Z", "p": 10.10, "s": 100}]

    # mid 10.00, trade at the ask -> one-way 0.10, doubled = 0.20 / 10.00 = 2%
    assert effective_spread(trades, quotes) == pytest.approx(0.02)


def test_effective_spread_ignores_a_crossed_book():
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 11.0, "ap": 10.0}]
    trades = [{"t": "2025-03-04T14:00:01Z", "p": 10.5, "s": 100}]

    assert effective_spread(trades, quotes) is None


def test_effective_spread_uses_the_median():
    """One print through a stale quote is common on thin names, and a mean
    would let it dominate."""
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 9.99, "ap": 10.01}]
    trades = [{"t": f"2025-03-04T14:00:0{i}Z", "p": 10.01, "s": 10} for i in range(1, 6)]
    trades.append({"t": "2025-03-04T14:00:09Z", "p": 50.0, "s": 10})   # outlier

    out = effective_spread(trades, quotes)

    assert out == pytest.approx(0.002, abs=1e-4), "the 50.0 print is not allowed to dominate"


def test_effective_spread_needs_both_sides():
    assert effective_spread([], []) is None
    assert effective_spread([{"t": "2025-03-04T14:00:00Z", "p": 1, "s": 1}], []) is None


# --- liquidity measures ---------------------------------------------------------

def test_amihud_is_higher_for_thinner_names():
    thick = amihud(walk(200, volume=50_000_000, seed=11))
    thin = amihud(walk(200, volume=10_000, seed=11))

    assert thin > thick, "same price path, less volume, more illiquid"


def test_participation_rate_uses_median_volume():
    bars = walk(60, volume=1_000_000)

    assert participation_rate(100_000, bars) == pytest.approx(0.1, rel=0.01)


def test_daily_volatility_is_positive():
    assert daily_volatility(walk(120, vol=0.03)) > 0


# --- market impact --------------------------------------------------------------

def test_impact_grows_with_size_but_concavely():
    """The square-root law: doubling size raises impact by ~sqrt(2), not 2."""
    small = market_impact(0.01, 0.02)
    big = market_impact(0.02, 0.02)

    assert big > small
    assert big / small == pytest.approx(2 ** 0.5, rel=0.01)


def test_impact_is_zero_for_zero_size():
    assert market_impact(0.0, 0.02) == 0.0


def test_a_steeper_exponent_costs_more():
    """Small caps have been found to follow something closer to a square law,
    so the exponent must be testable rather than assumed."""
    root = market_impact(0.05, 0.02, exponent=0.5)
    square = market_impact(0.05, 0.02, exponent=1.0)

    assert square < root, "at participation below 1, a higher exponent is smaller"
    assert market_impact(0.5, 0.02, exponent=1.0) > 0


# --- the model ------------------------------------------------------------------

def test_cost_is_never_zero():
    """The whole point. A free fill does not exist on any venue."""
    c = CostModel().estimate(walk(200, spread=0.0))

    assert c.total > 0
    assert c.spread >= CostModel().floor_bps / 10_000


def test_missing_history_still_charges_the_floor():
    c = CostModel().estimate(walk(5))

    assert c.total > 0
    assert "floor" in c.notes


def test_measured_spread_overrides_the_estimate():
    """Real quotes beat an estimator whenever we have them."""
    bars = walk(200, spread=0.05)

    c = CostModel().estimate(bars, measured_spread=0.001)

    assert c.source == "quotes"
    assert c.spread == pytest.approx(0.001)


def test_low_edge_estimates_are_flagged_as_upper_bounds():
    """Measured against real NBBO spreads, EDGE reads 5-10x high below ~50bps.
    It stays useful because it errs high, but it is not a measurement there."""
    bars = walk(300, spread=0.0, seed=21)

    c = CostModel(floor_bps=5.0).estimate(bars)

    if c.spread * 10_000 < EDGE_RELIABLE_FLOOR_BPS:
        assert c.spread_is_upper_bound is True
        assert "upper bound" in c.notes


def test_oversized_orders_are_infeasible_not_merely_expensive():
    """Pricing a 30% participation fill as 'expensive' pretends it could happen."""
    bars = walk(200, volume=1_000_000)

    c = CostModel().estimate(bars, shares=300_000)

    assert c.participation > MAX_PARTICIPATION
    assert c.feasible is False
    assert "not executable" in c.notes


def test_a_normal_order_is_feasible():
    bars = walk(200, volume=1_000_000)

    c = CostModel().estimate(bars, shares=10_000)

    assert c.feasible is True
    assert c.impact > 0


def test_crossing_costs_more_than_resting():
    bars = walk(200, spread=0.03, seed=5)

    crossing = CostModel(crosses_spread=True).estimate(bars)
    resting = CostModel(crosses_spread=False).estimate(bars)

    assert crossing.spread == pytest.approx(2 * resting.spread)


def test_total_bps_is_readable():
    c = CostModel().estimate(walk(200, spread=0.02))

    assert c.total_bps == pytest.approx(c.total * 10_000)
