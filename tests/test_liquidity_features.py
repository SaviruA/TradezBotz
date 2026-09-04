"""Tests for the liquidity features and the cuts built on them.

82 of 232 verdicts in the 5.5-year sweep were "costs exceed edge". That is not
a precision problem and no amount of extra data moves it: we pay a ~93bp median
round trip BECAUSE the universe is microcaps. The published insider result says
the same -- abnormal returns "vanish and even become negative when limiting the
tradable dollar amount to a reasonable size", being "negatively correlated with
stock liquidity".

Until these two fields existed there was no way to express "the same signal, in
names where a round trip does not eat it".
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradezbotz.research.candidates import (
    LIQUID_DOLLAR_VOLUME,
    MIN_TRADEABLE_PRICE,
    liquidity_candidates,
    insider_class_candidates,
)
from tradezbotz.research.features import features_at
from tradezbotz.research.prices import Bar


def _bars(n=80, close=10.0, volume=100_000):
    return [Bar(day=date(2020, 1, 1) + timedelta(days=i), open=close,
                high=close * 1.01, low=close * 0.99, close=close,
                volume=volume) for i in range(n)]


def _payload(**kw):
    base = {"transaction_code": "P", "acquired_disposed": "A"}
    base.update(kw)
    return base


def _select(name, candidates):
    return next(c for c in candidates if c.name == name).selector


# --- the features -----------------------------------------------------------

def test_dollar_volume_is_price_times_volume_over_the_prior_window():
    out = features_at(_bars(close=10.0, volume=100_000))

    assert out["dollar_volume_20d"] == pytest.approx(1_000_000.0)


def test_entry_close_is_the_last_bar_not_the_first():
    bars = _bars(close=10.0)
    bars[-1] = Bar(day=bars[-1].day, open=10.0, high=10.0, low=10.0,
                   close=42.0, volume=100_000)

    assert features_at(bars)["entry_close"] == 42.0


def test_a_zero_volume_window_is_zero_not_a_crash():
    """A halted or untraded name must report zero liquidity, which the cuts
    then exclude -- rather than dividing by nothing."""
    out = features_at(_bars(volume=0))

    assert out["dollar_volume_20d"] == 0.0


def test_the_window_is_the_prior_twenty_sessions():
    """Longer history must not dilute the reading with stale volume."""
    bars = _bars(n=80, volume=1_000)
    for i in range(60, 80):
        bars[i] = Bar(day=bars[i].day, open=10.0, high=10.0, low=10.0,
                      close=10.0, volume=1_000_000)

    out = features_at(bars)

    assert out["dollar_volume_20d"] == pytest.approx(10_000_000.0)


# --- the cuts ---------------------------------------------------------------

def test_a_liquid_name_passes_and_an_illiquid_one_does_not():
    sel = _select("buy + liquid", liquidity_candidates())

    assert sel(_payload(dollar_volume_20d=LIQUID_DOLLAR_VOLUME + 1), None)
    assert not sel(_payload(dollar_volume_20d=LIQUID_DOLLAR_VOLUME - 1), None)


def test_a_missing_liquidity_reading_fails_the_cut_rather_than_passing_it():
    """An event whose features could not be computed did not meet the
    condition. Treating unknown as met would fabricate trades in exactly the
    names we cannot price."""
    sel = _select("buy + liquid", liquidity_candidates())

    assert not sel(_payload(), None)


def test_the_price_floor_excludes_sub_three_dollar_names():
    sel = _select("buy + liquid + above $3", liquidity_candidates())
    liquid = {"dollar_volume_20d": LIQUID_DOLLAR_VOLUME * 2}

    assert sel(_payload(entry_close=MIN_TRADEABLE_PRICE, **liquid), None)
    assert not sel(_payload(entry_close=MIN_TRADEABLE_PRICE - 0.01, **liquid), None)


def test_the_illiquid_complement_exists_so_both_sides_are_visible():
    """If the liquid cut wins only because its complement loses, that is a cost
    story rather than an alpha story, and the report must show both."""
    sel = _select("buy + illiquid", liquidity_candidates())

    assert sel(_payload(dollar_volume_20d=LIQUID_DOLLAR_VOLUME - 1), None)
    assert not sel(_payload(dollar_volume_20d=LIQUID_DOLLAR_VOLUME + 1), None)
    # A name with no reading is not "illiquid", it is unmeasured.
    assert not sel(_payload(dollar_volume_20d=0), None)


def test_every_liquidity_cut_still_requires_an_open_market_buy():
    """These are refinements of the insider signal, not standalone screens."""
    sell = {"transaction_code": "S", "acquired_disposed": "D",
            "dollar_volume_20d": LIQUID_DOLLAR_VOLUME * 2, "entry_close": 50.0,
            "is_opportunistic": True}

    for cand in liquidity_candidates():
        assert not cand.selector(sell, None), cand.name


# --- the split, and its control ---------------------------------------------

def test_the_routine_control_is_present_alongside_the_opportunistic_cut():
    """A split that only ever reports its good half is not a test. If routine
    pays like opportunistic, the split carries nothing here."""
    names = {c.name for c in insider_class_candidates()}

    assert "opportunistic buy" in names
    assert "routine buy" in names


def test_the_two_classes_are_mutually_exclusive_selectors():
    opp = _select("opportunistic buy", insider_class_candidates())
    rou = _select("routine buy", insider_class_candidates())
    payload = _payload(is_opportunistic=True, is_routine=False)

    assert opp(payload, None)
    assert not rou(payload, None)


def test_an_unknown_classification_matches_neither_class():
    """UNKNOWN is a third state, and letting it fall into either bucket would
    make the population depend on how much history we happen to hold."""
    opp = _select("opportunistic buy", insider_class_candidates())
    rou = _select("routine buy", insider_class_candidates())

    assert not opp(_payload(insider_class="unknown"), None)
    assert not rou(_payload(insider_class="unknown"), None)


def test_the_combined_cut_requires_both_refinements():
    sel = _select("opportunistic buy + liquid", liquidity_candidates())
    liquid = {"dollar_volume_20d": LIQUID_DOLLAR_VOLUME * 2}

    assert sel(_payload(is_opportunistic=True, **liquid), None)
    assert not sel(_payload(is_opportunistic=False, **liquid), None)
    assert not sel(_payload(is_opportunistic=True, dollar_volume_20d=1.0), None)


def test_thresholds_are_the_published_convention_not_a_fitted_number():
    """A fitted threshold is a search, and a search that does not register its
    trials is the exact thing this apparatus exists to prevent."""
    assert MIN_TRADEABLE_PRICE == 3.0
    assert LIQUID_DOLLAR_VOLUME == 5_000_000.0
