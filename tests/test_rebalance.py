"""Tests for the cross-sectional rebalance population.

Two things are being protected. The quantile arithmetic, which is easy to get
subtly wrong at the boundaries and produces plausible output when it is. And the
point-in-time rule, which here has an extra edge the event studies do not have:
a ranking depends on *every other company*, so one symbol reporting late can
contaminate a whole cohort rather than just its own row.
"""

from __future__ import annotations

from datetime import date

import pytest

from tradezbotz.research.rebalance import (
    MIN_COHORT,
    QUANTILES,
    Cohort,
    build_cohorts,
    month_starts,
    rank_cohort,
    to_events,
    universe_warning,
)

AS_OF = date(2025, 6, 1)


def values(n: int, start: float = 1.0) -> dict[str, float]:
    """n symbols with strictly increasing metric values, so rank is unambiguous."""
    return {f"S{i:03d}": start + i for i in range(n)}


# --- dates ------------------------------------------------------------------

def test_month_starts_covers_the_range_inclusively():
    out = month_starts(date(2025, 1, 1), date(2025, 4, 1))

    assert out == [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1),
                   date(2025, 4, 1)]


def test_month_starts_skips_a_partial_leading_month():
    out = month_starts(date(2025, 1, 15), date(2025, 3, 1))

    assert out == [date(2025, 2, 1), date(2025, 3, 1)]


def test_month_starts_rolls_the_year():
    out = month_starts(date(2024, 11, 1), date(2025, 2, 1))

    assert out[0] == date(2024, 11, 1)
    assert out[-1] == date(2025, 2, 1)
    assert len(out) == 4


# --- ranking ----------------------------------------------------------------

def test_the_cheapest_quintile_is_quantile_zero():
    ranked = rank_cohort(values(100), "ev_to_ebitda", AS_OF)

    cheapest = [r for r in ranked if r.quantile == 0]
    assert len(cheapest) == 20
    assert {r.symbol for r in cheapest} == {f"S{i:03d}" for i in range(20)}


def test_every_symbol_lands_in_a_valid_bucket():
    """Off-by-one at the top is the classic failure: index n maps to a bucket
    that does not exist, and with a naive implementation it does so silently."""
    for n in (MIN_COHORT, 51, 99, 100, 101, 137, 500):
        ranked = rank_cohort(values(n), "m", AS_OF)
        assert len(ranked) == n
        assert all(0 <= r.quantile < QUANTILES for r in ranked)


def test_buckets_are_within_one_of_even():
    ranked = rank_cohort(values(100), "m", AS_OF)
    sizes = [sum(1 for r in ranked if r.quantile == q) for q in range(QUANTILES)]

    assert max(sizes) - min(sizes) <= 1


def test_a_cohort_below_the_floor_is_not_ranked():
    """A quintile of a handful of names has a return that is noise about one or
    two of them."""
    assert rank_cohort(values(MIN_COHORT - 1), "m", AS_OF) == []


def test_non_positive_and_missing_values_are_excluded_not_ranked_cheapest():
    """The trap that ruins every naive value screen: a negative multiple sorts
    below every genuinely cheap company, so 'cheapest' fills with loss-makers."""
    vals = values(MIN_COHORT + 3)
    vals["NEG"] = -5.0
    vals["ZERO"] = 0.0
    vals["NONE"] = None

    ranked = rank_cohort(vals, "m", AS_OF)
    names = {r.symbol for r in ranked}

    assert "NEG" not in names
    assert "ZERO" not in names
    assert "NONE" not in names
    assert ranked[0].cohort_size == MIN_COHORT + 3


def test_higher_is_cheaper_inverts_the_order():
    ranked = rank_cohort(values(100), "yield", AS_OF, lower_is_cheaper=False)

    assert ranked[0].symbol == "S099"


# --- point-in-time ----------------------------------------------------------

class FakeSnap:
    def __init__(self, value):
        self.value = value

    def ev_to_ebitda(self, price):
        return self.value


def test_a_symbol_with_nothing_filed_yet_is_dropped_from_that_cohort():
    """Not carried forward from an earlier month. Carrying forward would mean a
    stale multiple ranked against fresh ones, which is a different and quieter
    error than a missing row."""
    seen = {}

    def snapshot_for(symbol, as_of):
        seen[symbol] = as_of
        # LATE has filed nothing by the rebalance date.
        return None if symbol == "LATE" else FakeSnap(10.0)

    symbols = [f"S{i:03d}" for i in range(MIN_COHORT)] + ["LATE"]
    cohorts = build_cohorts([AS_OF], symbols, snapshot_for,
                            lambda s, d: 10.0, metrics=("ev_to_ebitda",))

    names = {r.symbol for r in cohorts[0].rankings["ev_to_ebitda"]}
    assert "LATE" not in names
    assert seen["LATE"] == AS_OF, "the snapshot was asked for at the rebalance date"


def test_the_snapshot_is_requested_at_the_rebalance_date():
    """`build_cohorts` cannot verify point-in-time itself -- that lives in
    `fundamentals.visible` -- but it must at least hand down the right date."""
    asked = []

    def snapshot_for(symbol, as_of):
        asked.append(as_of)
        return FakeSnap(5.0)

    dates = [date(2025, 1, 1), date(2025, 2, 1)]
    build_cohorts(dates, [f"S{i}" for i in range(MIN_COHORT)],
                  snapshot_for, lambda s, d: 10.0, metrics=("ev_to_ebitda",))

    assert set(asked) == set(dates)


def test_a_symbol_with_no_price_is_dropped():
    cohorts = build_cohorts(
        [AS_OF], [f"S{i:03d}" for i in range(MIN_COHORT)],
        lambda s, d: FakeSnap(10.0), lambda s, d: None,
        metrics=("ev_to_ebitda",))

    assert cohorts == []


# --- the engine handoff -----------------------------------------------------

def test_to_events_emits_one_row_per_symbol_and_date():
    ranked = rank_cohort(values(100), "ev_to_ebitda", AS_OF)
    events, payloads = to_events([Cohort(AS_OF, {"ev_to_ebitda": ranked})])

    assert len(events) == len(payloads) == 100
    assert all(e["observed_at"].startswith(AS_OF.isoformat()) for e in events)


def test_events_and_payloads_stay_aligned():
    """The labeller aligns by index. A misalignment here would pair every
    payload with another symbol's returns -- the bug that once made every
    selector produce identical results."""
    ranked = rank_cohort(values(80), "ev_to_ebitda", AS_OF)
    events, payloads = to_events([Cohort(AS_OF, {"ev_to_ebitda": ranked})])

    for event, payload in zip(events, payloads):
        assert event["symbol"] == payload["symbol"]


def test_the_payload_carries_the_flag_a_selector_reads():
    ranked = rank_cohort(values(100), "ev_to_ebitda", AS_OF)
    _, payloads = to_events([Cohort(AS_OF, {"ev_to_ebitda": ranked})])

    cheapest = [p for p in payloads if p["cheapest_ev_to_ebitda"]]
    assert len(cheapest) == 20
    assert all(p["ev_to_ebitda_quantile"] == 0 for p in cheapest)


def test_multiple_metrics_merge_into_one_row_per_symbol():
    a = rank_cohort(values(100), "ev_to_ebitda", AS_OF)
    b = rank_cohort(values(100, start=50.0), "price_to_sales", AS_OF)

    _, payloads = to_events([Cohort(AS_OF, {"ev_to_ebitda": a,
                                            "price_to_sales": b})])

    assert len(payloads) == 100
    assert "cheapest_ev_to_ebitda" in payloads[0]
    assert "cheapest_price_to_sales" in payloads[0]


# --- survivorship -----------------------------------------------------------

def test_an_all_alive_cohort_is_flagged():
    """A universe assembled from a price cache contains companies somebody
    fetched. If none of the oldest cohort ever delisted, the cohort has already
    selected for survival and every return measured on it is biased upward."""
    ranked = rank_cohort(values(100), "m", date(2016, 1, 1))
    text = universe_warning([Cohort(date(2016, 1, 1), {"m": ranked})],
                            lambda s: True)

    assert "WARNING" in text
    assert "selected for survival" in text


def test_a_cohort_with_real_attrition_is_not_flagged():
    ranked = rank_cohort(values(100), "m", date(2016, 1, 1))
    text = universe_warning([Cohort(date(2016, 1, 1), {"m": ranked})],
                            lambda s: int(s[1:]) % 4 != 0)

    assert "WARNING" not in text
    assert "75%" in text


def test_the_warning_reads_the_oldest_cohort_not_the_newest():
    """Attrition is cumulative, so the newest cohort always looks healthy and
    tells you nothing."""
    old = Cohort(date(2016, 1, 1),
                 {"m": rank_cohort(values(100), "m", date(2016, 1, 1))})
    new = Cohort(date(2025, 1, 1),
                 {"m": rank_cohort(values(100), "m", date(2025, 1, 1))})

    assert "2016-01-01" in universe_warning([new, old], lambda s: True)
