"""Tests for how `measure` chooses which events to measure.

The choice is not housekeeping. A contiguous slice of a chronological
partition is a sample of ONE REGIME, and an edge measured inside a single
regime is indistinguishable from a description of it. Two runs made the point:
the head of the partition charged a fallback cost constant on 99.3% of trades
(no prior bars to price a spread from), and the tail fixed that but cut a
2,108-day partition to its last 52 days.

So there are two separate jobs here, and conflating them is what went wrong
both times: drop what genuinely cannot be priced (the history floor), then
sample what remains across its whole span (the stride).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradezbotz.cli import _history_floor
from tradezbotz.research.prices import BASIS_PRICE, BASIS_TOTAL, Bar, PriceCache, Series


def _series(symbol: str, first: date, days: int, basis: str) -> Series:
    bars = tuple(Bar(day=first + timedelta(days=i), open=10.0, high=11.0,
                     low=9.0, close=10.0, volume=1_000) for i in range(days))
    return Series(symbol=symbol, bars=bars, requested_start=first,
                  requested_end=first + timedelta(days=days - 1), basis=basis)


def _cache(tmp_path, first: date, days: int = 400, basis: str = BASIS_TOTAL):
    cache = PriceCache(tmp_path / "bars.db")
    cache.put(_series("AAA", first, days, basis))
    return cache


# --- the history floor ------------------------------------------------------

def test_the_floor_is_measured_from_the_cache_not_the_partition(tmp_path):
    """The constraint is how far back the vendor's history goes, which has
    nothing to do with where a split boundary falls. Flooring from the
    partition would wrongly discard the opening months of validation and
    holdout, where history is already deep."""
    cache = _cache(tmp_path, date(2016, 1, 4))
    cache.close()

    floor = _history_floor(tmp_path / "bars.db", BASIS_TOTAL, 300)

    assert floor == date(2016, 1, 4) + timedelta(days=300)


def test_a_zero_floor_is_disabled_rather_than_zero_days(tmp_path):
    cache = _cache(tmp_path, date(2016, 1, 4))
    cache.close()

    assert _history_floor(tmp_path / "bars.db", BASIS_TOTAL, 0) is None


def test_an_empty_cache_floors_nothing_rather_than_dropping_everything(tmp_path):
    """A missing cache must not silently discard the entire population -- that
    reads as "no events" rather than "no prices"."""
    PriceCache(tmp_path / "bars.db").close()

    assert _history_floor(tmp_path / "bars.db", BASIS_TOTAL, 300) is None


def test_the_floor_respects_the_basis_it_was_asked_for(tmp_path):
    """A price-only series fetched later than the total-return one must not
    move the floor for a total-return run."""
    cache = _cache(tmp_path, date(2016, 1, 4), basis=BASIS_TOTAL)
    cache.put(_series("BBB", date(2020, 1, 2), 10, BASIS_PRICE))
    cache.close()

    total = _history_floor(tmp_path / "bars.db", BASIS_TOTAL, 300)
    price = _history_floor(tmp_path / "bars.db", BASIS_PRICE, 300)

    assert total == date(2016, 1, 4) + timedelta(days=300)
    assert price == date(2020, 1, 2) + timedelta(days=300)


def test_earliest_day_across_all_bases(tmp_path):
    cache = _cache(tmp_path, date(2016, 1, 4))
    try:
        assert cache.earliest_day() == date(2016, 1, 4)
    finally:
        cache.close()


# --- the stride -------------------------------------------------------------

def _stride(rows, limit):
    """The sampling rule as `cmd_measure` applies it."""
    if not limit or len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)]


def test_a_sample_spans_the_whole_partition_not_one_end_of_it():
    """The failure this exists to prevent: 40,000 events taken from the tail
    of a 2,108-day partition covered 52 days, so every candidate was measured
    inside late 2021 alone."""
    rows = list(range(100_000))

    sample = _stride(rows, 1_000)

    assert len(sample) == 1_000
    assert sample[0] == 0
    assert sample[-1] > 99_000, "the sample must reach the end of the partition"


def test_the_sample_is_evenly_spaced_rather_than_clumped():
    rows = list(range(10_000))

    sample = _stride(rows, 100)
    gaps = {sample[i + 1] - sample[i] for i in range(len(sample) - 1)}

    assert gaps <= {100}, f"uneven spacing would re-introduce regime bias: {gaps}"


def test_sampling_preserves_chronological_order():
    """Downstream clustering keys on date, and an unordered sample would make
    the first/last reported span meaningless."""
    rows = list(range(5_000))

    sample = _stride(rows, 250)

    assert sample == sorted(sample)


def test_a_limit_at_or_above_the_population_takes_everything():
    rows = list(range(500))

    assert _stride(rows, 500) == rows
    assert _stride(rows, 10_000) == rows


def test_no_limit_takes_everything():
    rows = list(range(500))

    assert _stride(rows, 0) == rows


def test_the_sample_never_repeats_an_event():
    """A stride computed with integer division rather than a float would
    duplicate rows near the end, inflating the trade count with copies."""
    rows = list(range(9_973))

    sample = _stride(rows, 1_000)

    assert len(set(sample)) == len(sample)


@pytest.mark.parametrize("population,limit", [(1, 1), (2, 1), (7, 3), (1_001, 7)])
def test_the_stride_is_in_range_for_awkward_sizes(population, limit):
    rows = list(range(population))

    sample = _stride(rows, limit)

    assert len(sample) == min(limit, population)
    assert all(0 <= s < population for s in sample)
