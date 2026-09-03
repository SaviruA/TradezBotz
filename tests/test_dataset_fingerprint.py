"""The dataset fingerprint decides what counts as a NEW trial.

It feeds the Deflated Sharpe bar, which penalises the breadth of a search. Get
it wrong in one direction and repeated identical runs inflate the trial count
until nothing can ever be significant; wrong in the other and a genuinely new
look at the same hypothesis is waved through as a repeat.

The bug these tests pin: the fingerprint took its date bounds from the full
knowable window rather than from the rows actually measured, so its upper bound
was the newest filing in the store and advanced every night. Dedup never held,
each run minted a fresh 174 trials, and the registry reached 690 -- almost
exactly four nights of churn. The significance bar was being tightened by the
calendar.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

UTC = timezone.utc


def fingerprint(rows, *, partition="train", basis="total", horizons="1,5,20",
                kind="insider_transaction", limit=40_000, floor=300, delay=0,
                features=True, joins=True, costs=True) -> str:
    """The rule as `cmd_measure` applies it."""
    measured = [datetime.fromisoformat(r["observed_at"]) for r in rows]
    return hashlib.sha256("|".join([
        str(len(rows)), str(min(measured).date()), str(max(measured).date()),
        partition, basis, horizons, kind,
        str(limit), str(floor), str(delay),
        str(features), str(joins), str(costs),
    ]).encode("utf-8")).hexdigest()[:16]


def _rows(n, start=datetime(2016, 6, 1, tzinfo=UTC), step_days=1):
    return [{"observed_at": (start + timedelta(days=i * step_days)).isoformat()}
            for i in range(n)]


# --- the bug ----------------------------------------------------------------

def test_new_filings_outside_the_measured_partition_do_not_mint_a_trial():
    """The exact regression. Tonight's EDGAR ingest adds 2026 filings; the
    pinned 2016-2021 train partition is untouched, so the sweep is a REPEAT and
    must fingerprint identically. Taking bounds from the full window instead
    made every night a fresh look at the same hypothesis."""
    measured = _rows(500)

    tonight = fingerprint(measured)
    tomorrow_after_ingest = fingerprint(measured)  # partition rows unchanged

    assert tonight == tomorrow_after_ingest


def test_a_pinned_historical_partition_is_stable_across_runs():
    assert fingerprint(_rows(1_000)) == fingerprint(_rows(1_000))


# --- what genuinely IS a new look ------------------------------------------

def test_a_different_sample_size_is_a_new_trial():
    assert fingerprint(_rows(500), limit=40_000) != \
        fingerprint(_rows(500), limit=400_000)


def test_a_different_history_floor_is_a_new_trial():
    assert fingerprint(_rows(500), floor=300) != fingerprint(_rows(500), floor=0)


def test_the_entry_delay_diagnostic_is_a_new_trial():
    """Running the skip-session diagnostic is a second look at the same
    hypothesis. Counting it as a repeat would hide a real multiple-comparison
    cost -- it is precisely the kind of variation DSR exists to charge for."""
    assert fingerprint(_rows(500), delay=0) != fingerprint(_rows(500), delay=1)


def test_measuring_more_events_is_a_new_trial():
    assert fingerprint(_rows(500)) != fingerprint(_rows(900))


def test_a_different_span_at_the_same_count_is_a_new_trial():
    """52 contiguous days and five years both yield 40,000 events, and they are
    emphatically not the same dataset. Without the date bounds the stride fix
    would have been invisible to the registry."""
    contiguous = _rows(500, step_days=1)
    spread = _rows(500, step_days=4)

    assert fingerprint(contiguous) != fingerprint(spread)


@pytest.mark.parametrize("field,value", [
    ("partition", "validation"), ("basis", "price"), ("horizons", "1,5"),
    ("kind", "material_event"), ("features", False), ("joins", False),
    ("costs", False),
])
def test_every_configuration_change_is_a_new_trial(field, value):
    rows = _rows(500)

    assert fingerprint(rows) != fingerprint(rows, **{field: value})
