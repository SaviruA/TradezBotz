"""Tests for point-in-time correctness.

Every test here is guarding against a way the backtest could see something the
market could not. These are the highest-value tests in the repo: a bug in the
strategy loses money slowly, a bug here invents money that never existed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradezbotz.research.eventstore import Event, EventStore, EventStoreError

UTC = timezone.utc


def ts(day: int, hour: int = 12) -> datetime:
    return datetime(2025, 3, day, hour, tzinfo=UTC)


def make_event(**overrides) -> Event:
    base = dict(
        source="test",
        external_id="e1",
        kind="insider_transaction",
        symbol="AAPL",
        observed_at=ts(10),
        occurred_at=ts(8),
        payload={"shares": 100},
    )
    base.update(overrides)
    return Event(**base)


@pytest.fixture
def store(tmp_path):
    with EventStore(tmp_path / "events.db") as s:
        yield s


def test_event_hidden_before_it_was_observable(store):
    """The core guarantee: a trade disclosed on the 10th is invisible on the 9th."""
    store.record(make_event())

    assert list(store.as_of(ts(9))) == []
    assert len(list(store.as_of(ts(10)))) == 1
    assert len(list(store.as_of(ts(11)))) == 1


def test_querying_never_keys_on_occurred_at(store):
    """occurred_at is 2 days earlier; it must not make the event visible early."""
    store.record(make_event(occurred_at=ts(8), observed_at=ts(10)))

    # The transaction happened on the 8th, but nobody could know until the 10th.
    assert list(store.as_of(ts(8, 23))) == []


def test_naive_datetime_rejected(store):
    """Naive datetimes silently shift by hours and cause session-boundary bugs."""
    with pytest.raises(EventStoreError, match="timezone-aware"):
        store.record(make_event(observed_at=datetime(2025, 3, 10, 12)))


def test_event_cannot_be_observed_before_it_occurred():
    with pytest.raises(EventStoreError, match="visible before it happened"):
        make_event(occurred_at=ts(12), observed_at=ts(10))


def test_revisions_do_not_leak_backwards(store):
    """A correction filed later must not rewrite what was known earlier."""
    store.record(make_event(revision=0, observed_at=ts(10), payload={"shares": 100}))
    store.record(make_event(revision=1, observed_at=ts(20), payload={"shares": 250}))

    early = list(store.as_of(ts(15)))
    assert len(early) == 1
    assert early[0]["payload"]["shares"] == 100, "restatement leaked into the past"

    late = list(store.as_of(ts(25)))
    assert len(late) == 1
    assert late[0]["payload"]["shares"] == 250


def test_reingest_is_idempotent(store):
    """Re-running an ingest over an overlapping range must not duplicate rows."""
    event = make_event()
    assert store.record(event) is True
    assert store.record(event) is False
    assert store.count() == 1


def test_filters_compose(store):
    store.record(make_event(external_id="a", symbol="AAPL", observed_at=ts(10)))
    store.record(make_event(external_id="b", symbol="MSFT", observed_at=ts(11)))
    store.record(
        make_event(external_id="c", symbol="AAPL", kind="news", observed_at=ts(12))
    )

    assert len(list(store.as_of(ts(20), symbol="AAPL"))) == 2
    assert len(list(store.as_of(ts(20), kind="news"))) == 1
    assert len(list(store.as_of(ts(20), since=ts(11)))) == 2


def test_results_ordered_by_observation_time(store):
    for i, day in enumerate([15, 11, 13]):
        store.record(make_event(external_id=f"e{i}", observed_at=ts(day)))

    seen = [r["observed_at"] for r in store.as_of(ts(20))]
    assert seen == sorted(seen)


# --- ingest checkpointing ----------------------------------------------------

def test_unseen_day_is_not_marked_ingested(store):
    from datetime import date

    assert store.day_ingested("sec_form4", date(2025, 3, 10)) is False


def test_marking_a_day_lets_a_sliced_run_skip_it(store):
    """Events dedupe on insert, so this is purely about not re-downloading.
    Re-fetching one EDGAR day costs ~1000 requests."""
    from datetime import date

    store.mark_day_ingested("sec_form4", date(2025, 3, 10), events=998)

    assert store.day_ingested("sec_form4", date(2025, 3, 10)) is True
    assert store.day_ingested("sec_form4", date(2025, 3, 11)) is False


def test_day_checkpoints_are_per_source(store):
    from datetime import date

    store.mark_day_ingested("sec_form4", date(2025, 3, 10), events=5)

    assert store.day_ingested("congress_ptr", date(2025, 3, 10)) is False


def test_remarking_a_day_is_idempotent(store):
    from datetime import date

    store.mark_day_ingested("sec_form4", date(2025, 3, 10), events=5)
    store.mark_day_ingested("sec_form4", date(2025, 3, 10), events=7)

    assert store.days_ingested("sec_form4") == {date(2025, 3, 10)}


def test_days_ingested_returns_the_completed_set(store):
    from datetime import date

    for d in (date(2025, 3, 10), date(2025, 3, 11)):
        store.mark_day_ingested("sec_form4", d, events=1)

    assert store.days_ingested("sec_form4") == {date(2025, 3, 10), date(2025, 3, 11)}
    assert store.days_ingested("other") == set()
