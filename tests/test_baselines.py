"""Tests for point-in-time insider baselines.

The lookahead risk here is subtler than in price data: it hides inside a *label*.
Classifying a June trade using September filings means the backtest knows an
insider was routine before anyone could have.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from tradezbotz.research.baselines import (
    baseline_start,
    classify_events,
    coverage_warning,
)
from tradezbotz.research.classify import InsiderClass


def ev(owner, txn_day: date, observed: datetime, symbol="AAPL"):
    return {
        "symbol": symbol,
        "observed_at": observed,
        "payload": {
            "owner_cik": owner,
            "owner_name": "DOE JANE",
            "transaction_date": txn_day.isoformat(),
        },
    }


def utc(y, m, d):
    return datetime(y, m, d, 20, 0, tzinfo=UTC)


def december_history(owner, years):
    """One December trade per year, disclosed two days later."""
    return [ev(owner, date(y, 12, 15), utc(y, 12, 17)) for y in years]


def test_routine_pattern_is_detected_with_enough_history():
    events = december_history("A", [2021, 2022, 2023]) + [
        ev("A", date(2024, 12, 10), utc(2024, 12, 12))
    ]

    result = {c.transaction_date: c.insider_class for c in classify_events(events)}

    assert result[date(2024, 12, 10)] is InsiderClass.ROUTINE


def test_off_pattern_trade_is_opportunistic():
    events = december_history("A", [2021, 2022, 2023]) + [
        ev("A", date(2024, 3, 10), utc(2024, 3, 12))
    ]

    result = {c.transaction_date: c.insider_class for c in classify_events(events)}

    assert result[date(2024, 3, 10)] is InsiderClass.OPPORTUNISTIC


def test_future_filings_cannot_influence_an_earlier_label():
    """The core guarantee. History disclosed AFTER the event being classified
    must be invisible, or the label itself carries lookahead."""
    target = ev("A", date(2022, 12, 10), utc(2022, 12, 12))
    # Three Decembers, but all disclosed years later.
    later = december_history("A", [2023, 2024, 2025])

    result = {
        (c.transaction_date, c.observed_at): c.insider_class
        for c in classify_events([target] + later)
    }

    assert result[(date(2022, 12, 10), utc(2022, 12, 12))] is InsiderClass.UNKNOWN


def test_other_insiders_history_is_not_borrowed():
    events = december_history("OTHER", [2021, 2022, 2023]) + [
        ev("A", date(2024, 12, 10), utc(2024, 12, 12))
    ]

    result = {
        c.owner_cik: c.insider_class
        for c in classify_events(events)
        if c.transaction_date == date(2024, 12, 10)
    }

    assert result["A"] is InsiderClass.UNKNOWN


def test_events_without_owner_or_date_are_skipped():
    bad = {"symbol": "X", "observed_at": utc(2024, 1, 1), "payload": {}}

    assert classify_events([bad]) == []


def test_iso_timestamps_are_accepted():
    events = december_history("A", [2021, 2022, 2023])
    events = [{**e, "observed_at": e["observed_at"].isoformat()} for e in events]
    events.append(ev("A", date(2024, 12, 10), utc(2024, 12, 12).isoformat()))

    assert len(classify_events(events)) == 4


# --- coverage warning --------------------------------------------------------

def test_shallow_ingest_triggers_the_inert_classifier_warning():
    """A 2-year ingest leaves every insider UNKNOWN, so the routine filter never
    fires and the backtest silently tests a weaker hypothesis."""
    shallow = december_history("A", [2025]) + [
        ev("A", date(2026, 3, 10), utc(2026, 3, 12))
    ]

    warning = coverage_warning(classify_events(shallow))

    assert warning is not None
    assert "UNKNOWN" in warning and "free and unbounded" in warning


def test_deep_ingest_produces_no_warning():
    deep = december_history("A", [2020, 2021, 2022, 2023, 2024])

    assert coverage_warning(classify_events(deep)) is None


def test_no_warning_for_empty_input():
    assert coverage_warning([]) is None


def test_baseline_start_reaches_further_back_than_the_label_window():
    label_start = date(2024, 8, 28)

    start = baseline_start(label_start)

    assert start < label_start
    assert (label_start - start).days >= 365 * 3
