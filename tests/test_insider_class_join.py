"""Tests for the routine/opportunistic join.

Cohen, Malloy & Pomorski (JF 2012) found routine trades -- an insider buying
the same calendar month year after year -- are over half the insider universe
and carry essentially no predictive power, while the remainder carried
82bp/month. `classify.RoutineClassifier` implemented this and was never
imported by `candidates`, so it had never appeared in a sweep: every insider
measurement this system produced pooled both populations.

Two failure modes would each produce a confident wrong answer, so both are
pinned here: leaking an insider's later trades into today's classification, and
collapsing UNKNOWN into OPPORTUNISTIC to pad the signal population.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.joins import InsiderClassJoin
from tradezbotz.research.labeler import Coverage, Label


class _Store:
    """Minimal stand-in exposing the one method the join uses."""

    def __init__(self, rows):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE events (kind TEXT, payload TEXT, observed_at TEXT)")
        for cik, tdate, observed in rows:
            self.conn.execute(
                "INSERT INTO events VALUES ('insider_transaction', ?, ?)",
                (json.dumps({"owner_cik": cik, "transaction_date": tdate}),
                 observed))
        self.conn.commit()

    def raw_query(self, sql, params=()):
        return self.conn.execute(sql, params)


def _label(day=date(2020, 3, 10), symbol="AAA"):
    return Label(symbol=symbol, observed_at=datetime(2020, 3, 9, tzinfo=UTC),
                 entry_day=day, entry_price=10.0, returns={5: 0.01},
                 coverage=Coverage.COMPLETE)


def _march(years, cik="C1"):
    """A same-month (March) trade in each given year, disclosed two days later."""
    return [(cik, f"{y}-03-05", f"{y}-03-07T00:00:00+00:00") for y in years]


# --- the classification itself ----------------------------------------------

def test_a_same_month_habit_over_three_years_is_routine():
    store = _Store(_march([2017, 2018, 2019]))
    join = InsiderClassJoin(store)

    out = join.features({"owner_cik": "C1"}, _label())

    assert out["insider_class"] == "routine"
    assert out["is_routine"] is True
    assert out["is_opportunistic"] is False


def test_a_trade_breaking_the_insiders_own_pattern_is_opportunistic():
    """Three years of history, but in a different month from this trade."""
    store = _Store([("C1", f"{y}-09-05", f"{y}-09-07T00:00:00+00:00")
                    for y in (2017, 2018, 2019)])
    join = InsiderClassJoin(store)

    out = join.features({"owner_cik": "C1"}, _label(date(2020, 3, 10)))

    assert out["insider_class"] == "opportunistic"
    assert out["is_opportunistic"] is True


def test_a_lapsed_same_month_habit_still_counts_as_routine():
    """2015-2017 in March, then a two-year gap, then a March 2020 trade.

    `_same_month_streak` takes the LONGEST run anywhere in the record rather
    than the most recent one, so a lapsed habit still classifies routine. That
    is arguable either way -- the insider's pattern had stopped -- and it is
    kept because it errs in the safe direction: calling a borderline case
    routine SHRINKS the opportunistic population, where the dangerous error
    would be padding the signal set with trades that carry nothing.
    """
    store = _Store(_march([2015, 2016, 2017]))
    join = InsiderClassJoin(store)

    out = join.features({"owner_cik": "C1"}, _label(date(2020, 3, 10)))

    assert out["insider_class"] == "routine"


# --- the two ways to fool yourself ------------------------------------------

def test_later_trades_never_reach_todays_classification():
    """The lookahead. An insider who becomes a March regular AFTER this trade
    was not a regular at the time, and using the full record would leak their
    future into today's label."""
    store = _Store(_march([2021, 2022, 2023]))
    join = InsiderClassJoin(store)

    out = join.features({"owner_cik": "C1"}, _label(date(2020, 3, 10)))

    assert out["insider_class"] == "unknown"


def test_a_trade_disclosed_on_the_entry_day_is_not_yet_knowable():
    """Entry is that session's open; a filing disseminated during the session
    was not available when we bought it."""
    store = _Store(_march([2017, 2018, 2019]) +
                   [("C1", "2020-03-10", "2020-03-10T14:00:00+00:00")])
    join = InsiderClassJoin(store)

    out = join.features({"owner_cik": "C1"}, _label(date(2020, 3, 10)))

    # Still routine from the three prior years -- the point is the same-day
    # filing neither adds nor removes anything.
    assert out["insider_class"] == "routine"


def test_too_little_history_is_unknown_never_opportunistic():
    """Collapsing UNKNOWN into OPPORTUNISTIC would pad the signal population
    with first-time filers, which is the obvious way to manufacture an edge."""
    store = _Store(_march([2019]))
    join = InsiderClassJoin(store)

    out = join.features({"owner_cik": "C1"}, _label())

    assert out["insider_class"] == "unknown"
    assert "is_opportunistic" not in out


def test_an_insider_with_no_record_at_all_is_unknown():
    join = InsiderClassJoin(_Store([]))

    out = join.features({"owner_cik": "NOBODY"}, _label())

    assert out["insider_class"] == "unknown"


# --- mechanics --------------------------------------------------------------

def test_history_is_restricted_to_the_owners_asked_for():
    """Indexing every insider in a 3.9M-event store to answer questions about a
    fraction of them costs hundreds of megabytes for nothing."""
    store = _Store(_march([2017, 2018, 2019], cik="C1") +
                   _march([2017, 2018, 2019], cik="C2"))
    join = InsiderClassJoin(store, owner_ciks={"C1"})

    assert join.features({"owner_cik": "C1"}, _label())["is_routine"] is True
    assert join.features({"owner_cik": "C2"}, _label())["insider_class"] == "unknown"


def test_a_payload_without_an_owner_yields_nothing_rather_than_a_guess():
    join = InsiderClassJoin(_Store(_march([2017, 2018, 2019])))

    assert join.features({}, _label()) == {}


def test_the_join_declares_that_it_needs_the_payload():
    """`enrich_all` dispatches on this rather than probing for a TypeError."""
    assert InsiderClassJoin.needs_payload is True


def test_the_summary_reports_both_populations_and_the_unknowns():
    store = _Store(_march([2017, 2018, 2019], cik="C1") +
                   [("C2", f"{y}-09-05", f"{y}-09-07T00:00:00+00:00")
                    for y in (2017, 2018, 2019)])
    join = InsiderClassJoin(store)
    join.features({"owner_cik": "C1"}, _label())
    join.features({"owner_cik": "C2"}, _label())
    join.features({"owner_cik": "C3"}, _label())

    text = join.summary()

    assert "1 opportunistic" in text
    assert "1 routine" in text
    assert "UNKNOWN, not assumed either way" in text


def test_a_malformed_transaction_date_is_skipped_not_fatal():
    store = _Store([("C1", "not-a-date", "2019-03-07T00:00:00+00:00")] +
                   _march([2017, 2018, 2019]))
    join = InsiderClassJoin(store)

    assert join.features({"owner_cik": "C1"}, _label())["is_routine"] is True
