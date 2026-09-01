"""Tests for the point-in-time joins.

Almost every test here is a lookahead test. The joins all answer "what else was
true about this symbol around this date", and the tempting implementation --
take the nearest record -- reads records disclosed *after* the entry. That
mistake does not fail: it produces a stronger, cleaner-looking result, which is
the reason it needs a test rather than a review.

The disclosure lags that make it possible are large and real: up to 45 days for
a House PTR, 45 for a 13F, 5 business days for a 13D, and a quarter or more for
an XBRL fact.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.eventstore import Event, EventStore
from tradezbotz.research.holdings import KIND_CONGRESS, KIND_STAKE
from tradezbotz.research.joins import (
    RELATED_WINDOW_DAYS,
    FundamentalsJoin,
    HoldingsJoin,
    ProfileJoin,
    enrich_all,
)
from tradezbotz.research.labeler import Coverage, Label

ENTRY = date(2025, 6, 10)


def label(symbol="AAA", entry=ENTRY) -> Label:
    return Label(symbol=symbol, observed_at=datetime(2025, 6, 9, tzinfo=UTC),
                 entry_day=entry, entry_price=10.0, returns={5: 0.01},
                 coverage=Coverage.COMPLETE)


def congress_event(symbol, filed: date, traded: date, purchase=True) -> Event:
    return Event(
        source="house_ptr", external_id=f"{symbol}:{traded}:{filed}",
        kind=KIND_CONGRESS, symbol=symbol,
        observed_at=datetime.combine(filed, datetime.min.time(), tzinfo=UTC),
        occurred_at=datetime.combine(traded, datetime.min.time(), tzinfo=UTC),
        payload={"is_purchase": purchase, "amount_high": 50_000.0,
                 "member": "A Member"},
    )


@pytest.fixture
def store(tmp_path):
    with EventStore(tmp_path / "e.db") as s:
        yield s


# --- the holdings join, which is where the lag bites ------------------------

def test_a_disclosure_filed_before_entry_is_seen(store):
    store.record(congress_event("AAA", filed=date(2025, 5, 1),
                                traded=date(2025, 4, 2)))

    out = HoldingsJoin(store).features(label())

    assert out["congress_bought"] is True
    assert out["congress_buy_count"] == 1


def test_a_trade_dated_before_entry_but_filed_after_is_not_seen(store):
    """The core lookahead case, and the one the 45-day PTR lag creates.

    The member traded on 2 June, comfortably inside the window. They disclosed
    it on 1 July, three weeks after we bought. Matching on the transaction date
    would count it; matching on the filing date does not.
    """
    store.record(congress_event("AAA", filed=date(2025, 7, 1),
                                traded=date(2025, 6, 2)))

    out = HoldingsJoin(store).features(label())

    assert out == {}, "a filing published after entry must be invisible"


def test_a_disclosure_on_the_entry_day_itself_is_not_seen(store):
    """We buy the entry session's OPEN. Something disclosed during that session
    is not knowable when the order goes out."""
    store.record(congress_event("AAA", filed=ENTRY, traded=date(2025, 5, 1)))

    assert HoldingsJoin(store).features(label()) == {}


def test_a_disclosure_older_than_the_window_is_not_seen(store):
    store.record(congress_event(
        "AAA", filed=ENTRY - timedelta(days=RELATED_WINDOW_DAYS + 10),
        traded=ENTRY - timedelta(days=RELATED_WINDOW_DAYS + 40)))

    assert HoldingsJoin(store).features(label()) == {}


def test_a_sale_is_not_counted_as_a_purchase(store):
    store.record(congress_event("AAA", filed=date(2025, 5, 1),
                                traded=date(2025, 4, 2), purchase=False))

    out = HoldingsJoin(store).features(label())

    assert out["congress_bought"] is False
    assert out["congress_buy_count"] == 0


def test_another_symbols_disclosure_does_not_match(store):
    store.record(congress_event("BBB", filed=date(2025, 5, 1),
                                traded=date(2025, 4, 2)))

    assert HoldingsJoin(store).features(label("AAA")) == {}


def test_an_activist_stake_is_distinguished_from_a_passive_one(store):
    store.record(Event(
        source="sec_13d", external_id="s1", kind=KIND_STAKE, symbol="AAA",
        observed_at=datetime(2025, 5, 20, tzinfo=UTC),
        payload={"activist": True, "holder": "Someone"}))

    out = HoldingsJoin(store).features(label())

    assert out["stake_filed"] is True
    assert out["activist_stake"] is True


def test_the_index_is_built_once_not_per_event(store):
    for i in range(3):
        store.record(congress_event(f"S{i}", filed=date(2025, 5, 1),
                                    traded=date(2025, 4, 2)))
    join = HoldingsJoin(store)
    calls = {"n": 0}
    original = store.as_of

    def counting(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    store.as_of = counting
    for i in range(3):
        join.features(label(f"S{i}"))

    # Three kinds, one pass each, regardless of how many events are enriched.
    assert calls["n"] == 3


# --- the fundamentals join --------------------------------------------------

class FakeFacts:
    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, cik):
        return self.mapping.get(str(int(cik)), {})


class FakeBars:
    """Daily closes, so the join has a prior close to price against."""

    def __init__(self, closes):
        self.closes = closes

    def get(self, symbol, start, end, basis=None):
        from tradezbotz.research.prices import Bar, Series
        bars = tuple(
            Bar(day=d, open=c, high=c, low=c, close=c, volume=1000.0)
            for d, c in sorted(self.closes.items()) if start <= d <= end
        )
        return Series(symbol=symbol, bars=bars, requested_start=start,
                      requested_end=end)


def facts_for(filed: date, end: date, revenue: float) -> dict:
    """A minimal companyfacts document with one annual revenue observation."""
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{
        "val": revenue, "start": (end - timedelta(days=364)).isoformat(),
        "end": end.isoformat(), "filed": filed.isoformat(), "form": "10-K",
    }]}}}}}


def test_a_fact_filed_after_entry_is_invisible():
    """`fundamentals.visible` enforces this, and the join has to actually route
    through it rather than reading the newest observation."""
    late = FakeFacts({"320193": facts_for(
        filed=date(2025, 8, 1), end=date(2025, 3, 31), revenue=1_000_000.0)})
    bars = FakeBars({ENTRY - timedelta(days=1): 10.0})

    out = FundamentalsJoin(late, price_cache=bars).features(
        {"issuer_cik": "0000320193"}, label())

    assert "price_to_sales" not in out


def test_a_fact_filed_before_entry_is_used():
    early = FakeFacts({"320193": facts_for(
        filed=date(2025, 4, 15), end=date(2025, 3, 31), revenue=1_000_000.0)})
    bars = FakeBars({ENTRY - timedelta(days=1): 10.0})

    out = FundamentalsJoin(early, price_cache=bars).features(
        {"issuer_cik": "0000320193"}, label())

    assert out["has_fundamentals"] is True


def test_the_price_used_is_the_prior_close_not_the_entry_open():
    """`label.entry_price` is the entry session's open, which the decision
    cannot see. Pricing a multiple off it puts hours of lookahead into every
    valuation feature."""
    facts = FakeFacts({"1": facts_for(
        filed=date(2025, 4, 15), end=date(2025, 3, 31), revenue=1_000_000.0)})
    bars = FakeBars({ENTRY - timedelta(days=1): 7.0, ENTRY: 99.0})

    join = FundamentalsJoin(facts, price_cache=bars)
    assert join._price(label()) == pytest.approx(7.0)


def test_no_cached_facts_is_counted_not_guessed():
    join = FundamentalsJoin(FakeFacts({}), price_cache=FakeBars({}))

    assert join.features({"issuer_cik": "999"}, label()) == {}
    assert join.skipped_no_facts == 1


def test_a_payload_without_a_cik_yields_nothing():
    join = FundamentalsJoin(FakeFacts({}), price_cache=FakeBars({}))

    assert join.features({}, label()) == {}


# --- dispatch ---------------------------------------------------------------

class NeedsLabel:
    needs_payload = False

    def features(self, label):
        return {"from_label": True}


class NeedsPayload:
    needs_payload = True

    def features(self, payload, label):
        return {"saw_cik": payload.get("issuer_cik")}


class Exploding:
    needs_payload = True

    def features(self, payload, label):
        raise TypeError("a real bug inside the join")


def test_enrich_all_dispatches_on_the_declared_flag():
    out = enrich_all([{"issuer_cik": "42"}], [label()],
                     NeedsLabel(), NeedsPayload())

    assert out[0]["from_label"] is True
    assert out[0]["saw_cik"] == "42"


def test_a_typeerror_inside_a_join_is_not_swallowed():
    """The reason dispatch is declared rather than probed. Catching TypeError to
    decide the signature would retry this join with the wrong arguments and
    turn a bug into quietly missing data."""
    with pytest.raises(TypeError, match="a real bug inside the join"):
        enrich_all([{}], [label()], Exploding())


def test_enrich_all_does_not_mutate_the_stored_payload():
    original = {"issuer_cik": "42"}

    enrich_all([original], [label()], NeedsLabel())

    assert original == {"issuer_cik": "42"}


# --- the profile join -------------------------------------------------------

class FakeProfiles:
    """Returns everything it holds, ignoring the requested range.

    Deliberately sloppy. If the join relied on the store's range query to
    exclude the entry session, this fake would let the entry session through
    and the test below would catch it. A well-behaved fake would hide that
    dependence and the test would pass either way.
    """

    def __init__(self, sessions):
        self.sessions = sessions
        self.calls = []

    def range(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        return list(self.sessions)


def session(day: date, low=9.0, high=11.0, close=10.0):
    from tradezbotz.research.intraday import SessionProfile
    return SessionProfile(
        symbol="AAA", day=day, low=low, high=high, volume=10_000.0, vwap=10.0,
        histogram=tuple([250.0] * 40), delta=1000.0, unsigned_volume=0.0,
        minute_count=390, session_open=10.0, session_close=close,
        low_minute=10, high_minute=200, volume_after_low=9_000.0,
        volume_after_high=2_000.0)


def test_the_entry_session_is_never_read():
    """Its profile encodes the very move being measured.

    The store here hands back the entry session even though it was not asked
    for, so the only thing that can exclude it is the join's own filter. Drop
    that filter and `profile_sessions` becomes 11.
    """
    days = [ENTRY - timedelta(days=i) for i in range(1, 11)]
    store = FakeProfiles([session(d) for d in days] + [session(ENTRY)])

    out = ProfileJoin(store).features(label())

    assert out["profile_sessions"] == 10


def test_the_profile_query_never_asks_for_the_entry_day():
    """Belt and braces on the same rule: the request itself stops the day
    before, so a store that honours its range cannot return the entry
    session either."""
    store = FakeProfiles([session(ENTRY - timedelta(days=i))
                          for i in range(1, 11)])

    ProfileJoin(store).features(label())

    _, _, end = store.calls[0]
    assert end < ENTRY


def test_too_few_prior_sessions_yields_nothing_and_is_counted():
    store = FakeProfiles([session(ENTRY - timedelta(days=i)) for i in (1, 2)])
    join = ProfileJoin(store)

    assert join.features(label()) == {}
    assert join.skipped_no_sessions == 1


def test_profile_features_are_produced_with_enough_history():
    store = FakeProfiles([session(ENTRY - timedelta(days=i))
                          for i in range(1, 11)])

    out = ProfileJoin(store).features(label())

    assert out["has_profile"] is True
    assert out["profile_sessions"] == 10
    assert "above_poc" in out
    assert "below_value_area" in out
