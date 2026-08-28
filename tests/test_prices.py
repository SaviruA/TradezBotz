"""Tests for the price adapter: caching, rate limiting, and vendor parsing."""

from __future__ import annotations

import time
from datetime import date

import pytest

from tradezbotz.research.prices import (
    Bar,
    MassivePriceSource,
    PriceCache,
    PriceError,
    RateLimiter,
    Series,
)


def make_bars(n=5, start=date(2025, 3, 3)):
    from datetime import timedelta

    out, day, i = [], start, 0
    while len(out) < n:
        if day.weekday() < 5:
            out.append(Bar(day, 100.0 + i, 102.0 + i, 98.0 + i, 101.0 + i, 1e6))
            i += 1
        day += timedelta(days=1)
    return tuple(out)


# --- Series ------------------------------------------------------------------

def test_index_on_or_after_finds_next_session():
    # Mar 3-7 (Mon-Fri), then Mar 10-12 (Mon-Wed).
    s = Series("T", make_bars(8, date(2025, 3, 3)), date(2025, 3, 1), date(2025, 3, 12))

    assert s.index_on_or_after(date(2025, 3, 3)) == 0
    assert s.index_on_or_after(date(2025, 3, 8)) == 5, "Saturday rolls to Monday"
    assert s.index_on_or_after(date(2026, 1, 1)) is None, "past the last bar"


def test_empty_series_reports_no_days():
    s = Series("T", (), date(2025, 3, 1), date(2025, 3, 10))

    assert s.first_day is None and s.last_day is None


# --- cache -------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    c = PriceCache(tmp_path / "bars.db")
    yield c
    c.close()


def test_cache_round_trips_bars(cache):
    bars = make_bars(5)
    cache.put(Series("AAPL", bars, date(2025, 3, 1), date(2025, 3, 31), is_active=True))

    got = cache.get("AAPL", date(2025, 3, 1), date(2025, 3, 31))

    assert len(got.bars) == 5
    assert got.bars[0].close == pytest.approx(101.0)
    assert got.is_active is True


def test_cache_reports_coverage_honestly(cache):
    cache.put(Series("AAPL", make_bars(5), date(2025, 3, 1), date(2025, 3, 31)))

    assert cache.covered("AAPL", date(2025, 3, 1), date(2025, 3, 31)) is True
    assert cache.covered("AAPL", date(2025, 3, 1), date(2025, 6, 30)) is False, (
        "a wider window than we fetched is not covered"
    )
    assert cache.covered("MSFT", date(2025, 3, 1), date(2025, 3, 31)) is False


def test_cache_preserves_delisted_flag(cache):
    cache.put(Series("GONE", make_bars(3), date(2025, 3, 1), date(2025, 3, 31), is_active=False))

    assert cache.get("GONE", date(2025, 3, 1), date(2025, 3, 31)).is_active is False


def test_cache_reingest_does_not_duplicate(cache):
    for _ in range(3):
        cache.put(Series("AAPL", make_bars(5), date(2025, 3, 1), date(2025, 3, 31)))

    assert len(cache.get("AAPL", date(2025, 3, 1), date(2025, 3, 31)).bars) == 5
    assert cache.symbols() == ["AAPL"]


# --- rate limiter ------------------------------------------------------------

def test_limiter_allows_burst_up_to_quota():
    lim = RateLimiter(per_minute=5)
    start = time.monotonic()

    for _ in range(5):
        lim.acquire()

    assert time.monotonic() - start < 1.0, "first N calls must not block"


def test_limiter_blocks_when_quota_exhausted(monkeypatch):
    """The 6th call inside a minute must wait rather than earn a 429."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    lim = RateLimiter(per_minute=3)
    for _ in range(4):
        lim.acquire()

    assert slept, "limiter should have waited before the 4th call"
    assert 0 < slept[0] <= 61


# --- vendor client -----------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")


class FakeSession:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, params or {}))
        return self._responses.pop(0) if self._responses else FakeResponse({"results": []})


AGG_PAYLOAD = {
    "status": "OK",
    "results": [
        {"t": 1741003200000, "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 1e6},
        {"t": 1741089600000, "o": 101.0, "h": 103.0, "l": 100.0, "c": 102.5, "v": 2e6},
    ],
}


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)

    with pytest.raises(PriceError, match="MASSIVE_API_KEY"):
        MassivePriceSource()


def test_parses_aggregates_and_requests_adjusted_bars():
    session = FakeSession(
        FakeResponse(AGG_PAYLOAD),
        FakeResponse({"results": [{"active": True}]}),
    )
    src = MassivePriceSource("k", session=session, per_minute=1000)

    series = src.daily_bars("aapl", date(2025, 3, 1), date(2025, 3, 31))

    assert series.symbol == "AAPL", "symbol is normalised"
    assert len(series.bars) == 2
    assert series.bars[0].close == pytest.approx(101.0)
    assert series.is_active is True
    assert session.requests[0][1]["adjusted"] == "true", (
        "unadjusted bars would invent returns across splits"
    )


def test_cache_prevents_a_second_fetch(tmp_path):
    session = FakeSession(
        FakeResponse(AGG_PAYLOAD),
        FakeResponse({"results": [{"active": True}]}),
    )
    cache = PriceCache(tmp_path / "bars.db")
    src = MassivePriceSource("k", cache=cache, session=session, per_minute=1000)

    src.daily_bars("AAPL", date(2025, 3, 1), date(2025, 3, 31))
    calls_after_first = len(session.requests)
    src.daily_bars("AAPL", date(2025, 3, 1), date(2025, 3, 31))

    assert len(session.requests) == calls_after_first, "second call must hit the cache"
    cache.close()


def test_403_explains_the_paid_tier():
    src = MassivePriceSource("k", session=FakeSession(FakeResponse({}, status=403)),
                             per_minute=1000)

    with pytest.raises(PriceError, match="paid plan"):
        src.daily_bars("AAPL", date(2025, 3, 1), date(2025, 3, 31))


def test_empty_result_does_not_claim_a_status():
    """No bars means we learned nothing about whether the ticker is alive."""
    src = MassivePriceSource("k", session=FakeSession(FakeResponse({"results": []})),
                             per_minute=1000)

    series = src.daily_bars("NOPE", date(2025, 3, 1), date(2025, 3, 31))

    assert series.bars == ()
    assert series.is_active is None


def test_delisted_ticker_is_detected_via_inactive_lookup():
    """Regression: the reference endpoint defaults to active=true, so a delisted
    ticker returns zero results instead of active:false. Trusting the first
    lookup reports None, which downgrades a real delisting to a coverage gap."""
    session = FakeSession(
        FakeResponse({"results": []}),                    # active lookup: nothing
        FakeResponse({"results": [{"active": False}]}),   # inactive lookup: found
    )
    src = MassivePriceSource("k", session=session, per_minute=1000)

    assert src.is_active("AACB") is False
    assert len(session.requests) == 2
    assert session.requests[1][1]["active"] == "false"


def test_active_ticker_costs_one_lookup():
    session = FakeSession(FakeResponse({"results": [{"active": True}]}))
    src = MassivePriceSource("k", session=session, per_minute=1000)

    assert src.is_active("AAPL") is True
    assert len(session.requests) == 1, "no wasted call on the common path"


def test_unknown_ticker_returns_none_after_both_lookups():
    session = FakeSession(FakeResponse({"results": []}), FakeResponse({"results": []}))
    src = MassivePriceSource("k", session=session, per_minute=1000)

    assert src.is_active("NOTREAL") is None
