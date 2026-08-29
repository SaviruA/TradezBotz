"""Tests for the ApeWisdom sentiment collector."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradezbotz.research.apewisdom import (
    ApeWisdomClient,
    ApeWisdomError,
    Mention,
    to_events,
)


class FakeResp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(self.status_code)


class FakeSession:
    def __init__(self, *pages):
        self._pages = list(pages)
        self.headers: dict[str, str] = {}
        self.urls: list[str] = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return self._pages.pop(0) if self._pages else FakeResp({"results": [], "pages": 1})


PAGE1 = {
    "count": 4, "pages": 2, "current_page": 1,
    "results": [
        {"rank": 1, "ticker": "NVDA", "name": "NVIDIA", "mentions": 303,
         "upvotes": 724, "rank_24h_ago": 1, "mentions_24h_ago": 714},
        {"rank": 2, "ticker": "SPY", "name": "SPDR S&amp;P 500 ETF Trust",
         "mentions": 229, "upvotes": 785, "rank_24h_ago": 5,
         "mentions_24h_ago": 187},
    ],
}
PAGE2 = {
    "count": 4, "pages": 2, "current_page": 2,
    "results": [
        {"rank": 3, "ticker": "mu", "name": "Micron", "mentions": 137,
         "upvotes": 276, "rank_24h_ago": None, "mentions_24h_ago": None},
        {"rank": 4, "ticker": "", "name": "broken", "mentions": 5},   # dropped
    ],
}


def client(*pages, interval=0.0):
    return ApeWisdomClient(FakeSession(*pages), interval=interval)


# --- fetching ----------------------------------------------------------------

def test_paginates_until_the_reported_last_page():
    c = client(FakeResp(PAGE1), FakeResp(PAGE2))

    out = c.mentions("wallstreetbets")

    assert [m.ticker for m in out] == ["NVDA", "SPY", "MU"]
    assert len(c.session.urls) == 2
    assert c.session.urls[1].endswith("/page/2")


def test_ticker_is_normalised_upper():
    assert client(FakeResp(PAGE2)).mentions("x")[0].ticker == "MU"


def test_html_escaped_names_are_decoded():
    """Names arrive escaped: 'SPDR S&amp;P 500 ETF Trust'."""
    spy = next(m for m in client(FakeResp(PAGE1)).mentions("x") if m.ticker == "SPY")

    assert spy.name == "SPDR S&P 500 ETF Trust"


def test_rows_without_ticker_or_mentions_are_dropped():
    out = client(FakeResp(PAGE2)).mentions("x")

    assert all(m.ticker for m in out)
    assert len(out) == 1, "the blank-ticker row is unusable, not zero"


def test_missing_24h_fields_are_none_not_zero():
    """Absent history is unknown, not 'no change'."""
    mu = client(FakeResp(PAGE2)).mentions("x")[0]

    assert mu.mentions_24h_ago is None
    assert mu.mention_change is None


def test_non_dict_payload_raises():
    c = ApeWisdomClient(FakeSession(FakeResp([])), interval=0.0)

    with pytest.raises(ApeWisdomError, match="unexpected payload"):
        c.mentions("x")


def test_pagination_is_bounded_against_a_malformed_page_count():
    runaway = {"pages": 9999, "results": [
        {"rank": 1, "ticker": "AAA", "mentions": 1, "upvotes": 0}]}
    c = ApeWisdomClient(FakeSession(*[FakeResp(runaway)] * 40), interval=0.0)

    c.mentions("x")

    assert len(c.session.urls) <= 25


# --- event conversion --------------------------------------------------------

NOW = datetime(2026, 8, 29, 14, 30, tzinfo=UTC)


def mention(ticker="NVDA", **kw):
    base = dict(ticker=ticker, name="NVIDIA", rank=1, mentions=303, upvotes=724,
                rank_24h_ago=1, mentions_24h_ago=714, filter_name="wallstreetbets")
    base.update(kw)
    return Mention(**base)


def test_observed_at_is_the_fetch_time():
    """The snapshot became knowable to us when we fetched it, not before."""
    e = next(to_events([mention()], NOW))

    assert e.observed_at == NOW


def test_occurred_at_is_left_unset():
    """A mention count aggregates an undisclosed window. Claiming a moment for
    it would invent precision the source does not provide."""
    e = next(to_events([mention()], NOW))

    assert e.occurred_at is None


def test_repeat_polls_within_an_hour_dedupe():
    a = next(to_events([mention()], datetime(2026, 8, 29, 14, 5, tzinfo=UTC)))
    b = next(to_events([mention()], datetime(2026, 8, 29, 14, 55, tzinfo=UTC)))

    assert a.external_id == b.external_id


def test_different_hours_are_distinct_observations():
    a = next(to_events([mention()], datetime(2026, 8, 29, 14, 5, tzinfo=UTC)))
    b = next(to_events([mention()], datetime(2026, 8, 29, 15, 5, tzinfo=UTC)))

    assert a.external_id != b.external_id


def test_same_ticker_in_two_communities_stays_distinct():
    """A mention concentrated in one subreddit is a different observation from
    the same ticker trending across several."""
    a = next(to_events([mention(filter_name="wallstreetbets")], NOW))
    b = next(to_events([mention(filter_name="stocks")], NOW))

    assert a.external_id != b.external_id


def test_payload_carries_the_change_for_later_analysis():
    e = next(to_events([mention()], NOW))

    assert e.payload["mention_change"] == 303 - 714
    assert e.payload["filter"] == "wallstreetbets"


def test_events_are_storable(tmp_path):
    from tradezbotz.research.eventstore import EventStore

    with EventStore(tmp_path / "e.db") as store:
        assert store.record_many(list(to_events([mention()], NOW))) == 1
        assert store.record_many(list(to_events([mention()], NOW))) == 0


# --- data quality flagging ---------------------------------------------------

def test_ambiguous_tickers_are_flagged_in_the_payload():
    """8 of the top 100 WSB tickers were ordinary English words when measured.
    Flagged, not filtered: excluding them would bias the universe as badly as
    trusting them, since a genuine ServiceNow discussion is real signal."""
    it = next(to_events([mention(ticker="IT")], NOW))
    nvda = next(to_events([mention(ticker="NVDA")], NOW))

    assert it.payload["ambiguous_ticker"] is True
    assert nvda.payload["ambiguous_ticker"] is False


def test_ambiguity_report_quantifies_contamination():
    from tradezbotz.research.apewisdom import ambiguity_report

    rep = ambiguity_report([
        mention(ticker="NVDA", mentions=300),
        mention(ticker="IT", mentions=100),
    ])

    assert rep["tickers"] == 2
    assert rep["ambiguous_tickers"] == 1
    assert rep["ambiguous_ticker_rate"] == pytest.approx(0.5)
    assert rep["ambiguous_mention_share"] == pytest.approx(0.25)


def test_ambiguity_report_handles_empty():
    from tradezbotz.research.apewisdom import ambiguity_report

    assert ambiguity_report([]) == {"total": 0}


def test_known_english_word_tickers_are_listed():
    from tradezbotz.research.apewisdom import AMBIGUOUS_TICKERS

    for t in ("IT", "ALL", "SO", "ON", "BE", "ANY", "NOW", "OPEN", "DD"):
        assert t in AMBIGUOUS_TICKERS
