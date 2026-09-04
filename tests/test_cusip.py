"""Tests for CUSIP resolution, the mapping that unblocks 13F entirely.

A 13F information table names each holding by CUSIP and issuer name, never by
ticker. `Filing13F.to_events()` therefore emits `symbol=None`, and
`HoldingsJoin` drops any event without a symbol -- so the run that first pulled
real 13F data added 124,786 events in a single day and the join still reported
"0 symbols carry a disclosure".

Two failure modes here are worse than no data at all, because nothing
downstream would report either one: attaching a foreign listing's ticker to a
US issuer, and misaligning a batch response so one company's ticker lands on
another company's CUSIP. Both are pinned below.
"""

from __future__ import annotations

import json

import pytest
import requests

from tradezbotz.research.cusip import (
    BATCH_UNAUTHENTICATED,
    CusipCache,
    CusipError,
    OpenFigiClient,
    Resolution,
    resolve_missing,
)


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.requests = []

    def post(self, url, headers=None, data=None, timeout=None):
        body = json.loads(data)
        self.requests.append((headers or {}, body))
        if callable(self.payload):
            return _Resp(self.payload(body), self.status)
        return _Resp(self.payload, self.status)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("tradezbotz.research.cusip.time.sleep", lambda s: None)


def _client(payload, **kw):
    return OpenFigiClient(session=_Session(payload), **kw)


# --- the US listing, not merely the first one -------------------------------

def test_the_us_listing_is_chosen_over_a_foreign_one():
    """A CUSIP maps to many venues. Taking the first row would sometimes return
    a foreign line whose ticker collides with an unrelated US symbol, silently
    attaching one company's disclosures to another company's returns."""
    payload = [{"data": [
        {"ticker": "XYZ", "exchCode": "GR", "name": "Frankfurt line"},
        {"ticker": "ABC", "exchCode": "US", "name": "Real Co"},
    ]}]

    out = _client(payload).resolve(["037833100"])

    assert out[0].symbol == "ABC"
    assert out[0].name == "Real Co"


def test_a_cusip_with_no_us_listing_resolves_to_nothing():
    payload = [{"data": [{"ticker": "XYZ", "exchCode": "GR", "name": "Foreign"}]}]

    out = _client(payload).resolve(["037833100"])

    assert out[0].symbol is None
    assert out[0].found is False


def test_an_empty_result_is_a_miss_not_a_crash():
    out = _client([{"warning": "No identifier found."}]).resolve(["000000000"])

    assert out[0].found is False


# --- batch alignment --------------------------------------------------------

def test_responses_are_matched_positionally_to_their_requests():
    """OpenFIGI answers positionally. Zipping against anything else attaches
    one issuer's ticker to another's CUSIP -- the single worst failure here."""
    payload = [
        {"data": [{"ticker": "AAA", "exchCode": "US"}]},
        {"data": [{"ticker": "BBB", "exchCode": "US"}]},
    ]

    out = _client(payload).resolve(["111111111", "222222222"])

    assert (out[0].cusip, out[0].symbol) == ("111111111", "AAA")
    assert (out[1].cusip, out[1].symbol) == ("222222222", "BBB")


def test_an_oversized_batch_is_refused_rather_than_truncated():
    """OpenFIGI fails the WHOLE request over its job limit, not the excess, so
    a silently truncated batch would lose every CUSIP in it."""
    client = _client([])

    with pytest.raises(CusipError, match="job limit"):
        client.resolve(["x"] * (BATCH_UNAUTHENTICATED + 1))


def test_a_key_raises_the_batch_size_and_is_sent_as_a_header():
    session = _Session([{"data": [{"ticker": "AAA", "exchCode": "US"}]}])
    client = OpenFigiClient(api_key="secret", session=session)
    client.resolve(["111111111"])

    assert client.batch_size > BATCH_UNAUTHENTICATED
    assert session.requests[0][0]["X-OPENFIGI-APIKEY"] == "secret"


def test_a_rate_limit_response_is_named_rather_than_a_bare_http_error():
    client = OpenFigiClient(session=_Session([], status=429))

    with pytest.raises(CusipError, match="rate limit"):
        client.resolve(["111111111"])


# --- the cache --------------------------------------------------------------

def test_the_cache_round_trips_a_resolution(tmp_path):
    with CusipCache(tmp_path / "c.db") as cache:
        cache.record([Resolution("111111111", "AAA", "A Co")])

        assert cache.get("111111111") == "AAA"
        assert cache.mapping() == {"111111111": "AAA"}


def test_a_miss_is_recorded_so_it_is_never_asked_again(tmp_path):
    """Most unresolved CUSIPs are delisted issuers or non-equity instruments
    the vendor genuinely does not carry. Re-asking nightly would spend the
    whole rate limit on questions already answered."""
    with CusipCache(tmp_path / "c.db") as cache:
        cache.record([Resolution("999999999", None, None)])

        assert cache.get("999999999") is None
        assert "999999999" in cache.known()
        assert cache.mapping() == {}


def test_lookups_are_case_insensitive(tmp_path):
    with CusipCache(tmp_path / "c.db") as cache:
        cache.record([Resolution("abc123456", "AAA", None)])

        assert cache.get("ABC123456") == "AAA"


def test_resolve_missing_skips_everything_already_known(tmp_path):
    session = _Session([{"data": [{"ticker": "BBB", "exchCode": "US"}]}])
    client = OpenFigiClient(session=session)
    with CusipCache(tmp_path / "c.db") as cache:
        cache.record([Resolution("111111111", "AAA", None),
                      Resolution("999999999", None, None)])

        stats = resolve_missing(cache, client,
                                ["111111111", "999999999", "222222222"])

    assert stats["asked"] == 1
    assert session.requests[0][1] == [{"idType": "ID_CUSIP",
                                       "idValue": "222222222"}]


def test_a_time_budget_leaves_the_rest_for_the_next_run(tmp_path):
    client = _client([{"data": []}] * BATCH_UNAUTHENTICATED)
    with CusipCache(tmp_path / "c.db") as cache:
        stats = resolve_missing(cache, client, [f"{i:09d}" for i in range(50)],
                                deadline=-1.0)

    assert stats["asked"] == 0
    assert stats["remaining"] == 50


def test_counts_report_hits_separately_from_questions_asked(tmp_path):
    with CusipCache(tmp_path / "c.db") as cache:
        cache.record([Resolution("111111111", "AAA", None),
                      Resolution("999999999", None, None)])

        assert cache.counts() == (2, 1)


# --- the join reads it ------------------------------------------------------

def test_an_unresolved_cusip_is_counted_and_reported_never_guessed():
    """Fuzzy issuer-name matching would attach one company's disclosures to
    another's returns, and nothing downstream would say so."""
    from datetime import UTC, datetime, timedelta

    from tradezbotz.research.joins import HoldingsJoin

    class _Store:
        def as_of(self, when, kind=None):
            if kind != "institutional_holding":
                return []
            return [{"symbol": None, "observed_at": "2020-01-02T00:00:00+00:00",
                     "payload": {"cusip": "037833100"}}]

    join = HoldingsJoin(_Store(), cusip_map={})
    join._index()

    assert join.unmapped_cusips == {"037833100"}
    assert "run `resolve-cusips`" in join.summary()


def test_a_resolved_cusip_makes_the_position_visible():
    from tradezbotz.research.joins import HoldingsJoin

    class _Store:
        def as_of(self, when, kind=None):
            if kind != "institutional_holding":
                return []
            return [{"symbol": None, "observed_at": "2020-01-02T00:00:00+00:00",
                     "payload": {"cusip": "037833100"}}]

    join = HoldingsJoin(_Store(), cusip_map={"037833100": "AAPL"})

    assert "AAPL" in join._index()
    assert join.unmapped_cusips == set()
