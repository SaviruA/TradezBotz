"""EDGAR's status codes, and the one that killed the holdings ingest.

`daily_filings` guards on FileNotFoundError (404) and EdgarError (403). The
design note says EDGAR answers 403 for a daily index that does not exist yet.
It answered **503**, `raise_for_status()` turned that into a bare
requests.HTTPError, and the bare type walked straight through both guards.

The blast radius was total and silent: `days.reverse()` put today first, so the
crash landed on the step's first request every night, 13F/13D/congressional
disclosures were never ingested at all, and the only symptom was downstream --
the joins reporting "0 symbols carry a disclosure", which reads as a fact about
the market rather than a fact about the fetch.

So the invariant under test is narrow and absolute: every failure leaves `_get`
as FileNotFoundError or EdgarError, never as anything else.
"""

from __future__ import annotations

import pytest
import requests

from tradezbotz.research.edgar import EdgarClient, EdgarError


class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


class _Session:
    """Serves a queue of responses, then repeats the last one forever."""

    def __init__(self, *statuses):
        self.queue = list(statuses)
        self.headers = {}
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        status = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return _Resp(status, text="body")


@pytest.fixture(autouse=True)
def _harness(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "test harness dev@example.com")
    monkeypatch.setattr("tradezbotz.research.edgar.time.sleep", lambda s: None)


# --- the regression ---------------------------------------------------------

def test_a_503_never_escapes_as_a_bare_http_error():
    """The exact crash. A bare HTTPError walks through every caller's guard."""
    client = EdgarClient(session=_Session(503))

    with pytest.raises(EdgarError):
        client._get("https://sec.gov/daily-index/today.idx")


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_every_server_error_becomes_an_edgar_error(status):
    client = EdgarClient(session=_Session(status))

    with pytest.raises(EdgarError):
        client._get("https://sec.gov/x")


def test_the_message_names_the_likely_cause():
    """A 503 on a daily index means "not published yet" far more often than it
    means anything else, and the operator should not have to rediscover that."""
    client = EdgarClient(session=_Session(503))

    with pytest.raises(EdgarError, match="not published yet"):
        client._get("https://sec.gov/daily-index/today.idx")


# --- retry, because 5xx is genuinely ambiguous ------------------------------

def test_a_transient_server_error_is_retried_and_succeeds():
    """SEC returns 5xx under load as well as for a missing index. Surrendering
    on the first one would drop real filing days during a busy minute."""
    session = _Session(503, 503, 200)
    client = EdgarClient(session=session)

    assert client._get("https://sec.gov/x") == "body"
    assert session.calls == 3


def test_retries_are_bounded():
    session = _Session(503)
    client = EdgarClient(session=session)

    with pytest.raises(EdgarError):
        client._get("https://sec.gov/x", retries=2)

    assert session.calls == 3


def test_a_404_is_still_file_not_found_and_is_not_retried():
    """A missing document is not ambiguous, so retrying it only burns rate."""
    session = _Session(404)
    client = EdgarClient(session=session)

    with pytest.raises(FileNotFoundError):
        client._get("https://sec.gov/x")

    assert session.calls == 1


def test_a_403_is_still_an_edgar_error_and_is_not_retried():
    session = _Session(403)
    client = EdgarClient(session=session)

    with pytest.raises(EdgarError, match="403"):
        client._get("https://sec.gov/x")

    assert session.calls == 1


def test_a_success_does_not_retry():
    session = _Session(200)
    client = EdgarClient(session=session)

    assert client._get("https://sec.gov/x") == "body"
    assert session.calls == 1


# --- the guard downstream actually holds now --------------------------------

def test_daily_filings_skips_an_unpublished_index_rather_than_raising():
    from tradezbotz.research.holdings import daily_filings

    client = EdgarClient(session=_Session(503))
    client._access_verified = True

    assert daily_filings(client, __import__("datetime").date(2026, 9, 3), ["13F-HR"]) == []
