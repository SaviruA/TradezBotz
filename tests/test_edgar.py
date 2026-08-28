"""Tests for Form 4 parsing and dissemination-time handling."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from tradezbotz.research.edgar import ET_TZ, _disseminated_at, parse_form4

FILING = """\
ACCESSION NUMBER:		0001234567-25-000123
CONFORMED SUBMISSION TYPE:	4
<ACCEPTANCE-DATETIME>20250310163000
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001111111</rptOwnerCik>
      <rptOwnerName>DOE JANE</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2025-03-07</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>150.25</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def test_parses_open_market_purchase():
    txns = parse_form4(FILING)
    assert len(txns) == 1
    t = txns[0]

    assert t.symbol == "AAPL"
    assert t.owner_name == "DOE JANE"
    assert t.is_officer is True
    assert t.is_director is False
    assert t.officer_title == "Chief Executive Officer"
    assert t.transaction_code == "P"
    assert t.shares == 5000
    assert t.price_per_share == 150.25
    assert t.notional == pytest.approx(751_250.0)
    assert t.is_open_market_buy is True


def test_transaction_date_and_dissemination_are_distinct():
    """The trade was on the 7th; it became public on the 10th. Three days of
    price action belong to nobody and must never reach the backtest."""
    t = parse_form4(FILING)[0]

    assert t.transaction_date == date(2025, 3, 7)
    assert t.disseminated_at.astimezone(ET_TZ).date() == date(2025, 3, 10)


def test_event_uses_dissemination_as_observed_at():
    event = parse_form4(FILING)[0].to_event()

    assert event.observed_at > event.occurred_at
    assert event.observed_at.astimezone(ET_TZ).date() == date(2025, 3, 10)
    assert event.symbol == "AAPL"


def test_accepted_before_cutoff_keeps_same_day():
    accepted = datetime(2025, 3, 10, 16, 30, tzinfo=ET_TZ)  # Monday 4:30pm
    assert _disseminated_at(accepted).astimezone(ET_TZ).date() == date(2025, 3, 10)


def test_accepted_after_cutoff_rolls_to_next_morning():
    accepted = datetime(2025, 3, 10, 22, 30, tzinfo=ET_TZ)  # Monday 10:30pm
    out = _disseminated_at(accepted).astimezone(ET_TZ)

    assert out.date() == date(2025, 3, 11)
    assert out.hour == 6


def test_friday_night_filing_rolls_past_the_weekend():
    accepted = datetime(2025, 3, 7, 23, 0, tzinfo=ET_TZ)  # Friday 11pm
    out = _disseminated_at(accepted).astimezone(ET_TZ)

    assert out.date() == date(2025, 3, 10), "must skip Sat/Sun"


def test_malformed_filing_yields_nothing_rather_than_guessing():
    assert parse_form4("garbage, no xml here") == []
    assert parse_form4(FILING.replace("<ACCEPTANCE-DATETIME>20250310163000", "")) == []


def test_filing_without_ticker_is_skipped():
    """Many filers have no listed ticker; those rows are unusable, not zero."""
    no_ticker = FILING.replace(
        "<issuerTradingSymbol>AAPL</issuerTradingSymbol>",
        "<issuerTradingSymbol></issuerTradingSymbol>",
    )
    assert parse_form4(no_ticker) == []


# --- daily index availability ------------------------------------------------

class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(self.status_code)


class _Session:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.headers = {}
        self.urls: list[str] = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return self._responses.pop(0) if self._responses else _Resp(200, "")


def _client(monkeypatch, *responses):
    from tradezbotz.research.edgar import EdgarClient

    monkeypatch.setenv("SEC_USER_AGENT", "test harness dev@example.com")
    return EdgarClient(session=_Session(*responses))


def test_unpublished_index_is_skipped_once_access_is_verified(monkeypatch):
    """EDGAR answers 403 -- not 404 -- for a daily index that does not exist
    yet. Yesterday's index routinely 403s, and that must not be fatal."""
    c = _client(monkeypatch, _Resp(200, "ok"), _Resp(403))
    c.verify_access()

    assert c.daily_form4_filings(date(2026, 8, 28)) == []


def test_403_before_verification_stays_fatal(monkeypatch):
    """A genuinely rejected User-Agent must not be silently swallowed as an
    empty day -- that would report zero filings for all of history."""
    from tradezbotz.research.edgar import EdgarError

    c = _client(monkeypatch, _Resp(403))

    with pytest.raises(EdgarError, match="403"):
        c.daily_form4_filings(date(2026, 8, 28))


def test_verify_access_raises_on_rejected_user_agent(monkeypatch):
    from tradezbotz.research.edgar import EdgarError

    c = _client(monkeypatch, _Resp(403))

    with pytest.raises(EdgarError):
        c.verify_access()


def test_missing_index_404_still_returns_empty(monkeypatch):
    c = _client(monkeypatch, _Resp(404))

    assert c.daily_form4_filings(date(2026, 8, 29)) == []
