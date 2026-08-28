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


# --- occurred_at vs observed_at ----------------------------------------------

def test_same_day_filing_does_not_violate_ordering():
    """Regression: insiders often trade and file the same day. A fixed
    end-of-day placeholder for the trade time then lands AFTER dissemination
    and trips the event store's ordering invariant, which killed every day of
    a real ingest run."""
    from tradezbotz.research.edgar import _occurred_at

    # Traded 2026-08-27, filed the same day at 14:00 ET.
    disseminated = datetime(2026, 8, 27, 14, 0, tzinfo=ET_TZ)
    occurred = _occurred_at(date(2026, 8, 27), disseminated)

    assert occurred <= disseminated


def test_occurred_at_uses_market_open_when_filing_is_later():
    from tradezbotz.research.edgar import _occurred_at

    disseminated = datetime(2026, 8, 31, 6, 0, tzinfo=ET_TZ)
    occurred = _occurred_at(date(2026, 8, 27), disseminated).astimezone(ET_TZ)

    assert occurred.date() == date(2026, 8, 27)
    assert (occurred.hour, occurred.minute) == (9, 30)


def test_same_day_premarket_filing_clamps_to_dissemination():
    """Pathological but real: a filing timestamped before that day's open."""
    from tradezbotz.research.edgar import _occurred_at

    disseminated = datetime(2026, 8, 27, 7, 0, tzinfo=ET_TZ)

    assert _occurred_at(date(2026, 8, 27), disseminated) == disseminated


def test_event_construction_survives_same_day_filings():
    """End to end: the whole point is that to_event() no longer raises."""
    same_day = FILING.replace("<value>2025-03-07</value>", "<value>2025-03-10</value>")
    txns = parse_form4(same_day)

    assert len(txns) == 1
    event = txns[0].to_event()          # must not raise
    assert event.occurred_at <= event.observed_at


# --- transaction identity ----------------------------------------------------

TWO_IDENTICAL_LINES = FILING.replace(
    "  </nonDerivativeTable>",
    """    <nonDerivativeTransaction>
      <transactionDate><value>2025-03-07</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>150.25</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>""",
)


def test_identical_lines_in_one_filing_stay_distinct():
    """Regression: real Form 4 filings contain lines identical on owner, date,
    code, shares and price -- two equal tax withholdings, two equal conversions.
    Without a line index they collide on external_id and all but one are
    silently dropped. A real day lost 920 of 1740 transactions to this."""
    txns = parse_form4(TWO_IDENTICAL_LINES)

    assert len(txns) == 2
    ids = {t.to_event().external_id for t in txns}
    assert len(ids) == 2, "identical transaction lines must remain distinct events"


def test_line_index_follows_document_order():
    txns = parse_form4(TWO_IDENTICAL_LINES)

    assert [t.line_index for t in txns] == [0, 1]


def test_reparsing_yields_stable_ids():
    """Ingestion is idempotent only if identity survives a re-parse."""
    first = [t.to_event().external_id for t in parse_form4(TWO_IDENTICAL_LINES)]
    second = [t.to_event().external_id for t in parse_form4(TWO_IDENTICAL_LINES)]

    assert first == second


# --- daily index deduplication -----------------------------------------------

IDX = """Form Type  Company Name    CIK  Date Filed  File Name
--------------------------------------------------------------
4          ACME INC        111  2026-08-27  edgar/data/111/0001104659-26-102532.txt
4          DOE JANE        222  2026-08-27  edgar/data/222/0001104659-26-102532.txt
4          ROE JOHN        333  2026-08-27  edgar/data/333/0001104659-26-102532.txt
4          OTHER CORP      444  2026-08-27  edgar/data/444/0001193125-26-371778.txt
"""


def test_one_filing_is_fetched_once_not_once_per_cik(monkeypatch):
    """EDGAR indexes a Form 4 once per involved CIK -- issuer plus each
    reporting owner -- as different URLs for the SAME document. On 2026-08-27
    that turned 425 filings into 870 rows, doubling the request count."""
    c = _client(monkeypatch, _Resp(200, "ok"), _Resp(200, IDX))
    c.verify_access()

    rows = c.daily_form4_filings(date(2026, 8, 27))

    assert len(rows) == 2, "three rows share one accession and collapse to one"
    assert {r[1].split("/")[-1] for r in rows} == {
        "0001104659-26-102532.txt",
        "0001193125-26-371778.txt",
    }


def test_dedup_keeps_the_first_listed_path(monkeypatch):
    c = _client(monkeypatch, _Resp(200, "ok"), _Resp(200, IDX))
    c.verify_access()

    rows = c.daily_form4_filings(date(2026, 8, 27))

    assert rows[0] == ("111", "edgar/data/111/0001104659-26-102532.txt")
