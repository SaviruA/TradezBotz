"""SEC EDGAR Form 4 ingestion, with point-in-time timestamps.

Form 4 discloses an insider transaction within two business days of the trade.
Two distinct timestamps therefore matter and must never be conflated:

  occurred_at  -- <transactionDate> in the XML. When the insider traded.
                  Unknowable to the public at that moment.
  observed_at  -- when the filing was disseminated by EDGAR and became
                  actionable. This is the ONLY timestamp a backtest may key on.

Filings accepted after the dissemination cutoff are rolled to the next business
morning; see `_disseminated_at`.

SEC access rules (https://www.sec.gov/os/webmaster-faq#developers):
  * a descriptive User-Agent with contact info is REQUIRED
  * requests are capped at 10/second
Both are enforced below.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Iterator
from zoneinfo import ZoneInfo

import requests

from .eventstore import Event

EDGAR_BASE = "https://www.sec.gov/Archives"
DAILY_INDEX = EDGAR_BASE + "/edgar/daily-index/{year}/QTR{qtr}/form.{ymd}.idx"
ET_TZ = ZoneInfo("America/New_York")

#: Form 4 may be transmitted until 22:00 ET and still receive that day's filing
#: date; ordinary filings cut off at 17:30 ET.
FORM4_CUTOFF = dtime(22, 0)

MAX_REQUESTS_PER_SECOND = 8  # below the SEC's 10/s ceiling, with headroom

#: Open-market purchase. The only transaction code with consistent academic
#: support as an informative signal -- grants (A), option exercises (M) and
#: tax withholding (F) are compensation mechanics, not conviction.
CODE_OPEN_MARKET_BUY = "P"
CODE_OPEN_MARKET_SELL = "S"


class EdgarError(RuntimeError):
    pass


def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise EdgarError(
            "SEC_USER_AGENT must be set to a contact string including an email, "
            "e.g. 'TradezBotz research you@example.com'. The SEC requires this "
            "and blocks clients that omit it."
        )
    return ua


@dataclass(frozen=True)
class InsiderTransaction:
    accession: str
    issuer_cik: str
    symbol: str
    owner_name: str
    owner_cik: str
    is_officer: bool
    is_director: bool
    is_ten_percent: bool
    officer_title: str | None
    transaction_code: str
    shares: float
    price_per_share: float | None
    acquired_disposed: str
    transaction_date: date
    disseminated_at: datetime

    @property
    def notional(self) -> float | None:
        if self.price_per_share is None:
            return None
        return self.shares * self.price_per_share

    @property
    def is_open_market_buy(self) -> bool:
        return (
            self.transaction_code == CODE_OPEN_MARKET_BUY
            and self.acquired_disposed == "A"
        )

    def to_event(self) -> Event:
        return Event(
            source="sec_form4",
            external_id=(
                f"{self.accession}:{self.owner_cik}:{self.transaction_date}:"
                f"{self.transaction_code}:{self.shares:g}"
            ),
            kind="insider_transaction",
            symbol=self.symbol,
            observed_at=self.disseminated_at,
            occurred_at=datetime.combine(
                self.transaction_date, dtime(21, 0), tzinfo=timezone.utc
            ),
            payload={
                "accession": self.accession,
                "issuer_cik": self.issuer_cik,
                "owner_name": self.owner_name,
                "owner_cik": self.owner_cik,
                "is_officer": self.is_officer,
                "is_director": self.is_director,
                "is_ten_percent": self.is_ten_percent,
                "officer_title": self.officer_title,
                "transaction_code": self.transaction_code,
                "shares": self.shares,
                "price_per_share": self.price_per_share,
                "acquired_disposed": self.acquired_disposed,
                "transaction_date": self.transaction_date.isoformat(),
                "notional": self.notional,
            },
        )


class EdgarClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}
        )
        self._last_request = 0.0

    def _get(self, url: str) -> str:
        elapsed = time.monotonic() - self._last_request
        min_gap = 1.0 / MAX_REQUESTS_PER_SECOND
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        resp = self._session.get(url, timeout=30)
        self._last_request = time.monotonic()
        if resp.status_code == 404:
            raise FileNotFoundError(url)
        if resp.status_code == 403:
            raise EdgarError(
                f"SEC returned 403 for {url}. Usually a missing or rejected "
                "User-Agent, or exceeding the rate limit."
            )
        resp.raise_for_status()
        return resp.text

    def daily_form4_filings(self, day: date) -> list[tuple[str, str]]:
        """Return (cik, document_path) for every Form 4 filed on `day`.

        Weekends and holidays have no index file; those return an empty list.
        """
        url = DAILY_INDEX.format(
            year=day.year, qtr=(day.month - 1) // 3 + 1, ymd=day.strftime("%Y%m%d")
        )
        try:
            body = self._get(url)
        except FileNotFoundError:
            return []

        out: list[tuple[str, str]] = []
        for line in body.splitlines():
            # Columns: Form Type | Company Name | CIK | Date Filed | File Name
            if not line.startswith("4 "):
                continue
            parts = [p for p in re.split(r"\s{2,}", line.strip()) if p]
            if len(parts) < 5 or parts[0] != "4":
                continue
            out.append((parts[2], parts[4]))
        return out

    def fetch_filing(self, document_path: str) -> str:
        return self._get(f"{EDGAR_BASE}/{document_path.lstrip('/')}")


def _disseminated_at(accepted: datetime) -> datetime:
    """Map an EDGAR acceptance time to when the filing was actually public.

    Form 4 accepted at or before 22:00 ET keeps that day. Anything later is not
    disseminated until 06:00 ET the next business day. Getting this wrong grants
    the backtest up to a full session of free lookahead.
    """
    accepted = accepted.astimezone(ET_TZ)
    if accepted.time() <= FORM4_CUTOFF:
        return accepted.astimezone(timezone.utc)
    nxt = accepted.date() + timedelta(days=1)
    while nxt.weekday() >= 5:  # Sat/Sun; market holidays handled downstream
        nxt += timedelta(days=1)
    return datetime.combine(nxt, dtime(6, 0), tzinfo=ET_TZ).astimezone(timezone.utc)


def _text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    value = node.findtext("value")
    text = value if value is not None else node.text
    return text.strip() if text else None


def _float(node: ET.Element | None) -> float | None:
    raw = _text(node)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_form4(raw: str) -> list[InsiderTransaction]:
    """Parse a raw EDGAR .txt submission into non-derivative transactions.

    Derivative transactions (options, RSUs) are deliberately excluded: they are
    dominated by compensation mechanics and dilute the open-market-purchase
    signal we care about.
    """
    accession_m = re.search(r"ACCESSION NUMBER:\s*(\S+)", raw)
    accepted_m = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", raw)
    xml_m = re.search(r"(<ownershipDocument>.*?</ownershipDocument>)", raw, re.S)
    if not (accession_m and accepted_m and xml_m):
        return []

    accepted = datetime.strptime(accepted_m.group(1), "%Y%m%d%H%M%S").replace(
        tzinfo=ET_TZ
    )
    disseminated = _disseminated_at(accepted)
    accession = accession_m.group(1)

    try:
        doc = ET.fromstring(xml_m.group(1))
    except ET.ParseError:
        return []

    symbol = (_text(doc.find("issuer/issuerTradingSymbol")) or "").upper()
    issuer_cik = _text(doc.find("issuer/issuerCik")) or ""
    if not symbol:
        return []

    owner = doc.find("reportingOwner")
    owner_name = (
        _text(owner.find("reportingOwnerId/rptOwnerName")) if owner is not None else None
    )
    owner_cik = (
        _text(owner.find("reportingOwnerId/rptOwnerCik")) if owner is not None else None
    )
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None

    def flag(name: str) -> bool:
        return (_text(rel.find(name)) if rel is not None else None) in ("1", "true")

    results: list[InsiderTransaction] = []
    for txn in doc.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(txn.find("transactionCoding/transactionCode"))
        tdate = _text(txn.find("transactionDate"))
        shares = _float(txn.find("transactionAmounts/transactionShares"))
        if not (code and tdate and shares):
            continue
        results.append(
            InsiderTransaction(
                accession=accession,
                issuer_cik=issuer_cik,
                symbol=symbol,
                owner_name=owner_name or "UNKNOWN",
                owner_cik=owner_cik or "UNKNOWN",
                is_officer=flag("isOfficer"),
                is_director=flag("isDirector"),
                is_ten_percent=flag("isTenPercentOwner"),
                officer_title=(
                    _text(rel.find("officerTitle")) if rel is not None else None
                ),
                transaction_code=code,
                shares=shares,
                price_per_share=_float(
                    txn.find("transactionAmounts/transactionPricePerShare")
                ),
                acquired_disposed=_text(
                    txn.find("transactionAmounts/transactionAcquiredDisposedCode")
                )
                or "",
                transaction_date=date.fromisoformat(tdate),
                disseminated_at=disseminated,
            )
        )
    return results


def ingest_day(client: EdgarClient, day: date) -> Iterator[InsiderTransaction]:
    for _cik, path in client.daily_form4_filings(day):
        try:
            yield from parse_form4(client.fetch_filing(path))
        except (FileNotFoundError, EdgarError):
            continue
