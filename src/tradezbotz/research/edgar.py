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

#: Accession number as it appears in an EDGAR document path.
ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")

MAX_REQUESTS_PER_SECOND = 8  # below the SEC's 10/s ceiling, with headroom

#: `issuerTradingSymbol` is free text and filers treat it as such. Measured over
#: 4,740 distinct symbols in the store, 50 (1.1%) were unusable as written and
#: carried 1,268 events (0.69%). They failed the price backfill with a 400 from
#: the vendor and were parked as "failures", which read as absent data when it
#: was actually absent parsing.
#:
#: The loss is small but not random: the dual-class spellings below are all
#: established mid and large caps -- SIRI, LEN, HEI, GEF, PARA, NYCB -- because
#: a company with two share classes has two symbols to write in one field. So
#: the malformed set skews toward exactly the larger, more liquid names, which
#: is the opposite of the bias we can afford in a microcap study.
#:
#: Observed families:
#:     NASDAQ:DHC  NYSE: KRC  NTIP-NYSE  NYSE/TRN   exchange prefix or suffix
#:     (SIRI)  (CALX)  "'LTRX"  CHEA]  QSAM)        wrapped in punctuation
#:     GEF, GEF-B   LEN, LEN.B   MOGA/MOGB          two classes in one field
#:     -   --   N/A   1314152   N O G               junk
_EXCHANGE_PREFIX = re.compile(
    r"^\s*(?:NYSE|NASDAQ|NASD|AMEX|NYSEAMERICAN|NYSE\s*MKT|OTC|CBOE|ASX|TSX)"
    r"\s*[:\-/]\s*", re.I)
_EXCHANGE_SUFFIX = re.compile(
    r"\s*[\-/]\s*(?:NYSE|NASDAQ|NASD|AMEX|OTC)\s*$", re.I)
#: Where a filer wrote two classes, the first is the one the filing is about
#: often enough to be the only defensible choice, and picking arbitrarily is
#: better than dropping the issuer entirely.
_SPLIT_ON = re.compile(r"[,;/]|\s+AND\s+|\s{2,}", re.I)
#: Class suffixes on US listings are one or two characters -- .A .B .U .W .WS
#: .PR .RT and so on. Allowing more was too loose: "AB-LEND" and "M-6697" both
#: pass a 4-character rule while being nothing of the kind, and a wrong ticker
#: attaches an insider's trade to another company's price series. Checked
#: against the store: exactly two symbols carry a longer suffix, both junk, and
#: neither has price data.
_VALID_SYMBOL = re.compile(r"^[A-Z][A-Z0-9]{0,6}(?:[.\-][A-Z0-9]{1,2})?$")
#: Placeholders filers use for "none". Checked after normalisation so that a
#: real ticker is never caught by them.
_PLACEHOLDERS = {"NA", "N/A", "NONE", "-", "--", "---", "N.A.", "TBD", ""}


def normalise_symbol(raw: str | None) -> str:
    """Extract a usable ticker from the free-text issuer symbol field.

    Returns "" when nothing usable is present, which callers already treat as
    "skip this filing". Deliberately conservative: a wrong ticker attaches an
    insider's trade to another company's price series, which is far worse than
    dropping the filing, so anything that does not end up looking like a ticker
    is discarded rather than guessed at.
    """
    if not raw:
        return ""
    text = raw.strip().upper()
    if text in _PLACEHOLDERS:
        return ""

    # Punctuation wrappers first: "(NYSE:FBC)" has to lose its brackets before
    # the exchange prefix is visible.
    text = text.strip("\"'()[]{}<> \t")
    text = _EXCHANGE_PREFIX.sub("", text)
    text = _EXCHANGE_SUFFIX.sub("", text)
    text = text.strip("\"'()[]{}<> \t")

    # Two classes in one field: take the first and let the rest go. Splitting
    # into two events would double-count one filing.
    first = _SPLIT_ON.split(text)[0].strip()
    first = first.strip("\"'()[]{}<> \t")
    # A trailing bare space-separated class ("GEF B", "SNV PR E") -- keep the
    # root, which is the tradeable line.
    if " " in first:
        head = first.split()[0]
        # "N O G" is a spaced-out ticker, not a root plus a class. Rejoining
        # single letters recovers it; anything else keeps the head.
        parts = first.split()
        if all(len(p) == 1 for p in parts) and 2 <= len(parts) <= 6:
            first = "".join(parts)
        else:
            first = head

    if first in _PLACEHOLDERS or not _VALID_SYMBOL.match(first):
        return ""
    return first

#: Open-market purchase. The only transaction code with consistent academic
#: support as an informative signal -- grants (A), option exercises (M) and
#: tax withholding (F) are compensation mechanics, not conviction.
CODE_OPEN_MARKET_BUY = "P"
CODE_OPEN_MARKET_SELL = "S"

#: Regular-session open. Form 4 reports only a transaction DATE, never a time,
#: so this stands in for the unknown intraday moment -- the earliest a trade
#: could plausibly have happened that day.
MARKET_OPEN_ET = dtime(9, 30)


def _occurred_at(transaction_date: date, disseminated_at: datetime) -> datetime:
    """Best estimate of when the trade happened, never later than when it
    became public.

    Same-day filings are common: an insider trades at 10:00 and the filing agent
    submits by 14:00. Any fixed end-of-day placeholder would then sit *after*
    dissemination and trip the event store's ordering invariant. Clamping keeps
    the estimate honest without weakening the invariant -- which is doing its
    job here, since it caught exactly this bug.
    """
    approx = datetime.combine(
        transaction_date, MARKET_OPEN_ET, tzinfo=ET_TZ
    ).astimezone(timezone.utc)
    return min(approx, disseminated_at)


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
    #: Ordinal position within the filing's nonDerivativeTable. Real filings
    #: contain lines identical on every other field -- two 27-share tax
    #: withholdings at one price, two equal conversions -- so without this
    #: they collide on external_id and all but one are silently dropped.
    #: XML element order is stable, so re-parsing yields the same index and
    #: ingestion stays idempotent.
    line_index: int = 0

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
            external_id=f"{self.accession}:{self.line_index}:{self.owner_cik}",
            kind="insider_transaction",
            symbol=self.symbol,
            observed_at=self.disseminated_at,
            occurred_at=_occurred_at(self.transaction_date, self.disseminated_at),
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

    def verify_access(self) -> None:
        """Confirm our User-Agent is accepted before interpreting any 403s.

        EDGAR answers 403 -- not 404 -- for a daily index that does not exist
        yet, which is indistinguishable from a rejected User-Agent by status
        code alone. Checking a known-stable URL once up front removes the
        ambiguity: after this passes, a 403 on a daily index means "not
        published", and `daily_form4_filings` may safely skip it.
        """
        self._get("https://www.sec.gov/Archives/edgar/daily-index/")
        self._access_verified = True

    def daily_form4_filings(self, day: date) -> list[tuple[str, str]]:
        """Return (cik, document_path) for every Form 4 filed on `day`.

        Weekends, holidays, and days whose index has not been published yet all
        return an empty list. Today and often yesterday fall in that last group:
        EDGAR publishes the daily index on a lag.
        """
        url = DAILY_INDEX.format(
            year=day.year, qtr=(day.month - 1) // 3 + 1, ymd=day.strftime("%Y%m%d")
        )
        try:
            body = self._get(url)
        except FileNotFoundError:
            return []
        except EdgarError:
            if getattr(self, "_access_verified", False):
                return []  # index not published for this day
            raise  # credentials never proven; a real 403 must stay loud

        # A Form 4 involves an issuer and one or more reporting owners, and EDGAR
        # indexes it once per CIK -- same document, different URL
        # (edgar/data/<CIK>/<accession>.txt). Fetching every row would download
        # each filing roughly twice: on 2026-08-27, 870 rows carried just 425
        # distinct accessions. Deduplicating here halves the request count, which
        # at 8 req/s is hours off a multi-year backfill.
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for line in body.splitlines():
            # Columns: Form Type | Company Name | CIK | Date Filed | File Name
            if not line.startswith("4 "):
                continue
            parts = [p for p in re.split(r"\s{2,}", line.strip()) if p]
            if len(parts) < 5 or parts[0] != "4":
                continue
            cik, path = parts[2], parts[4]
            match = ACCESSION_RE.search(path)
            key = match.group(1) if match else path
            if key in seen:
                continue
            seen.add(key)
            out.append((cik, path))
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

    symbol = normalise_symbol(_text(doc.find("issuer/issuerTradingSymbol")))
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
    for idx, txn in enumerate(
        doc.findall("nonDerivativeTable/nonDerivativeTransaction")
    ):
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
                line_index=idx,
            )
        )
    return results


def ingest_day(client: EdgarClient, day: date) -> Iterator[InsiderTransaction]:
    for _cik, path in client.daily_form4_filings(day):
        try:
            yield from parse_form4(client.fetch_filing(path))
        except (FileNotFoundError, EdgarError):
            continue
