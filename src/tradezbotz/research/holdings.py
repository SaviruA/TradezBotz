"""Who else is buying: 13F holdings, 13D/G stakes, and congressional trades.

Three disclosure regimes, one idea -- follow people who may know something --
but they differ enough that conflating them would be a mistake.

  13F      quarterly holdings from managers over $100M. **45-day lag.** Not
           transactions: a position list. The signal is the *change* between
           quarters, and a new position is a different event from an add.
  13D/G    stakes above 5%. **5 business days** for 13D, so far fresher than
           13F, and 13D specifically signals intent to influence.
  House    congressional trades under the STOCK Act, disclosed within 45 days.

**Operating companies file 13F too**, which is the part usually missed. NVIDIA
has 11 on file and Alphabet 13 -- so "NVIDIA took a stake in X" is a disclosed,
machine-readable quarterly event, not something to learn from a press cycle.
Apple and Tesla file none, because they hold no large public-equity portfolio;
they appear in 13D/G instead. Covering both forms is therefore not redundancy,
it is the only way to see the whole picture.

**On following Congress.** Ziobrowski (2004, 2011) reported Senate portfolios
beating the market by ~12%/yr and House by ~6%. Later work reversed it: Eggers
& Hainmueller (2013) and Belmont (2022) find members would have done better in
an index fund. The methodological detail is what matters here -- Ziobrowski's
significance appears only in the *aggregate trade-weighted* portfolio, which
loads on a few large trades by a few members; weight each member equally and the
edge is indistinguishable from noise.

That is an argument for ranking rather than against the strategy, and it is why
`rank_by_trailing_return` exists. But ranking on past performance is itself a
data-mining risk, so `persistence` is the check that has to pass first: rank on
an early window, measure on a later one. Without it we would be selecting the
lucky and calling them skilled -- which is precisely the error the replications
diagnosed.

**Senate is deliberately absent.** efdsearch.senate.gov returns 403 to a plain
request; it is CSRF-session-gated. House publishes a bulk zip and needs no key,
so House is built and Senate is left for a decision about paying for access.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence

from .edgar import ACCESSION_RE, DAILY_INDEX, EdgarClient, EdgarError, _disseminated_at
from .eventstore import Event

SOURCE_13F = "sec_13f"
SOURCE_13DG = "sec_13dg"
SOURCE_HOUSE = "house_ptr"

KIND_HOLDING = "institutional_holding"
KIND_STAKE = "beneficial_stake"
KIND_CONGRESS = "congress_trade"

HOUSE_BULK = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"

#: 13F is filed within 45 days of quarter end, so a holding is stale by
#: construction. Recorded as a constant because any strategy built on it has to
#: reason about the lag explicitly rather than forget it exists.
THIRTEEN_F_LAG_DAYS = 45

#: 13D is due within 5 business days of crossing 5%; 13G is a passive-intent
#: alternative with a longer deadline. The distinction is the signal: 13D means
#: intent to influence, 13G means a passive holder.
THIRTEEN_D_LAG_DAYS = 5

_CIK_RE = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)")
_ACCEPTED_RE = re.compile(r"ACCEPTANCE-DATETIME>\s*(\d{14})")
_COMPANY_RE = re.compile(r"COMPANY CONFORMED NAME:\s*(.+)")
_SUBJECT_RE = re.compile(r"SUBJECT COMPANY:(.*?)(?:FILED BY:|</SEC-HEADER>)", re.S)


class HoldingsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Holding:
    """One line of a 13F: a manager's position in one issuer."""

    cusip: str
    issuer: str
    shares: float
    value_usd: float

    def key(self) -> str:
        return self.cusip


@dataclass(frozen=True)
class Filing13F:
    accession: str
    filer_cik: str
    filer_name: str
    period_end: date
    observed_at: datetime
    holdings: tuple[Holding, ...]

    @property
    def total_value(self) -> float:
        return sum(h.value_usd for h in self.holdings)

    @property
    def lag_days(self) -> int:
        """How stale the position list already was when it became public."""
        return (self.observed_at.date() - self.period_end).days

    def to_events(self) -> Iterator[Event]:
        """One event per position.

        Per position rather than per filing: a strategy cares about a manager's
        stake in one issuer, and a single event carrying 900 holdings could not
        be joined to a symbol.
        """
        total = self.total_value
        for h in self.holdings:
            yield Event(
                source=SOURCE_13F,
                kind=KIND_HOLDING,
                external_id=f"{self.accession}:{h.cusip}",
                observed_at=self.observed_at,
                occurred_at=self.observed_at,
                payload={
                    "filer_cik": self.filer_cik,
                    "filer_name": self.filer_name,
                    "cusip": h.cusip,
                    "issuer": h.issuer,
                    "shares": h.shares,
                    "value_usd": h.value_usd,
                    "period_end": self.period_end.isoformat(),
                    "lag_days": self.lag_days,
                    # Concentration is the interesting part: a 15% position is a
                    # conviction statement, a 0.1% one is index-hugging.
                    "weight": (h.value_usd / total) if total > 0 else 0.0,
                },
            )


@dataclass(frozen=True)
class PositionChange:
    """What moved between two consecutive 13F filings by the same manager."""

    filer_cik: str
    cusip: str
    issuer: str
    kind: str            # new | add | trim | exit
    prior_shares: float
    shares: float
    weight: float
    observed_at: datetime

    @property
    def delta(self) -> float:
        return self.shares - self.prior_shares


def position_changes(previous: Filing13F | None,
                     current: Filing13F) -> list[PositionChange]:
    """Diff two filings into new / add / trim / exit.

    The diff is the signal, not the holding. A manager who has owned a name for
    six quarters is telling you nothing new this quarter; one who just opened a
    position is. `new` and `add` are kept distinct for the same reason -- the
    literature on cloning finds the edge concentrated in high-conviction *new*
    positions rather than in incremental adds.
    """
    prior = {h.cusip: h for h in (previous.holdings if previous else ())}
    now = {h.cusip: h for h in current.holdings}
    total = current.total_value
    out: list[PositionChange] = []

    for cusip, h in now.items():
        was = prior.get(cusip)
        prior_shares = was.shares if was else 0.0
        if was is None:
            kind = "new"
        elif h.shares > prior_shares:
            kind = "add"
        elif h.shares < prior_shares:
            kind = "trim"
        else:
            continue          # unchanged carries no information
        out.append(PositionChange(
            current.filer_cik, cusip, h.issuer, kind, prior_shares, h.shares,
            (h.value_usd / total) if total > 0 else 0.0, current.observed_at))

    for cusip, h in prior.items():
        if cusip not in now:
            out.append(PositionChange(
                current.filer_cik, cusip, h.issuer, "exit", h.shares, 0.0, 0.0,
                current.observed_at))
    return out


# --- parsing ---------------------------------------------------------------------

def _observed_at(raw: str) -> datetime | None:
    m = _ACCEPTED_RE.search(raw)
    if not m:
        return None
    return _disseminated_at(datetime.strptime(m.group(1), "%Y%m%d%H%M%S"))


def parse_13f(raw: str, path: str = "") -> Filing13F | None:
    """Extract holdings from a 13F-HR information table.

    Modern filings carry an XML information table; the tag namespace varies by
    filer agent, so tags are matched by local name rather than by a fixed
    prefix. Filings with no parseable table return None rather than an empty
    holdings list -- "filed nothing" and "we could not read it" are different
    facts and must not be conflated.
    """
    observed = _observed_at(raw)
    if observed is None:
        return None
    header_end = raw.find("</SEC-HEADER>")
    header = raw[:header_end] if header_end > 0 else raw[:4000]
    cik_m = _CIK_RE.search(header)
    name_m = _COMPANY_RE.search(header)

    period = None
    pm = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", header)
    if pm:
        period = datetime.strptime(pm.group(1), "%Y%m%d").date()

    holdings: list[Holding] = []
    for block in re.findall(r"<infoTable>(.*?)</infoTable>", raw, re.S | re.I):
        def tag(name: str) -> str | None:
            m = re.search(rf"<(?:\w+:)?{name}>(.*?)</(?:\w+:)?{name}>",
                          block, re.S | re.I)
            return m.group(1).strip() if m else None

        cusip = tag("cusip")
        if not cusip:
            continue
        try:
            value = float((tag("value") or "0").replace(",", ""))
            shares = float((tag("sshPrnamt") or "0").replace(",", ""))
        except ValueError:
            continue
        holdings.append(Holding(cusip.upper(), tag("nameOfIssuer") or "",
                                shares, value))

    if not holdings:
        return None
    return Filing13F(
        accession=(ACCESSION_RE.search(path or raw).group(1)
                   if ACCESSION_RE.search(path or raw) else ""),
        filer_cik=(cik_m.group(1).lstrip("0") if cik_m else ""),
        filer_name=(name_m.group(1).strip() if name_m else ""),
        period_end=period or observed.date(),
        observed_at=observed,
        holdings=tuple(holdings),
    )


@dataclass(frozen=True)
class Stake:
    """A 13D or 13G: someone crossed 5% of an issuer."""

    accession: str
    filer_cik: str
    filer_name: str
    subject_name: str
    form_type: str        # SC 13D, SC 13G, and their amendments
    observed_at: datetime

    @property
    def activist(self) -> bool:
        """13D means intent to influence; 13G is the passive alternative."""
        return self.form_type.upper().startswith("SC 13D")

    def to_event(self) -> Event:
        return Event(
            source=SOURCE_13DG,
            kind=KIND_STAKE,
            external_id=self.accession,
            observed_at=self.observed_at,
            occurred_at=self.observed_at,
            payload={
                "filer_cik": self.filer_cik,
                "filer_name": self.filer_name,
                "subject_name": self.subject_name,
                "form_type": self.form_type,
                "activist": self.activist,
                "amendment": "/A" in self.form_type,
            },
        )


def parse_13dg(raw: str, form_type: str, path: str = "") -> Stake | None:
    observed = _observed_at(raw)
    if observed is None:
        return None
    header_end = raw.find("</SEC-HEADER>")
    header = raw[:header_end] if header_end > 0 else raw[:6000]
    cik_m = _CIK_RE.search(header)
    names = _COMPANY_RE.findall(header)
    subject_block = _SUBJECT_RE.search(header)
    subject = ""
    if subject_block:
        sm = _COMPANY_RE.search(subject_block.group(1))
        subject = sm.group(1).strip() if sm else ""
    return Stake(
        accession=(ACCESSION_RE.search(path or raw).group(1)
                   if ACCESSION_RE.search(path or raw) else ""),
        filer_cik=(cik_m.group(1).lstrip("0") if cik_m else ""),
        filer_name=(names[-1].strip() if names else ""),
        subject_name=subject or (names[0].strip() if names else ""),
        form_type=form_type.upper(),
        observed_at=observed,
    )


# --- ranking ----------------------------------------------------------------------

@dataclass(frozen=True)
class FilerScore:
    filer_cik: str
    filer_name: str
    n_positions: int
    mean_return: float
    window_start: date
    window_end: date


def rank_by_trailing_return(scores: Sequence[FilerScore],
                            min_positions: int = 10) -> list[FilerScore]:
    """Rank filers by mean realised return, best first.

    `min_positions` is not a nicety. A filer with three disclosed positions can
    top a leaderboard on luck alone, and a ranking dominated by tiny samples is
    the mechanism by which Ziobrowski's aggregate result appeared while the
    equal-weighted one did not.
    """
    eligible = [s for s in scores if s.n_positions >= min_positions]
    return sorted(eligible, key=lambda s: s.mean_return, reverse=True)


def persistence(early: Sequence[FilerScore], late: Sequence[FilerScore],
                top_n: int = 10) -> dict[str, float]:
    """Do the filers who ranked well early still rank well later?

    **The check that has to pass before any ranking is trusted.** Selecting on
    past performance always produces a leaderboard; the question is whether the
    ordering carries into a period it was not fitted on. If it does not, the
    ranking is a record of who got lucky and following it is expensive noise.

    Returns the overlap between the two top-N sets, the rank correlation across
    filers present in both, and the later-period mean return of the early
    top-N -- which is the number a copy strategy would actually have earned.
    """
    early_rank = {s.filer_cik: i for i, s in enumerate(rank_by_trailing_return(early))}
    late_rank = {s.filer_cik: i for i, s in enumerate(rank_by_trailing_return(late))}
    shared = [c for c in early_rank if c in late_rank]

    early_top = [s.filer_cik for s in rank_by_trailing_return(early)[:top_n]]
    late_top = {s.filer_cik for s in rank_by_trailing_return(late)[:top_n]}
    late_by_cik = {s.filer_cik: s for s in late}

    followed = [late_by_cik[c].mean_return for c in early_top if c in late_by_cik]
    out = {
        "shared_filers": float(len(shared)),
        "top_n_overlap": float(len(set(early_top) & late_top)),
        "top_n_overlap_rate": (len(set(early_top) & late_top) / top_n) if top_n else 0.0,
        "followed_mean_return": (sum(followed) / len(followed)) if followed else 0.0,
        "all_mean_return": (sum(s.mean_return for s in late) / len(late)) if late else 0.0,
    }
    if len(shared) >= 3:
        from .clustering import rank_correlation
        out["rank_correlation"] = rank_correlation(
            [early_rank[c] for c in shared], [late_rank[c] for c in shared])
    else:
        out["rank_correlation"] = 0.0
    # The number that decides it: did following the early leaders beat the field?
    out["edge_over_field"] = out["followed_mean_return"] - out["all_mean_return"]
    return out


# --- fetching -----------------------------------------------------------------------

FORMS_13F = ("13F-HR", "13F-HR/A")
FORMS_13DG = ("SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A")


def daily_filings(client: EdgarClient, day: date,
                  forms: Sequence[str]) -> list[tuple[str, str, str]]:
    """(form, cik, path) for the requested forms filed on `day`."""
    url = DAILY_INDEX.format(
        year=day.year, qtr=(day.month - 1) // 3 + 1, ymd=day.strftime("%Y%m%d"))
    try:
        body = client._get(url)
    except FileNotFoundError:
        return []
    except EdgarError:
        if getattr(client, "_access_verified", False):
            return []
        raise

    wanted = {f.upper() for f in forms}
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for line in body.splitlines():
        parts = [p for p in re.split(r"\s{2,}", line.strip()) if p]
        if len(parts) < 5:
            continue
        form = parts[0].upper()
        if form not in wanted:
            continue
        cik, path = parts[2], parts[4]
        m = ACCESSION_RE.search(path)
        key = m.group(1) if m else path
        if key in seen:
            continue
        seen.add(key)
        out.append((form, cik, path))
    return out


def ingest_day(client: EdgarClient, day: date,
               forms: Sequence[str] = FORMS_13F + FORMS_13DG
               ) -> Iterator[Filing13F | Stake]:
    """Yield parsed 13F and 13D/G filings for one day.

    Parse failures are skipped rather than raised: one malformed information
    table must not end a multi-year backfill, the same policy as every other
    ingest path here.
    """
    for form, _cik, path in daily_filings(client, day, forms):
        try:
            raw = client.fetch_filing(path)
        except Exception:  # noqa: BLE001
            continue
        try:
            parsed = (parse_13f(raw, path) if form.startswith("13F")
                      else parse_13dg(raw, form, path))
        except Exception:  # noqa: BLE001
            continue
        if parsed is not None:
            yield parsed


# --- House periodic transaction reports ------------------------------------------
#
# The bulk zip is an INDEX, not the data: it lists who filed what and when, and
# the transactions live in per-filing PDFs. Those are generated rather than
# scanned so the text extracts cleanly -- but it is still text extraction, and
# the parser drops a line it cannot read rather than guessing at it. These are
# disclosures; a wrong ticker is worse than a missing one.
#
# Filing type P in the index is the periodic transaction report. The rest are
# annual reports, candidate filings, amendments and terminations, none of which
# carry transactions. In 2025 that was 515 PTRs out of 2,913 index rows.

HOUSE_PTR_PDF = ("https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/"
                 "{year}/{doc_id}.pdf")

HOUSE_PTR_TYPE = "P"

#: Ownership codes. Who holds the asset changes what the trade means: a
#: member's own account is a stronger signal than a dependent child's.
OWNER_CODES = {"SP": "spouse", "DC": "dependent_child", "JT": "joint"}

_TXN_RE = re.compile(
    r"^(?P<code>P|S|E)(?:\s*\(partial\))?\s+"
    r"(?P<txn>\d{2}/\d{2}/\d{4})\s+(?P<notified>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<amount>\$[\d,]+(?:\s*-\s*\$?[\d,]+)?)",
    re.IGNORECASE)

#: Ticker and asset-type code. Matched against *reflowed* text, never raw lines:
#: the PDF extractor wraps mid-record, and a real filing reads
#:
#:     SP Rollins, Inc. Common Stock (ROL)
#:     [ST]
#:     P 12/12/2024 01/08/2025 $15,001 -
#:     $50,000
#:
#: so requiring `(ROL) [ST]` on one physical line finds nothing, and reading the
#: amount from one line yields $15,001-$15,001 for a $15,001-$50,000 trade. Both
#: were live defects caught against real filings, which is why `_reflow` exists
#: rather than the parser working line by line.
_TICKER_RE = re.compile(r"\(([A-Z][A-Z.\-]{0,6})\)\s*\[(\w{2})\]")
_OWNER_RE = re.compile(r"^(SP|DC|JT)\s+")

#: The asset-type code, matched independently of the ticker.
#:
#: Separate because they do not always co-occur: a Treasury reads
#: `(91282CJP7) [GS]`, whose parenthesised value is a CUSIP and correctly fails
#: the ticker pattern -- but the `[GS]` is still worth keeping. Roughly half a
#: typical filing is government securities and funds, and "no ticker" and "not
#: an equity" are different facts. Recording the type lets those be excluded
#: deliberately rather than by absence.
_ASSET_TYPE_RE = re.compile(r"\[(\w{2})\]")

#: A continuation line: the tail of a wrapped amount, or a bare asset-type code.
_CONTINUATION_RE = re.compile(r"^(?:\$[\d,]+|\[\w{2}\])\s*$")


def _reflow(lines: list[str]) -> list[str]:
    """Join continuation fragments back onto the line they belong to.

    The PDF text extractor breaks records at the column boundaries of the
    original table, so a single disclosed transaction arrives as two to four
    physical lines. Reflowing first is what lets the rest of the parser stay
    simple, and it is the difference between reading a $50,000 sale correctly
    and recording it as $15,001.
    """
    out: list[str] = []
    for line in lines:
        if out and _CONTINUATION_RE.match(line):
            out[-1] = f"{out[-1]} {line}".strip()
        else:
            out.append(line)
    return out


@dataclass(frozen=True)
class CongressTrade:
    """One disclosed transaction from a House periodic transaction report."""

    doc_id: str
    member: str
    state_district: str
    symbol: str | None
    asset: str
    code: str             # P purchase, S sale, E exchange
    owner: str            # self | spouse | dependent_child | joint
    transaction_date: date
    #: When the filing became public. **The only defensible observed_at.** The
    #: transaction date is up to 45 days earlier and the notification date is
    #: when the member learned of it -- neither was knowable to us, and using
    #: either would be lookahead of exactly the kind the STOCK Act's reporting
    #: lag creates.
    filed: date
    #: Disclosed as a bracket, never a number. Storing a midpoint as though it
    #: were the size would invent precision the filing does not contain.
    amount_low: float
    amount_high: float
    asset_type: str = ""

    @property
    def is_purchase(self) -> bool:
        return self.code.upper() == "P"

    @property
    def disclosure_lag_days(self) -> int:
        return (self.filed - self.transaction_date).days

    def to_event(self) -> Event:
        return Event(
            source=SOURCE_HOUSE,
            kind=KIND_CONGRESS,
            external_id=f"{self.doc_id}:{self.symbol or self.asset[:40]}:"
                        f"{self.transaction_date.isoformat()}:{self.code}",
            observed_at=datetime.combine(self.filed, datetime.min.time(),
                                         tzinfo=timezone.utc),
            occurred_at=datetime.combine(self.transaction_date,
                                         datetime.min.time(), tzinfo=timezone.utc),
            payload={
                "member": self.member,
                "state_district": self.state_district,
                "symbol": self.symbol,
                "asset": self.asset,
                "code": self.code,
                "is_purchase": self.is_purchase,
                "owner": self.owner,
                "amount_low": self.amount_low,
                "amount_high": self.amount_high,
                "disclosure_lag_days": self.disclosure_lag_days,
                "asset_type": self.asset_type,
            },
        )


@dataclass(frozen=True)
class HouseIndexRow:
    last: str
    first: str
    filing_type: str
    state_district: str
    year: str
    filed: date
    doc_id: str

    @property
    def is_ptr(self) -> bool:
        return self.filing_type.upper() == HOUSE_PTR_TYPE

    @property
    def member(self) -> str:
        return f"{self.first} {self.last}".strip()

    @property
    def pdf_url(self) -> str:
        return HOUSE_PTR_PDF.format(year=self.year, doc_id=self.doc_id)


def parse_house_index(zip_bytes: bytes) -> list[HouseIndexRow]:
    """Read the year's filing index out of the bulk zip.

    utf-8-sig because the file carries a BOM. A plain utf-8 read leaves it glued
    to the first column name, which breaks the header mapping silently rather
    than raising -- every row would then come back with an empty Prefix key and
    no error to show for it.
    """
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    txt = next((n for n in z.namelist() if n.lower().endswith(".txt")), None)
    if txt is None:
        raise HoldingsError("no index .txt in the House bulk zip")
    lines = z.read(txt).decode("utf-8-sig", errors="replace").splitlines()
    if not lines:
        return []
    header = [h.strip() for h in lines[0].split("\t")]
    out: list[HouseIndexRow] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        row = dict(zip(header, line.split("\t")))
        try:
            filed = datetime.strptime((row.get("FilingDate") or "").strip(),
                                      "%m/%d/%Y").date()
        except ValueError:
            continue
        out.append(HouseIndexRow(
            last=(row.get("Last") or "").strip(),
            first=(row.get("First") or "").strip(),
            filing_type=(row.get("FilingType") or "").strip(),
            state_district=(row.get("StateDst") or "").strip(),
            year=(row.get("Year") or "").strip(),
            filed=filed,
            doc_id=(row.get("DocID") or "").strip(),
        ))
    return out


def parse_amount(text: str) -> tuple[float, float]:
    """Turn a disclosed bracket into its bounds.

    A single figure means an exact amount, so both bounds are that figure.
    Returning a range rather than a midpoint is deliberate: the filing discloses
    a bracket, and collapsing it to one number manufactures precision that was
    never disclosed.
    """
    nums = [float(n.replace(",", "")) for n in re.findall(r"[\d,]+", text)]
    if not nums:
        return 0.0, 0.0
    return (nums[0], nums[-1]) if len(nums) > 1 else (nums[0], nums[0])


def parse_house_ptr(text: str, row: HouseIndexRow) -> list[CongressTrade]:
    """Extract transactions from an extracted PTR document.

    The generated PDFs use a consistent two-part shape: an asset line carrying
    `(TICKER) [TYPE]`, then a line with the transaction code, both dates and the
    amount bracket. The asset name wraps often enough that the ticker can sit
    one to three lines above, so the search walks back rather than assuming the
    immediately preceding line.
    """
    lines = _reflow([ln.strip() for ln in text.splitlines() if ln.strip()])
    out: list[CongressTrade] = []
    for i, line in enumerate(lines):
        m = _TXN_RE.match(line)
        if not m:
            continue
        symbol, asset, asset_type, owner = None, None, "", "self"
        for back in range(1, 4):
            if i - back < 0:
                break
            candidate = lines[i - back]
            if asset is None:
                asset = candidate
            am = _ASSET_TYPE_RE.search(candidate)
            if not am:
                continue
            # The asset-type marker anchors the description line; the ticker is
            # optional on it, because a Treasury carries a CUSIP instead.
            asset, asset_type = candidate, am.group(1)
            tm = _TICKER_RE.search(candidate)
            if tm:
                symbol = tm.group(1)
            om = _OWNER_RE.match(candidate)
            if om:
                owner = OWNER_CODES.get(om.group(1).upper(), "self")
            break
        try:
            txn_date = datetime.strptime(m.group("txn"), "%m/%d/%Y").date()
        except ValueError:
            continue
        # A transaction cannot postdate its own disclosure. Real filings contain
        # dates that do -- a live 2025 filing disclosed a trade dated
        # 2026-12-26 -- because members type these by hand. The event store
        # rejects such a row outright, so catching it here keeps one typo from
        # ending an ingest, and dropping it is right regardless: a date we know
        # to be wrong cannot be used to place an entry.
        if txn_date > row.filed:
            continue
        low, high = parse_amount(m.group("amount"))
        out.append(CongressTrade(
            doc_id=row.doc_id, member=row.member,
            state_district=row.state_district,
            symbol=symbol, asset=(asset or "").strip(),
            code=m.group("code").upper(), owner=owner,
            transaction_date=txn_date, filed=row.filed,
            amount_low=low, amount_high=high, asset_type=asset_type,
        ))
    return out


def extract_pdf_text(data: bytes) -> str:
    """Text from a generated PTR PDF.

    pypdf is optional: only this path needs it, and the trades it produces are
    stored once as events, so nothing downstream depends on the reader.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment
        raise HoldingsError(
            "House PTRs are PDFs. Install the reader:\n    pip install pypdf"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)
