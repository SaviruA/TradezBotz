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
