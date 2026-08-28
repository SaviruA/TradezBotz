"""Bulk ingestion from the SEC's quarterly Form 345 data sets.

https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

The SEC publishes each quarter's Form 3/4/5 data already flattened into TSVs.
One 12.8 MB download carries ~63,000 submissions and ~103,000 non-derivative
transactions -- a whole quarter that would otherwise cost ~27,000 individual
filing fetches. Five years of baselines becomes ~20 downloads instead of roughly
half a million requests.

**What this costs, and why it is safe.** `SUBMISSION.FILING_DATE` is date-only:
there is no acceptance time, so we cannot tell whether a filing landed before
that day's open. We therefore stamp `observed_at` at the 22:00 ET Form 4 cutoff,
which makes the labeller take the *next* session's open for every bulk event.

Measured against real per-filing data, only ~6.7% of filings disseminate before
09:30 ET, so this changes the entry day for about one event in fifteen -- and it
changes it in the safe direction. Entering later than reality understates
returns; it can never invent them.

That is why bulk is used for **baselines** (where the classifier needs only
transaction dates, and intraday timing is irrelevant) while the per-filing path
in `edgar.py` still serves the **labelling window**, where entry precision
actually moves the measured return.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Iterable, Iterator

from .edgar import ET_TZ, FORM4_CUTOFF, EdgarClient, _occurred_at
from .eventstore import Event

QUARTER_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{year}q{quarter}_form345.zip"
)

SOURCE = "sec_form4"
KIND = "insider_transaction"

#: Form 4 and its amendments. Form 3 (initial holdings) and Form 5 (deferred
#: reporting) describe different events and are excluded. Amendments are kept:
#: the classifier keys on (month, year) sets, so a restated trade cannot
#: double-count, and dropping them would lose trades whose only correct record
#: is the amendment.
FORM4_TYPES = {"4", "4/A"}

MONTHS = {m: i for i, m in enumerate(
    "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), start=1)}


class BulkError(RuntimeError):
    pass


def parse_sec_date(raw: str) -> date | None:
    """Parse the data set's `31-MAR-2025` format."""
    parts = (raw or "").strip().upper().split("-")
    if len(parts) != 3 or parts[1] not in MONTHS:
        return None
    try:
        return date(int(parts[2]), MONTHS[parts[1]], int(parts[0]))
    except ValueError:
        return None


def quarters_between(start: date, end: date) -> list[tuple[int, int]]:
    """Every (year, quarter) touching the range, oldest first."""
    out: list[tuple[int, int]] = []
    year, quarter = start.year, (start.month - 1) // 3 + 1
    last = (end.year, (end.month - 1) // 3 + 1)
    while (year, quarter) <= last:
        out.append((year, quarter))
        quarter += 1
        if quarter > 4:
            year, quarter = year + 1, 1
    return out


def download_quarter(
    client: EdgarClient, year: int, quarter: int, dest_dir: str | Path
) -> Path:
    """Fetch one quarterly archive, skipping the download if already present."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{year}q{quarter}_form345.zip"
    if path.exists() and path.stat().st_size > 0:
        return path

    url = QUARTER_URL.format(year=year, quarter=quarter)
    resp = client._session.get(url, timeout=180)
    if resp.status_code == 404:
        raise FileNotFoundError(url)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _rows(zf: zipfile.ZipFile, name: str) -> Iterator[dict[str, str]]:
    with zf.open(name) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        yield from csv.DictReader(text, delimiter="\t")


@dataclass(frozen=True)
class _Submission:
    filing_date: date
    symbol: str
    issuer_cik: str


def _relationship_flags(raw: str) -> tuple[bool, bool, bool]:
    """(is_officer, is_director, is_ten_percent) from the relationship text.

    The bulk data collapses the XML's separate boolean flags into one string
    like `Officer` or `Director,TenPercentOwner`.
    """
    text = (raw or "").lower()
    return ("officer" in text, "director" in text, "tenpercent" in text)


def events_from_archive(
    zip_path: str | Path,
    *,
    before: date | None = None,
) -> Iterator[Event]:
    """Yield events for every Form 4 non-derivative transaction in the archive.

    `before` excludes filings on or after that date. Use it to keep bulk output
    strictly older than the per-filing labelling window: the two paths mint
    different `external_id` formats, so an overlap would store the same
    transaction twice rather than deduplicating it.
    """
    with zipfile.ZipFile(zip_path) as zf:
        submissions: dict[str, _Submission] = {}
        for row in _rows(zf, "SUBMISSION.tsv"):
            if row.get("DOCUMENT_TYPE") not in FORM4_TYPES:
                continue
            symbol = (row.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
            filed = parse_sec_date(row.get("FILING_DATE", ""))
            if not symbol or symbol == "NONE" or not filed:
                continue
            if before and filed >= before:
                continue
            submissions[row["ACCESSION_NUMBER"]] = _Submission(
                filing_date=filed,
                symbol=symbol,
                issuer_cik=(row.get("ISSUERCIK") or "").strip(),
            )

        owners: dict[str, dict[str, str]] = {}
        for row in _rows(zf, "REPORTINGOWNER.tsv"):
            acc = row.get("ACCESSION_NUMBER")
            if acc in submissions and acc not in owners:
                owners[acc] = row  # first reporting owner, matching edgar.py

        for row in _rows(zf, "NONDERIV_TRANS.tsv"):
            acc = row.get("ACCESSION_NUMBER")
            sub = submissions.get(acc)
            if sub is None:
                continue
            trans_date = parse_sec_date(row.get("TRANS_DATE", ""))
            code = (row.get("TRANS_CODE") or "").strip()
            shares = _float(row.get("TRANS_SHARES"))
            sk = (row.get("NONDERIV_TRANS_SK") or "").strip()
            if not (trans_date and code and shares and sk):
                continue

            owner = owners.get(acc, {})
            is_officer, is_director, is_ten = _relationship_flags(
                owner.get("RPTOWNER_RELATIONSHIP", "")
            )
            price = _float(row.get("TRANS_PRICEPERSHARE"))

            # Date-only source: stamp the Form 4 cutoff so the labeller always
            # takes the next session's open. Conservative by construction.
            observed = datetime.combine(
                sub.filing_date, FORM4_CUTOFF, tzinfo=ET_TZ
            )

            yield Event(
                source=SOURCE,
                # NONDERIV_TRANS_SK is the SEC's own stable surrogate key, so
                # identical transaction lines stay distinct without the
                # document-order index the per-filing parser needs.
                external_id=f"{acc}:sk{sk}",
                kind=KIND,
                symbol=sub.symbol,
                observed_at=observed,
                occurred_at=_occurred_at(trans_date, observed),
                payload={
                    "accession": acc,
                    "issuer_cik": sub.issuer_cik,
                    "owner_name": (owner.get("RPTOWNERNAME") or "").strip(),
                    "owner_cik": (owner.get("RPTOWNERCIK") or "").strip(),
                    "is_officer": is_officer,
                    "is_director": is_director,
                    "is_ten_percent": is_ten,
                    "officer_title": (owner.get("RPTOWNER_TITLE") or "").strip() or None,
                    "transaction_code": code,
                    "shares": shares,
                    "price_per_share": price,
                    "acquired_disposed": (
                        row.get("TRANS_ACQUIRED_DISP_CD") or ""
                    ).strip(),
                    "transaction_date": trans_date.isoformat(),
                    "notional": shares * price if price else None,
                    "precision": "date_only",  # provenance: no acceptance time
                },
            )


def _float(raw: str | None) -> float | None:
    try:
        return float((raw or "").strip())
    except (TypeError, ValueError):
        return None
