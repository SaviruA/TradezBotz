"""Material-event and offering filings: 8-K and 424B.

**Why these two, and why for small caps specifically.** Measured against the
Benzinga news feed, journalist coverage collapses with market cap: NVDA, AAPL
and TSLA each drew over 200 articles in June 2024, BDX -- a large-cap S&P 500
name -- drew five, and XELB, ARI and AULT drew zero. Over the same period XELB
filed eleven 8-Ks and ARI filed six.

Disclosure obligations do not scale with market cap. Coverage does. That
inversion is the entire argument for reading filings rather than news on the
population where insider buying concentrates: for a company nobody writes about,
the 8-K *is* the news.

**8-K item codes are the signal, not the filing.** An 8-K on its own means
almost nothing -- Management Science finds firms with *higher* 8-K filing
intensity earn *lower* future returns, a 4.3%/yr long-short spread, so raw
filing frequency is a negative signal rather than a positive one. The items
carry the information: 2.02 (results), 7.01 (Reg FD) and 8.01 (other) draw the
abnormal attention and returns, and 5.02 (officer departure) is the one item
reliably associated with an attention spike on the filing day.

**424B is the other half, and it points down.** Registered offerings cause 20-30%
average drops in small caps, and the prospectus sits on EDGAR for hours before
any of it reaches the media. The subtype matters: 424B4 is a priced deal with
immediate dilution, 424B3 and 424B5 are usually ATM programmes -- a standing
marginal seller rather than a single hit. This is a *negative* signal that pairs
directly with the positive insider one: an insider buying into a company about
to run an ATM is a different trade from one that is not.

Point-in-time handling is identical to Form 4: what matters is when the filing
became *knowable*, not the event date printed inside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterator, Sequence

from .edgar import ACCESSION_RE, DAILY_INDEX, EdgarClient, EdgarError, _disseminated_at
from .eventstore import Event

SOURCE_8K = "sec_8k"
SOURCE_424B = "sec_424b"
KIND_MATERIAL_EVENT = "material_event"
KIND_OFFERING = "offering"

#: The official 8-K item codes. Used to filter regex hits: a filing's exhibits
#: routinely contain strings like "Item 3.05" from unrelated documents, and
#: without a whitelist those become phantom events.
ITEM_NAMES: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate a Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Election of Directors or Principal Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Amendments to the Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of Servicer or Trustee",
    "6.03": "Change in Credit Enhancement or External Support",
    "6.04": "Failure to Make a Required Distribution",
    "6.05": "Securities Act Updating Disclosure",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

#: Items carrying documented abnormal returns or attention. Kept as a named set
#: rather than inlined so a hypothesis can say what it is testing.
ITEMS_RESULTS = frozenset({"2.02"})
ITEMS_ATTENTION = frozenset({"2.02", "7.01", "8.01"})
ITEMS_MANAGEMENT = frozenset({"5.02", "5.01"})
ITEMS_DISTRESS = frozenset({"1.03", "2.06", "3.01", "4.02"})
ITEMS_DILUTIVE = frozenset({"3.02", "2.03"})

#: Purely administrative. 9.01 accompanies almost every 8-K and carries no
#: information on its own; treating it as an event would bury everything else.
ITEMS_ROUTINE = frozenset({"9.01", "5.03"})

#: 424B subtypes, and what each implies about dilution.
OFFERING_TYPES: dict[str, str] = {
    "424B1": "prospectus, previously omitted information",
    "424B2": "prospectus supplement, priced offering",
    "424B3": "prospectus supplement, often ATM programme",
    "424B4": "priced offering completed, immediate dilution",
    "424B5": "shelf supplement, often ATM capacity",
    "424B7": "prospectus supplement, selling shareholders",
    "424B8": "prospectus supplement",
}

#: Subtypes signalling an immediate, priced hit rather than a drip.
#:
#: 424B2 is deliberately NOT here, despite being a priced supplement by
#: definition. Measured on 2026-08-26, the day's 424B filings broke down as:
#:
#:     424B2  447    424B3  47    424B5  4    424B4  2    424B7  1
#:
#: and the 424B2 rows were overwhelmingly Bank of America and BofA Finance
#: structured-note takedowns. Treating those as dilution events would bury the
#: two 424B4 filings that actually are dilution under four hundred that are not,
#: and would attach a "dilution" label to the largest bank in the country.
#:
#: 424B2 is still ingested -- it is a real filing and a small-cap 424B2 is a real
#: offering -- but the immediacy flag is reserved for the subtypes where the
#: small-cap base rate is high. Filtering 424B2 properly needs an issuer-size
#: screen, which is a separate piece of work.
IMMEDIATE_DILUTION = frozenset({"424B1", "424B4"})

#: Subtypes whose daily volume is dominated by large financial issuers rolling
#: shelf programmes. Present for transparency about what the base rate is.
LARGE_ISSUER_HEAVY = frozenset({"424B2"})

_ITEM_RE = re.compile(r"Item\s+(\d\.\d{2})", re.IGNORECASE)
_HEADER_ITEM_RE = re.compile(r"ITEM INFORMATION:\s*(.+)")
_CIK_RE = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)")
_ACCEPTED_RE = re.compile(r"ACCEPTANCE-DATETIME>\s*(\d{14})")
_SYMBOL_RE = re.compile(r"\(Trading Symbol[^)]*\)|\btrading symbol\b", re.IGNORECASE)


@dataclass(frozen=True)
class MaterialEvent:
    """One 8-K, reduced to its item codes."""

    accession: str
    cik: str
    company: str
    items: tuple[str, ...]
    observed_at: datetime
    #: True when the header's item count disagrees with the codes found in the
    #: body. Recorded rather than resolved: a silent guess about which source is
    #: right would be worse than a visible disagreement.
    item_mismatch: bool = False

    @property
    def informative_items(self) -> tuple[str, ...]:
        """Items excluding the purely administrative ones."""
        return tuple(i for i in self.items if i not in ITEMS_ROUTINE)

    def has_any(self, group: frozenset[str]) -> bool:
        return any(i in group for i in self.items)

    def to_event(self) -> Event:
        return Event(
            source=SOURCE_8K,
            kind=KIND_MATERIAL_EVENT,
            external_id=self.accession,
            observed_at=self.observed_at,
            occurred_at=self.observed_at,
            payload={
                "cik": self.cik,
                "company": self.company,
                "items": list(self.items),
                "informative_items": list(self.informative_items),
                "item_mismatch": self.item_mismatch,
                "n_items": len(self.informative_items),
            },
        )


@dataclass(frozen=True)
class Offering:
    """One 424B prospectus."""

    accession: str
    cik: str
    company: str
    form_type: str
    observed_at: datetime

    @property
    def immediate(self) -> bool:
        return self.form_type.upper() in IMMEDIATE_DILUTION

    def to_event(self) -> Event:
        return Event(
            source=SOURCE_424B,
            kind=KIND_OFFERING,
            external_id=self.accession,
            observed_at=self.observed_at,
            occurred_at=self.observed_at,
            payload={
                "cik": self.cik,
                "company": self.company,
                "form_type": self.form_type,
                "immediate_dilution": self.immediate,
                "description": OFFERING_TYPES.get(self.form_type.upper(), "unknown"),
            },
        )


def daily_filings(client: EdgarClient, day: date,
                  forms: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return (form_type, cik, document_path) for `forms` filed on `day`.

    Generalises `EdgarClient.daily_form4_filings` to any form. Deduplicates on
    accession for the same reason: EDGAR indexes a filing once per involved CIK,
    so the raw rows overcount documents.
    """
    url = DAILY_INDEX.format(
        year=day.year, qtr=(day.month - 1) // 3 + 1, ymd=day.strftime("%Y%m%d")
    )
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
        match = ACCESSION_RE.search(path)
        key = match.group(1) if match else path
        if key in seen:
            continue
        seen.add(key)
        out.append((form, cik, path))
    return out


def _accession_from(raw: str, fallback: str = "") -> str:
    match = ACCESSION_RE.search(raw)
    return match.group(1) if match else fallback


def _observed_at(raw: str) -> datetime | None:
    """Dissemination time from the SGML acceptance stamp.

    The acceptance timestamp is when the SEC took the filing, which is when it
    became knowable. The event date printed inside the document is frequently
    days earlier and must never be used as the observation time.
    """
    match = _ACCEPTED_RE.search(raw)
    if not match:
        return None
    stamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return _disseminated_at(stamp)


def _company(raw: str) -> str:
    match = re.search(r"COMPANY CONFORMED NAME:\s*(.+)", raw)
    return match.group(1).strip() if match else ""


def parse_8k(raw: str, path: str = "") -> MaterialEvent | None:
    """Extract item codes from a raw 8-K submission.

    Item codes are taken from the document body and filtered against
    `ITEM_NAMES`, because exhibits attached to the filing contain strings like
    "Item 3.05" belonging to other documents entirely. The SGML header's
    `ITEM INFORMATION` lines give an independent count, and a disagreement
    between the two is recorded rather than silently resolved.
    """
    header_end = raw.find("</SEC-HEADER>")
    header = raw[:header_end] if header_end > 0 else raw[:4000]

    cik_match = _CIK_RE.search(header)
    observed = _observed_at(raw)
    if observed is None:
        return None

    body_items = sorted({
        code for code in _ITEM_RE.findall(raw)
        if code in ITEM_NAMES
    })
    header_count = len(_HEADER_ITEM_RE.findall(header))
    if not body_items:
        return None

    return MaterialEvent(
        accession=_accession_from(path or raw),
        cik=(cik_match.group(1).lstrip("0") if cik_match else ""),
        company=_company(header),
        items=tuple(body_items),
        observed_at=observed,
        item_mismatch=bool(header_count) and header_count != len(body_items),
    )


def parse_424b(raw: str, form_type: str, path: str = "") -> Offering | None:
    header_end = raw.find("</SEC-HEADER>")
    header = raw[:header_end] if header_end > 0 else raw[:4000]
    observed = _observed_at(raw)
    if observed is None:
        return None
    cik_match = _CIK_RE.search(header)
    return Offering(
        accession=_accession_from(path or raw),
        cik=(cik_match.group(1).lstrip("0") if cik_match else ""),
        company=_company(header),
        form_type=form_type.upper(),
        observed_at=observed,
    )


#: Every 424B variant EDGAR publishes.
FORMS_424B = tuple(OFFERING_TYPES)


def ingest_day(client: EdgarClient, day: date,
               forms: Sequence[str] = ("8-K",) + FORMS_424B
               ) -> Iterator[MaterialEvent | Offering]:
    """Yield parsed 8-K and 424B filings for one day.

    One malformed filing must not end a multi-year backfill, so parse failures
    are skipped rather than raised -- the same policy as the Form 4 path.
    """
    for form, _cik, path in daily_filings(client, day, forms):
        try:
            raw = client.fetch_filing(path)
        except Exception:  # noqa: BLE001
            continue
        try:
            if form == "8-K":
                parsed = parse_8k(raw, path)
            else:
                parsed = parse_424b(raw, form, path)
        except Exception:  # noqa: BLE001
            continue
        if parsed is not None:
            yield parsed


# --- selectors ----------------------------------------------------------------
#
# Bound to an event payload so they compose with the price selectors through
# `backtest.all_of`. The pairing that motivates all of this is insider buying
# conditioned on -- or *not* conditioned on -- a pending offering.

def has_item(payload: dict, group: frozenset[str]) -> bool:
    return any(i in group for i in (payload.get("items") or ()))


def is_results_8k(payload: dict) -> bool:
    """Item 2.02. The item with the best-documented abnormal return."""
    return has_item(payload, ITEMS_RESULTS)


def is_management_change(payload: dict) -> bool:
    """Item 5.02, the one item reliably associated with an attention spike on
    the filing day."""
    return has_item(payload, ITEMS_MANAGEMENT)


def is_distress_8k(payload: dict) -> bool:
    """Delisting notice, impairment, bankruptcy, or a restatement."""
    return has_item(payload, ITEMS_DISTRESS)


def is_informative_8k(payload: dict) -> bool:
    """Any item beyond the purely administrative ones.

    The useful baseline: 9.01 accompanies nearly every 8-K and 5.03 is routine
    housekeeping, so "an 8-K was filed" is close to contentless without this.
    """
    return bool(payload.get("informative_items"))


def is_immediate_dilution(payload: dict) -> bool:
    return bool(payload.get("immediate_dilution"))
