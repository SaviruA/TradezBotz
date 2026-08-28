"""Building insider baselines from the event store, point-in-time correctly.

The routine/opportunistic classifier needs an insider's trading history. That
history is itself a source of lookahead bias, and a subtler one than the price
data, because it hides inside a *label* rather than a return:

    Classifying a trade on 2025-06-01 using filings disseminated in 2025-09
    means the label depends on information that did not exist yet. The backtest
    then "knows" an insider was routine before anyone could have known.

So baselines are built strictly from filings whose `observed_at` precedes the
event being classified -- the same rule the event store enforces for returns.

There is a second, quieter requirement: **ingest depth**. `MIN_YEARS_FOR_ROUTINE`
is 3, so an insider needs three-plus years of prior filings before they can be
called routine. Ingesting only the 2 years covered by free price data leaves
every insider UNKNOWN and renders the filter inert -- the backtest would then be
testing "insiders bought" rather than "insiders bought unusually", which is a
different and far weaker hypothesis. EDGAR history is free and unbounded, so
baselines should be ingested deeper than the labelling window.
`coverage_warning()` exists to make that failure loud rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

from .classify import (
    MIN_YEARS_FOR_ROUTINE,
    InsiderClass,
    PriorTrade,
    RoutineClassifier,
)

SOURCE = "sec_form4"
KIND = "insider_transaction"


@dataclass(frozen=True)
class ClassifiedEvent:
    symbol: str
    owner_cik: str
    owner_name: str
    observed_at: datetime
    transaction_date: date
    insider_class: InsiderClass
    payload: dict


def _prior_trades(events: Iterable[dict], before: datetime, owner_cik: str) -> list[PriorTrade]:
    out = []
    for e in events:
        if e["payload"].get("owner_cik") != owner_cik:
            continue
        observed = e["observed_at"]
        if isinstance(observed, str):
            observed = datetime.fromisoformat(observed)
        if observed >= before:
            continue  # not yet public when the event we are labelling occurred
        tdate = e["payload"].get("transaction_date")
        if tdate:
            out.append(PriorTrade(owner_cik, date.fromisoformat(tdate)))
    return out


def classify_events(events: Sequence[dict]) -> list[ClassifiedEvent]:
    """Classify each event using only filings public before that event.

    `events` should be everything the store holds for the source, not just the
    ones being labelled -- the extra rows are the baseline. Ordering does not
    matter; visibility is decided per event by `observed_at`.
    """
    by_owner: dict[str, list[dict]] = {}
    for e in events:
        cik = e["payload"].get("owner_cik")
        if cik:
            by_owner.setdefault(cik, []).append(e)

    out: list[ClassifiedEvent] = []
    for e in events:
        payload = e["payload"]
        cik = payload.get("owner_cik")
        tdate = payload.get("transaction_date")
        if not cik or not tdate:
            continue
        observed = e["observed_at"]
        if isinstance(observed, str):
            observed = datetime.fromisoformat(observed)

        clf = RoutineClassifier()
        clf.add_history(_prior_trades(by_owner[cik], observed, cik))

        out.append(
            ClassifiedEvent(
                symbol=e.get("symbol") or "",
                owner_cik=cik,
                owner_name=payload.get("owner_name", ""),
                observed_at=observed,
                transaction_date=date.fromisoformat(tdate),
                insider_class=clf.classify(cik, date.fromisoformat(tdate)),
                payload=payload,
            )
        )
    return out


def coverage_warning(
    classified: Sequence[ClassifiedEvent],
    *,
    threshold: float = 0.9,
) -> str | None:
    """Return a warning when the classifier is effectively inert.

    If nearly everything lands in UNKNOWN, the ingest is too shallow to establish
    baselines and the routine/opportunistic distinction is not actually being
    applied -- results would describe a different hypothesis than intended.
    """
    if not classified:
        return None
    unknown = sum(1 for c in classified if c.insider_class is InsiderClass.UNKNOWN)
    share = unknown / len(classified)
    if share < threshold:
        return None
    return (
        f"{share:.0%} of events classify as UNKNOWN. The classifier needs "
        f"{MIN_YEARS_FOR_ROUTINE}+ years of an insider's prior filings, so a "
        "shallow ingest leaves it inert and the routine/opportunistic filter is "
        "not being applied. Ingest more EDGAR history (it is free and unbounded, "
        "unlike the price window) before trusting any result."
    )


def baseline_start(label_start: date, years: int = MIN_YEARS_FOR_ROUTINE) -> date:
    """Earliest filing date needed to classify events from `label_start`.

    One extra year of margin: the streak test looks at the same calendar month
    across consecutive prior years, so an insider needs a full span of years
    strictly before the trade, not merely `years` of calendar coverage.
    """
    return label_start - timedelta(days=365 * (years + 1))
