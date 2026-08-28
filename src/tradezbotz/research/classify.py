"""Routine vs opportunistic insider classification.

Cohen, Malloy & Pomorski (Decoding Inside Information, NBER w16454) found that
insider trades split cleanly into two populations:

  routine       -- an insider who trades in the same calendar month for several
                   consecutive years. Diversification, tax planning, a standing
                   plan. Carries essentially no information.
  opportunistic -- everything else. A trade that breaks the insider's own
                   pattern. This subset carried the abnormal returns in their
                   sample (82bp/month value-weighted, 180bp/month equal-weighted).

The classification is therefore about *deviation from an individual's baseline*,
not about the size or direction of the trade.

Two warnings that belong in the code rather than a README:

1. The published magnitudes are from a 1986-2007 sample. Assume material decay
   since publication. Treat them as a reason to investigate, never as an
   expected return.
2. Classification requires history. An insider with fewer than
   `MIN_YEARS_FOR_ROUTINE` years of prior trades cannot be shown to be routine,
   so they fall to UNKNOWN rather than being assumed opportunistic. Collapsing
   UNKNOWN into OPPORTUNISTIC would inflate the signal population with
   first-time filers and is the obvious way to fool yourself here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Iterable, Sequence


class InsiderClass(str, Enum):
    ROUTINE = "routine"
    OPPORTUNISTIC = "opportunistic"
    UNKNOWN = "unknown"


#: Consecutive years of same-month trading required to call an insider routine.
#: The paper uses three; lowering this makes the routine bucket greedier and
#: shrinks the opportunistic population.
MIN_YEARS_FOR_ROUTINE = 3


@dataclass(frozen=True)
class PriorTrade:
    """A trade by one insider, used only to establish their baseline."""

    owner_cik: str
    transaction_date: date


def _same_month_streak(years: Sequence[int]) -> int:
    """Longest run of consecutive years present in `years`."""
    if not years:
        return 0
    ordered = sorted(set(years))
    best = run = 1
    for prev, cur in zip(ordered, ordered[1:]):
        run = run + 1 if cur == prev + 1 else 1
        best = max(best, run)
    return best


class RoutineClassifier:
    """Classifies insiders from their own trading history.

    History must be restricted by the caller to trades that were *public* before
    the trade being classified. Feeding in the full history would leak the
    future into the label, which is the same lookahead bug the event store
    exists to prevent -- just moved one layer up.
    """

    def __init__(self, min_years: int = MIN_YEARS_FOR_ROUTINE) -> None:
        self.min_years = min_years
        # owner_cik -> calendar month -> set of years traded in that month
        self._history: dict[str, dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

    def add_history(self, trades: Iterable[PriorTrade]) -> None:
        for t in trades:
            self._history[t.owner_cik][t.transaction_date.month].add(
                t.transaction_date.year
            )

    def years_of_history(self, owner_cik: str) -> int:
        months = self._history.get(owner_cik)
        if not months:
            return 0
        years = {y for ys in months.values() for y in ys}
        return len(years)

    def classify(self, owner_cik: str, when: date) -> InsiderClass:
        if self.years_of_history(owner_cik) < self.min_years:
            return InsiderClass.UNKNOWN

        prior_years = {
            y for y in self._history[owner_cik][when.month] if y < when.year
        }
        if _same_month_streak(sorted(prior_years)) >= self.min_years:
            return InsiderClass.ROUTINE
        return InsiderClass.OPPORTUNISTIC


@dataclass(frozen=True)
class ScoredSignal:
    symbol: str
    owner_cik: str
    owner_name: str
    insider_class: InsiderClass
    conviction: float
    reasons: tuple[str, ...]


def score(
    *,
    symbol: str,
    owner_cik: str,
    owner_name: str,
    insider_class: InsiderClass,
    is_open_market_buy: bool,
    is_officer: bool,
    is_director: bool,
    officer_title: str | None,
    notional: float | None,
) -> ScoredSignal:
    """Produce a bounded conviction score in [0, 1].

    These weights are a starting hypothesis, NOT a fitted model. They exist so
    the backtest has something concrete to measure; the whole point of Phase 1
    is to find out which of these components actually carry information and to
    discard the ones that do not. Do not tune them against the same data you
    then report results on.
    """
    reasons: list[str] = []
    conviction = 0.0

    if not is_open_market_buy:
        return ScoredSignal(
            symbol, owner_cik, owner_name, insider_class, 0.0,
            ("not an open-market purchase",),
        )

    conviction += 0.4
    reasons.append("open-market purchase")

    if insider_class is InsiderClass.OPPORTUNISTIC:
        conviction += 0.3
        reasons.append("breaks insider's established pattern")
    elif insider_class is InsiderClass.ROUTINE:
        conviction -= 0.3
        reasons.append("matches insider's routine calendar")
    else:
        reasons.append("insufficient history to classify")

    title = (officer_title or "").upper()
    if is_officer and any(k in title for k in ("CHIEF EXECUTIVE", "CEO", "PRESIDENT")):
        conviction += 0.2
        reasons.append("CEO-level officer")
    elif is_officer and any(k in title for k in ("CHIEF FINANCIAL", "CFO")):
        conviction += 0.15
        reasons.append("CFO-level officer")
    elif is_director:
        conviction += 0.05
        reasons.append("director")

    if notional is not None:
        if notional >= 1_000_000:
            conviction += 0.15
            reasons.append("notional >= $1M")
        elif notional >= 100_000:
            conviction += 0.05
            reasons.append("notional >= $100k")
        elif notional < 10_000:
            conviction -= 0.1
            reasons.append("token size (< $10k)")

    return ScoredSignal(
        symbol=symbol,
        owner_cik=owner_cik,
        owner_name=owner_name,
        insider_class=insider_class,
        conviction=max(0.0, min(1.0, conviction)),
        reasons=tuple(reasons),
    )
