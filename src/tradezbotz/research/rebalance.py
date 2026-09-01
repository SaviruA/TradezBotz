"""A rebalance-date population, so cross-sectional strategies fit the engine.

Everything else in this system is an event study: something happened on a date,
measure what followed. A valuation multiple is not that. "Hold the cheapest
quintile and rebalance monthly" has no event -- the strategy acts on a calendar,
and the decision depends on every *other* company as much as on this one.

That mismatch, not the ratios and not the data, is what blocked the large-cap
value candidates. The fix is to notice that the engine's requirements are
weaker than they look: `run` wants (payload, label) pairs where the label
carries forward returns. A rebalance date is a perfectly good `observed_at`, and
"this symbol was in the cheapest quintile that month" is a perfectly good
payload field. So the population is synthesised rather than the engine rewritten.

**The ranking is what makes this cross-sectional and what makes it dangerous.**
A quintile is computed across the universe *on that date*, so the obvious
implementation ranks every symbol using whatever data is in hand -- which for a
symbol that reported late means data published after the rebalance. Each cohort
here is therefore built from snapshots taken strictly before its own date, and a
symbol with nothing filed by then is dropped from that month rather than carried
forward from an earlier one.

**Survivorship is the second trap and it is not fully solved.** The universe is
assembled from symbols we hold prices for, and a company that delisted in 2019
is in the price cache only if it was fetched before it went. `universe_warning`
reports how much of each cohort is still trading today, because a cohort that is
100% alive is a cohort that has already selected for survival.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Callable, Iterable, Sequence

#: Rebalance cadence. Monthly is the convention in the value literature and is
#: the shortest cadence that does not turn a value strategy into a trading-cost
#: study: Gray & Vogel and Loughran & Wellman both rebalance annually, and
#: monthly is already more turnover than the evidence they built on.
REBALANCE_DAY = 1

#: Quantiles the universe is cut into. Five is the convention; it is also about
#: the finest cut our universe supports, since a decile of a few hundred names
#: is a few dozen and the trade-count floor is 30.
QUANTILES = 5

#: Minimum symbols in a cohort before ranking means anything. Below this the
#: "cheapest quintile" is a handful of names and its return is noise about one
#: or two of them.
MIN_COHORT = 50


@dataclass(frozen=True)
class RankedSymbol:
    """One symbol's standing in one rebalance cohort."""

    symbol: str
    as_of: date
    metric: str
    value: float
    #: 0 is the cheapest quintile. Named rather than numbered in the payload so
    #: a selector reads `cheapest_quintile` rather than `quintile == 0`.
    quantile: int
    cohort_size: int


def month_starts(start: date, end: date) -> list[date]:
    """Rebalance dates in [start, end]."""
    out: list[date] = []
    year, month = start.year, start.month
    while True:
        day = date(year, month, REBALANCE_DAY)
        if day > end:
            break
        if day >= start:
            out.append(day)
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return out


def rank_cohort(values: dict[str, float], metric: str, as_of: date,
                *, quantiles: int = QUANTILES,
                lower_is_cheaper: bool = True) -> list[RankedSymbol]:
    """Cut one date's universe into quantiles on one metric.

    Ties are broken by the sort's stability rather than by splitting them
    across a boundary, which keeps a cohort of identical values from being
    arbitrarily labelled cheap and expensive.
    """
    usable = {s: v for s, v in values.items() if v is not None and v > 0}
    n = len(usable)
    if n < MIN_COHORT:
        return []
    order = sorted(usable.items(), key=lambda kv: kv[1],
                   reverse=not lower_is_cheaper)
    out: list[RankedSymbol] = []
    for i, (symbol, value) in enumerate(order):
        # Floor rather than round: with n not divisible by `quantiles` the last
        # bucket absorbs the remainder, and clamping stops index n landing in a
        # bucket that does not exist.
        q = min(i * quantiles // n, quantiles - 1)
        out.append(RankedSymbol(symbol, as_of, metric, value, q, n))
    return out


@dataclass(frozen=True)
class Cohort:
    """One rebalance date's ranked universe, across every metric."""

    as_of: date
    rankings: dict[str, list[RankedSymbol]]

    @property
    def size(self) -> int:
        return max((len(r) for r in self.rankings.values()), default=0)


#: Metrics ranked at each rebalance. Every one is "lower is cheaper".
RANKED_METRICS = ("ev_to_ebitda", "price_to_free_cash_flow",
                  "price_to_earnings", "price_to_sales")


def build_cohorts(
    dates: Sequence[date],
    symbols: Sequence[str],
    snapshot_for: Callable[[str, date], object | None],
    price_for: Callable[[str, date], float | None],
    *,
    metrics: Sequence[str] = RANKED_METRICS,
) -> list[Cohort]:
    """Rank the universe at each date, using only what was filed before it.

    `snapshot_for` must itself be point-in-time -- it is handed the rebalance
    date and must return a snapshot built from facts filed strictly before it.
    This function cannot verify that and does not try; the guarantee lives in
    `fundamentals.visible`, which every caller routes through.
    """
    cohorts: list[Cohort] = []
    for as_of in dates:
        values: dict[str, dict[str, float]] = {m: {} for m in metrics}
        for symbol in symbols:
            snap = snapshot_for(symbol, as_of)
            if snap is None:
                continue
            price = price_for(symbol, as_of)
            if price is None or price <= 0:
                continue
            for metric in metrics:
                getter = getattr(snap, metric, None)
                if getter is None:
                    continue
                value = getter(price) if callable(getter) else getter
                if value is not None and value > 0:
                    values[metric][symbol] = value
        rankings = {
            m: rank_cohort(values[m], m, as_of) for m in metrics
        }
        if any(rankings.values()):
            cohorts.append(Cohort(as_of, rankings))
    return cohorts


def to_events(cohorts: Sequence[Cohort]) -> tuple[list[dict], list[dict]]:
    """Flatten cohorts into (events, payloads) the labeller and engine accept.

    One row per (symbol, rebalance date). `observed_at` is the rebalance date
    itself: the ranking is computed from data filed before it, so the decision
    is knowable that morning and the labeller's usual entry rule -- buy the next
    session's open -- applies unchanged.
    """
    by_key: dict[tuple[str, date], dict] = {}
    for cohort in cohorts:
        for metric, ranked in cohort.rankings.items():
            for row in ranked:
                key = (row.symbol, row.as_of)
                payload = by_key.setdefault(key, {
                    "symbol": row.symbol,
                    "rebalance_date": row.as_of.isoformat(),
                    "cohort_size": row.cohort_size,
                })
                payload[metric] = row.value
                payload[f"{metric}_quantile"] = row.quantile
                payload[f"cheapest_{metric}"] = row.quantile == 0

    events, payloads = [], []
    for (symbol, as_of), payload in sorted(by_key.items()):
        events.append({
            "symbol": symbol,
            "observed_at": datetime.combine(
                as_of, dtime(0, 0), tzinfo=timezone.utc).isoformat(),
        })
        payloads.append(payload)
    return events, payloads


def universe_warning(cohorts: Sequence[Cohort],
                     still_trading: Callable[[str], bool]) -> str:
    """Report how much of the oldest cohort is still listed.

    A universe assembled from a price cache is a universe of companies somebody
    fetched, and companies that delisted years ago are the ones most likely to
    be missing. If the oldest cohort is ~100% alive, it has already selected for
    survival and every backtest on it is measuring survivors -- which reads as a
    strategy result and is not one.
    """
    if not cohorts:
        return "no cohorts built"
    oldest = min(cohorts, key=lambda c: c.as_of)
    names = {r.symbol for ranked in oldest.rankings.values() for r in ranked}
    if not names:
        return "oldest cohort is empty"
    alive = sum(1 for s in names if still_trading(s))
    share = alive / len(names)
    line = (f"survivorship: {alive:,} of {len(names):,} names in the "
            f"{oldest.as_of} cohort are still trading ({share:.0%})")
    if share > 0.95:
        line += ("\n  WARNING: a cohort this alive has selected for survival. "
                 "Delisted names are missing from the price cache rather than "
                 "from the market, and returns measured on what is left are "
                 "biased upward by an amount nothing here can estimate.")
    return line
