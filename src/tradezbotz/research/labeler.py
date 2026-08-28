"""Forward-return labelling.

Turns an event log into a dataset you can measure. Three decisions here carry
the entire validity of everything downstream, so they are explicit rather than
buried.

**1. Entry price is the next tradeable OPEN, never the signal-day close.**
A Form 4 disseminated at 18:40 ET cannot be traded at that day's close -- the
close already happened. Using it is the single most common way event studies
manufacture returns. We convert `observed_at` to Eastern and take:

    observed before 09:30 ET  ->  that session's open is tradeable
    observed at/after 09:30   ->  the next session's open

Sessions come from the bars themselves, so market holidays need no calendar.

**2. Returns are measured from the entry open, not the event-day close.**
Measuring from the close bakes in the overnight move you could never have
captured.

**3. A ticker that stops trading mid-window is recorded, not dropped.**
Dropping it silently reintroduces exactly the survivorship bias the point-in-time
store exists to prevent -- and it drops preferentially the names that failed,
which for an insider-buy study is the population that matters most. We mark the
label `DELISTED_DURING_WINDOW` and record the last observed bar. How to treat
those in aggregate is an analysis decision that must be made in the open:
dropping them is optimistic, assuming -100% is too harsh, and using the last
print is mildly optimistic. `coverage_report()` reports the rate so the choice
is never made by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from enum import Enum
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from .prices import PriceSource, Series

ET = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)

#: Horizons in trading sessions after the entry session.
#: 0 is the entry session's own close.
DEFAULT_HORIZONS: tuple[int, ...] = (0, 1, 5, 20)

#: Extra calendar days of bars to request beyond the longest horizon, so weekends
#: and holidays cannot truncate the window.
HORIZON_PADDING_DAYS = 20


class Coverage(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    DELISTED_DURING_WINDOW = "delisted_during_window"
    NO_ENTRY_BAR = "no_entry_bar"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class Label:
    symbol: str
    observed_at: datetime
    entry_day: date | None
    entry_price: float | None
    returns: dict[int, float]
    coverage: Coverage
    last_available_day: date | None = None
    is_active: bool | None = None

    @property
    def usable(self) -> bool:
        """Whether any horizon resolved. Not a claim that the label is unbiased."""
        return bool(self.returns)


def _entry_index(series: Series, observed_at: datetime) -> int | None:
    """Index of the first bar whose OPEN we could actually have traded.

    Conservative at the boundary: an event landing exactly at 09:30 ET counts as
    too late for that open.
    """
    et = observed_at.astimezone(ET)
    if et.time() < MARKET_OPEN:
        # Same session's open is still ahead of us.
        return series.index_on_or_after(et.date())
    # Need a session strictly after the observation date.
    idx = series.index_on_or_after(et.date())
    if idx is None:
        return None
    while idx < len(series.bars) and series.bars[idx].day <= et.date():
        idx += 1
    return idx if idx < len(series.bars) else None


def label_event(
    series: Series,
    observed_at: datetime,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    as_of: date | None = None,
) -> Label:
    """Compute forward returns for one event against one symbol's bars."""
    if not series.bars:
        return Label(
            symbol=series.symbol, observed_at=observed_at, entry_day=None,
            entry_price=None, returns={}, coverage=Coverage.NO_DATA,
            is_active=series.is_active,
        )

    idx = _entry_index(series, observed_at)
    if idx is None:
        return Label(
            symbol=series.symbol, observed_at=observed_at, entry_day=None,
            entry_price=None, returns={}, coverage=Coverage.NO_ENTRY_BAR,
            last_available_day=series.last_day, is_active=series.is_active,
        )

    entry = series.bars[idx]
    if entry.open <= 0:
        return Label(
            symbol=series.symbol, observed_at=observed_at, entry_day=entry.day,
            entry_price=None, returns={}, coverage=Coverage.NO_ENTRY_BAR,
            last_available_day=series.last_day, is_active=series.is_active,
        )

    returns: dict[int, float] = {}
    for h in horizons:
        j = idx + h
        if j < len(series.bars):
            returns[h] = series.bars[j].close / entry.open - 1.0

    longest = max(horizons)
    if len(returns) == len(horizons):
        coverage = Coverage.COMPLETE
    elif series.is_active is False:
        # Bars ran out AND the vendor says the ticker is no longer trading:
        # this is a delisting, not a gap in our request window.
        coverage = Coverage.DELISTED_DURING_WINDOW
    else:
        coverage = Coverage.PARTIAL

    return Label(
        symbol=series.symbol,
        observed_at=observed_at,
        entry_day=entry.day,
        entry_price=entry.open,
        returns=returns,
        coverage=coverage,
        last_available_day=series.last_day,
        is_active=series.is_active,
    )


class Labeller:
    """Labels many events, fetching each symbol's bars at most once."""

    def __init__(
        self,
        source: PriceSource,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
    ) -> None:
        self.source = source
        self.horizons = tuple(horizons)

    def label(self, events: Iterable[dict]) -> list[Label]:
        """Label events, where each event has `symbol` and `observed_at`.

        Events are grouped by symbol so one fetch serves every event on that
        ticker -- the difference between a ten-hour backfill and a ten-day one
        at 5 requests per minute.
        """
        by_symbol: dict[str, list[datetime]] = {}
        for e in events:
            symbol = (e.get("symbol") or "").upper()
            if not symbol:
                continue
            observed = e["observed_at"]
            if isinstance(observed, str):
                observed = datetime.fromisoformat(observed)
            by_symbol.setdefault(symbol, []).append(observed)

        out: list[Label] = []
        for symbol, times in by_symbol.items():
            start = min(times).date()
            end = _pad_end(max(times).date(), max(self.horizons))
            series = self.source.daily_bars(symbol, start, end)
            for t in times:
                out.append(label_event(series, t, horizons=self.horizons))
        return out


def _pad_end(last_event_day: date, longest_horizon: int) -> date:
    from datetime import timedelta

    # Trading sessions are ~7/10 of calendar days; pad generously so a holiday
    # run cannot truncate the longest horizon.
    calendar_days = int(longest_horizon * 10 / 7) + HORIZON_PADDING_DAYS
    return last_event_day + timedelta(days=calendar_days)


def coverage_report(labels: Sequence[Label]) -> dict[str, float | int]:
    """Summarise how much of the dataset is actually measurable.

    Report this next to every backtest result. A strategy evaluated on 80%
    coverage is a strategy with 20% of its evidence missing, and the missing
    fifth is not missing at random.
    """
    total = len(labels)
    if not total:
        return {"total": 0}

    counts = {c: 0 for c in Coverage}
    for lab in labels:
        counts[lab.coverage] += 1

    return {
        "total": total,
        "complete": counts[Coverage.COMPLETE],
        "partial": counts[Coverage.PARTIAL],
        "delisted_during_window": counts[Coverage.DELISTED_DURING_WINDOW],
        "no_entry_bar": counts[Coverage.NO_ENTRY_BAR],
        "no_data": counts[Coverage.NO_DATA],
        "complete_rate": counts[Coverage.COMPLETE] / total,
        "missing_rate": (counts[Coverage.NO_DATA] + counts[Coverage.NO_ENTRY_BAR]) / total,
        "delisting_rate": counts[Coverage.DELISTED_DURING_WINDOW] / total,
    }
