"""Bar-level data validation.

Concretum Group ran our exact experiment -- identical Opening Range Breakout
logic across Polygon/Massive, Alpaca, IQFeed, IBKR and Databento over ten years
-- and final portfolio values ranged from **$226k to $726k on identical $25k
starting capital**. Over 3x dispersion from the same code on different data.
They also found the *same* provider returned different results when re-downloaded
years apart: "data is not a neutral input".

They name five failure modes. Our cross-check only compared closes, so it caught
roughly one of them. This module covers the rest at the single-series level:

  phantom high/low   an isolated spike no other venue confirms. Triggers false
                     Donchian breakouts and false stop-losses -- and ATR,
                     Donchian and any stop logic read highs and lows, which
                     nothing was validating.
  stale bar          open==high==low==close, often with negligible volume.
                     Signals a carried-forward price rather than a traded one.
                     Depresses measured volatility, which widens Bollinger
                     signals and shrinks ATR stops.
  missing bar        Alpaca documents that it emits no bar when an interval has
                     no trades *or only a single trade*. On thin names IEX
                     therefore drops days entirely, so a 20-bar window spans a
                     different calendar period in each source.
  invalid OHLC       high < low, close outside [low, high], non-positive prices.
                     Rare, but silently poisons every downstream statistic.

**Why modified Z-score rather than standard deviations.** An outlier inflates
the very standard deviation used to detect it, so a 3-sigma rule misses the
spikes that matter most. The median-absolute-deviation form is robust to
contamination, which is why the ultra-high-frequency cleaning literature
(Brownlees & Gallo) builds on it.

**Nothing here repairs data.** Bars are flagged, never interpolated or
forward-filled. Manufacturing a plausible price is how a cleaning step becomes
a source of fictional returns.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Sequence

from .prices import Bar, Series

#: Consistency constant making MAD comparable to a standard deviation for
#: normally distributed data.
MAD_SCALE = 0.6745

#: Iglewicz & Hoaglin's constant for the mean-absolute-deviation fallback used
#: when MAD is zero, chosen so both measures agree for normal data.
MEAN_AD_SCALE = 1.253314

#: Modified Z-score above which a bar is called a spike. 3.5 is the conventional
#: threshold in the outlier-detection literature.
SPIKE_THRESHOLD = 3.5

#: Window either side of a bar used to judge it. Wide enough to be stable,
#: narrow enough to track changing volatility.
SPIKE_WINDOW = 20

#: A bar whose entire range is this fraction of price or less is treated as
#: having no real range at all.
FLAT_RANGE_EPS = 1e-9


class Issue(str, Enum):
    INVALID_OHLC = "invalid_ohlc"
    STALE = "stale"
    ZERO_VOLUME = "zero_volume"
    SPIKE_HIGH = "spike_high"
    SPIKE_LOW = "spike_low"
    SPIKE_RETURN = "spike_return"


@dataclass(frozen=True)
class BarIssue:
    day: date
    issue: Issue
    detail: str


@dataclass(frozen=True)
class QualityReport:
    symbol: str
    n_bars: int
    issues: tuple[BarIssue, ...] = ()
    missing_sessions: tuple[date, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        out = {i.value: 0 for i in Issue}
        for b in self.issues:
            out[b.issue.value] += 1
        return out

    @property
    def flagged_days(self) -> set[date]:
        return {b.day for b in self.issues}

    @property
    def clean_rate(self) -> float:
        if not self.n_bars:
            return 0.0
        return 1.0 - len(self.flagged_days) / self.n_bars

    def summary(self) -> str:
        c = self.counts
        parts = [f"{k}={v}" for k, v in c.items() if v]
        return (
            f"{self.symbol:<8} bars {self.n_bars:>5}  clean {self.clean_rate:>6.1%}"
            + (f"  [{', '.join(parts)}]" if parts else "")
            + (f"  missing {len(self.missing_sessions)}" if self.missing_sessions else "")
        )


def modified_zscores(values: Sequence[float]) -> list[float]:
    """MAD-based Z-scores. Robust: an outlier cannot inflate its own denominator.

    **The MAD-zero trap.** When more than half the values are identical the
    median absolute deviation is exactly zero, and a naive implementation then
    either divides by zero or reports no outliers at all -- going blind on the
    very cases that matter, such as a flat, thinly traded series with one
    genuine spike.

    Iglewicz & Hoaglin's fallback handles it: substitute the *mean* absolute
    deviation, scaled by 1.253314 so the two measures agree for normal data.
    Only when that is also zero is the window truly constant and no deviation
    meaningful.
    """
    if len(values) < 3:
        return [0.0] * len(values)
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]

    mad = statistics.median(deviations)
    if mad > 0:
        return [MAD_SCALE * (v - med) / mad for v in values]

    mean_ad = statistics.fmean(deviations)
    if mean_ad > 0:
        return [(v - med) / (MEAN_AD_SCALE * mean_ad) for v in values]

    return [0.0] * len(values)


def _check_ohlc(bar: Bar) -> BarIssue | None:
    if min(bar.open, bar.high, bar.low, bar.close) <= 0:
        return BarIssue(bar.day, Issue.INVALID_OHLC, "non-positive price")
    if bar.high < bar.low:
        return BarIssue(bar.day, Issue.INVALID_OHLC, f"high {bar.high} < low {bar.low}")
    if not (bar.low <= bar.open <= bar.high):
        return BarIssue(bar.day, Issue.INVALID_OHLC, f"open {bar.open} outside range")
    if not (bar.low <= bar.close <= bar.high):
        return BarIssue(bar.day, Issue.INVALID_OHLC, f"close {bar.close} outside range")
    return None


def expected_sessions(bars: Sequence[Bar]) -> list[date]:
    """Weekdays between the first and last bar that carry no bar.

    Market holidays appear here as false positives -- shipping a holiday
    calendar would trade one dependency for a small amount of noise, and the
    count is only ever used comparatively between two sources over the same
    span, where holidays cancel.
    """
    if len(bars) < 2:
        return []
    have = {b.day for b in bars}
    out, day = [], bars[0].day
    while day <= bars[-1].day:
        if day.weekday() < 5 and day not in have:
            out.append(day)
        day += timedelta(days=1)
    return out


def validate(series: Series) -> QualityReport:
    """Flag suspect bars in one series. Never modifies or removes them."""
    bars = list(series.bars)
    if not bars:
        return QualityReport(series.symbol, 0)

    issues: list[BarIssue] = []
    for bar in bars:
        bad = _check_ohlc(bar)
        if bad:
            issues.append(bad)
            continue
        rng = bar.high - bar.low
        if rng <= FLAT_RANGE_EPS * max(bar.close, 1.0):
            issues.append(
                BarIssue(bar.day, Issue.STALE,
                         f"o=h=l=c={bar.close:g} vol={bar.volume:g}")
            )
        if bar.volume <= 0:
            issues.append(BarIssue(bar.day, Issue.ZERO_VOLUME, "no volume"))

    # Spikes are judged against a local neighbourhood, so volatility regime
    # changes do not read as anomalies.
    closes = [b.close for b in bars]
    for i, bar in enumerate(bars):
        lo = max(0, i - SPIKE_WINDOW // 2)
        hi = min(len(bars), lo + SPIKE_WINDOW)
        window = bars[lo:hi]
        if len(window) < 5:
            continue

        # An isolated high or low that no neighbouring bar approaches.
        for attr, kind in (("high", Issue.SPIKE_HIGH), ("low", Issue.SPIKE_LOW)):
            vals = [getattr(b, attr) for b in window]
            z = modified_zscores(vals)[window.index(bar)]
            if abs(z) > SPIKE_THRESHOLD:
                issues.append(
                    BarIssue(bar.day, kind, f"{attr}={getattr(bar, attr):g} z={z:+.1f}")
                )

        if i > 0 and closes[i - 1] > 0:
            rets = [
                (window[j].close / window[j - 1].close - 1)
                for j in range(1, len(window))
                if window[j - 1].close > 0
            ]
            if len(rets) >= 5:
                r = closes[i] / closes[i - 1] - 1
                zs = modified_zscores(rets + [r])
                if abs(zs[-1]) > SPIKE_THRESHOLD:
                    issues.append(
                        BarIssue(bar.day, Issue.SPIKE_RETURN, f"ret={r:+.2%} z={zs[-1]:+.1f}")
                    )

    return QualityReport(
        symbol=series.symbol,
        n_bars=len(bars),
        issues=tuple(issues),
        missing_sessions=tuple(expected_sessions(bars)),
    )


def compare_coverage(primary: Series, secondary: Series) -> dict[str, int]:
    """Session coverage differences between two sources.

    Alpaca emits no bar when an interval has no trades *or only a single trade*,
    so on thin names IEX drops sessions entirely. A 20-bar window then spans a
    different calendar period in each source -- a silent misalignment that no
    price comparison reveals.
    """
    a = {b.day for b in primary.bars}
    b = {b.day for b in secondary.bars}
    return {
        "primary_bars": len(a),
        "secondary_bars": len(b),
        "shared": len(a & b),
        "primary_only": len(a - b),
        "secondary_only": len(b - a),
    }


def summarise(reports: Sequence[QualityReport]) -> dict[str, float | int]:
    if not reports:
        return {"symbols": 0}
    total_bars = sum(r.n_bars for r in reports)
    totals = {i.value: 0 for i in Issue}
    for r in reports:
        for k, v in r.counts.items():
            totals[k] += v
    flagged = sum(len(r.flagged_days) for r in reports)
    return {
        "symbols": len(reports),
        "bars": total_bars,
        "flagged_bars": flagged,
        "clean_rate": 1.0 - flagged / total_bars if total_bars else 0.0,
        **totals,
        "missing_sessions": sum(len(r.missing_sessions) for r in reports),
    }
