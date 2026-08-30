"""Volume profile and order flow, built from intraday bars.

These are the two strategies that daily bars simply cannot express, so both are
computed here from `intraday.MinuteBar` and stored in reduced form.

**Volume profile** asks where volume actually traded, not where price closed.
The point of control is the price with the most volume; the value area is the
contiguous band around it holding 70% of volume. The classical Market Profile
claim is that price is drawn back to high-volume prices and moves quickly
through low-volume ones. That claim is not tested here -- it is *made testable*
here, which is the distinction that matters.

**Order flow** asks who was the aggressor. A trade printing at the ask was
someone lifting an offer; at the bid, someone hitting a bid. Delta is the
difference. We compute it two ways deliberately:

  tick rule     -- from minute bars, comparing each minute's close to the last.
                   Cheap enough to run over the whole universe.
  Lee-Ready     -- from trade prints against the prevailing NBBO quote. Exact,
                   but a full session of a liquid name is millions of prints.

The scalable one is the tick rule; the exact one exists to measure how much the
scalable one loses. `compare_classifiers` is that measurement, and running it
before trusting any delta-based result is the whole point of having both.

**A warning about resolution.** A minute bar is already an aggregation: a minute
containing 500 buys and 500 sells nets to whatever its close did, and the tick
rule sees one signed number. Minute-bar delta is therefore a low-resolution
proxy for order flow, not order flow. On a thin small cap with a handful of
prints per minute it is close to exact; on a liquid name it is not. That is why
`compare_classifiers` reports agreement per symbol rather than as one number.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from .intraday import PROFILE_BUCKETS, MinuteBar, SessionProfile

#: Fraction of volume defining the value area. 70% is the Market Profile
#: convention, chosen originally as roughly one standard deviation.
VALUE_AREA_FRACTION = 0.70

#: A trade at the midpoint cannot be signed by quote position; Lee-Ready falls
#: back to the tick rule for these, and we count them so the share is visible.
MIDPOINT_TOLERANCE = 1e-9

_FRACTION = re.compile(r"\.(\d+)")


def parse_ts(value: str) -> datetime:
    """Parse an Alpaca RFC-3339 timestamp to an aware UTC datetime.

    Not optional, and not replaceable by comparing the strings directly. Alpaca
    stamps trades and quotes with nanosecond precision but omits the fraction
    when it is zero, so `...:58Z` and `...:58.267Z` sort the wrong way round
    lexicographically ('.' is 0x2E, 'Z' is 0x5A). Ordering quotes against trades
    that way silently pairs some prints with a quote that came after them.

    Python parses at most microseconds, so the fraction is truncated rather than
    rounded -- truncating keeps a quote at or before its true time, which is the
    safe direction when the whole point is not to look ahead.
    """
    text = value.strip()
    m = _FRACTION.search(text)
    if m and len(m.group(1)) > 6:
        text = text[: m.start() + 7] + text[m.end():]
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


# --- session reduction --------------------------------------------------------

def build_profile(symbol: str, day: date, bars: Sequence[MinuteBar],
                  *, buckets: int = PROFILE_BUCKETS,
                  rth_only: bool = True) -> SessionProfile | None:
    """Reduce one session's minute bars to a stored profile.

    Volume is attributed to each minute's own VWAP rather than its close: a
    minute that opened at 10 and closed at 11 traded across that range, and
    putting all of it at 11 shifts the point of control toward wherever minutes
    happened to end.
    """
    bars = [b for b in bars if b.volume > 0]
    if not bars:
        return None

    low = min(b.low for b in bars)
    high = max(b.high for b in bars)
    total = sum(b.volume for b in bars)
    vwap = sum(b.typical * b.volume for b in bars) / total if total > 0 else low

    hist = [0.0] * buckets
    width = (high - low) / buckets if high > low else 0.0
    for b in bars:
        if width > 0:
            idx = int((b.typical - low) / width)
            idx = min(max(idx, 0), buckets - 1)
        else:
            idx = buckets // 2      # degenerate flat session
        hist[idx] += b.volume

    delta, unsigned = tick_rule_delta(bars)
    return SessionProfile(
        symbol=symbol.upper(), day=day, low=low, high=high, volume=total,
        vwap=vwap, histogram=tuple(hist), delta=delta, unsigned_volume=unsigned,
        minute_count=len(bars), rth_only=rth_only,
    )


# --- volume profile -----------------------------------------------------------

@dataclass(frozen=True)
class VolumeProfile:
    """A merged profile over one or more sessions."""

    low: float
    high: float
    buckets: tuple[float, ...]
    #: Price with the most traded volume.
    poc: float
    #: Value area bounds: the contiguous band around the POC holding
    #: VALUE_AREA_FRACTION of total volume.
    value_area_low: float
    value_area_high: float
    total_volume: float
    sessions: int

    @property
    def bucket_width(self) -> float:
        if self.high <= self.low or not self.buckets:
            return 0.0
        return (self.high - self.low) / len(self.buckets)

    def position_of(self, price: float) -> float | None:
        """Where a price sits in the profile, 0 at the low and 1 at the high."""
        if self.high <= self.low:
            return None
        return (price - self.low) / (self.high - self.low)

    def volume_at(self, price: float) -> float:
        """Volume traded in the bucket containing `price`.

        Used for low-volume-node detection: the Market Profile claim is that
        price traverses thin prices quickly, so a breakout into a volume void
        is a different event from one into thick prior trade.
        """
        w = self.bucket_width
        if w <= 0 or not (self.low <= price <= self.high):
            return 0.0
        idx = min(int((price - self.low) / w), len(self.buckets) - 1)
        return self.buckets[idx]


def merge_profiles(profiles: Sequence[SessionProfile],
                   *, buckets: int = PROFILE_BUCKETS) -> VolumeProfile | None:
    """Combine stored sessions onto a common price grid.

    Sessions have different ranges and therefore different bucket widths, so
    volume is redistributed by *overlap area* rather than by assigning each old
    bucket wholesale to whichever new bucket its midpoint lands in. Midpoint
    assignment produces visible comb artefacts when the widths are close, which
    then show up as a spurious point of control.
    """
    profiles = [p for p in profiles if p.volume > 0 and p.histogram]
    if not profiles:
        return None

    low = min(p.low for p in profiles)
    high = max(p.high for p in profiles)
    grid = [0.0] * buckets
    if high <= low:
        grid[buckets // 2] = sum(p.volume for p in profiles)
        return _finish(low, high, grid, len(profiles))

    new_width = (high - low) / buckets
    for p in profiles:
        old_width = p.bucket_width
        for i, vol in enumerate(p.histogram):
            if vol <= 0:
                continue
            if old_width <= 0:
                idx = min(int((p.low - low) / new_width), buckets - 1)
                grid[idx] += vol
                continue
            lo = p.low + old_width * i
            hi = lo + old_width
            first = max(int((lo - low) / new_width), 0)
            last = min(int((hi - low) / new_width), buckets - 1)
            for j in range(first, last + 1):
                b_lo = low + new_width * j
                b_hi = b_lo + new_width
                overlap = min(hi, b_hi) - max(lo, b_lo)
                if overlap > 0:
                    grid[j] += vol * (overlap / old_width)
    return _finish(low, high, grid, len(profiles))


def _finish(low: float, high: float, grid: list[float], sessions: int) -> VolumeProfile:
    total = sum(grid)
    width = (high - low) / len(grid) if high > low else 0.0
    poc_idx = max(range(len(grid)), key=lambda i: grid[i])
    poc = low + width * (poc_idx + 0.5) if width > 0 else low
    va_lo_idx, va_hi_idx = _value_area(grid, poc_idx, total)
    return VolumeProfile(
        low=low, high=high, buckets=tuple(grid), poc=poc,
        value_area_low=low + width * va_lo_idx if width > 0 else low,
        value_area_high=low + width * (va_hi_idx + 1) if width > 0 else high,
        total_volume=total, sessions=sessions,
    )


def _value_area(grid: Sequence[float], poc_idx: int, total: float) -> tuple[int, int]:
    """Expand from the POC until the band holds VALUE_AREA_FRACTION of volume.

    The standard construction compares the two buckets above against the two
    below and takes the heavier pair. Expanding one bucket at a time instead
    gives a subtly different band, so the convention is followed exactly.
    """
    if total <= 0:
        return poc_idx, poc_idx
    target = total * VALUE_AREA_FRACTION
    lo = hi = poc_idx
    covered = grid[poc_idx]
    n = len(grid)
    while covered < target and (lo > 0 or hi < n - 1):
        above = grid[hi + 1] + (grid[hi + 2] if hi + 2 < n else 0.0) if hi < n - 1 else -1.0
        below = grid[lo - 1] + (grid[lo - 2] if lo - 2 >= 0 else 0.0) if lo > 0 else -1.0
        if above >= below:
            for _ in range(2):
                if hi < n - 1:
                    hi += 1
                    covered += grid[hi]
        else:
            for _ in range(2):
                if lo > 0:
                    lo -= 1
                    covered += grid[lo]
    return lo, hi


# --- order flow ---------------------------------------------------------------

def tick_rule_delta(bars: Sequence[MinuteBar]) -> tuple[float, float]:
    """Sign each minute's volume by the direction of its close.

    Returns (delta, unsigned_volume). A minute closing above the previous close
    is treated as buyer-initiated, below as seller-initiated, unchanged as
    unsignable -- carried forward as `unsigned_volume` rather than silently
    dropped or arbitrarily assigned, because on thin names unchanged minutes are
    a large share of the session and hiding them would overstate confidence.
    """
    delta = unsigned = 0.0
    prev: float | None = None
    for b in bars:
        if prev is None or b.close == prev:
            unsigned += b.volume
        elif b.close > prev:
            delta += b.volume
        else:
            delta -= b.volume
        prev = b.close
    return delta, unsigned


def lee_ready(trades: Sequence[dict], quotes: Sequence[dict]) -> tuple[float, float]:
    """Classify individual prints against the prevailing NBBO.

    A print above the quote midpoint is buyer-initiated, below is
    seller-initiated, and exactly at the midpoint falls back to the tick rule --
    the original Lee & Ready (1991) construction.

    We use the *contemporaneous* quote rather than their five-second lag. That
    lag existed because 1980s trade reporting ran behind the quote feed; on
    modern nanosecond-stamped SIP data applying it degrades the classification
    instead of improving it.
    """
    if not trades:
        return 0.0, 0.0
    q_ts = [parse_ts(q["t"]) for q in quotes]
    delta = unsigned = 0.0
    prev_price: float | None = None
    for t in trades:
        price, size = t["p"], t["s"]
        mid: float | None = None
        if q_ts:
            i = bisect.bisect_right(q_ts, parse_ts(t["t"])) - 1
            if i >= 0:
                q = quotes[i]
                bid, ask = q.get("bp") or 0.0, q.get("ap") or 0.0
                if bid > 0 and ask > 0 and ask >= bid:
                    mid = (bid + ask) / 2.0
        if mid is not None and abs(price - mid) > MIDPOINT_TOLERANCE:
            delta += size if price > mid else -size
        elif prev_price is not None and price != prev_price:
            delta += size if price > prev_price else -size
        else:
            unsigned += size
        prev_price = price
    return delta, unsigned


def compare_classifiers(bars: Sequence[MinuteBar], trades: Sequence[dict],
                        quotes: Sequence[dict]) -> dict[str, float]:
    """Measure how far minute-bar delta drifts from tick-level Lee-Ready.

    Run this before trusting any delta-based backtest result. If the two
    disagree on sign, the cheap classifier is not measuring what the expensive
    one measures, and a strategy built on it is describing minute-bar closes
    rather than order flow.

    **The window is aligned first.** Trade pagination is capped for liquid names,
    so a raw comparison would put a truncated morning of prints against a whole
    session of bars and report a disagreement that is really a sampling
    artefact. Bars outside the trade window are dropped before comparing.
    """
    if trades:
        first, last = parse_ts(trades[0]["t"]), parse_ts(trades[-1]["t"])
        # A bar is stamped at the START of its minute and covers the following
        # 60 seconds, so it belongs in the comparison when that interval
        # OVERLAPS the trade window. Testing the stamp alone would drop the bar
        # containing the first trade whenever that trade is mid-minute.
        minute = timedelta(minutes=1)
        bars = [b for b in bars if b.ts <= last and b.ts + minute > first]
    minute_delta, minute_unsigned = tick_rule_delta(bars)
    exact_delta, exact_unsigned = lee_ready(trades, quotes)
    minute_vol = sum(b.volume for b in bars)
    exact_vol = sum(t["s"] for t in trades)
    return {
        "minute_delta": minute_delta,
        "exact_delta": exact_delta,
        "minute_delta_ratio": minute_delta / minute_vol if minute_vol else 0.0,
        "exact_delta_ratio": exact_delta / exact_vol if exact_vol else 0.0,
        "same_sign": float((minute_delta > 0) == (exact_delta > 0)),
        "minute_unsigned_share": minute_unsigned / minute_vol if minute_vol else 0.0,
        "exact_unsigned_share": exact_unsigned / exact_vol if exact_vol else 0.0,
        "minute_volume": minute_vol,
        "exact_volume": exact_vol,
    }


def with_exact_flow(profile: SessionProfile, trades: Sequence[dict],
                    quotes: Sequence[dict]) -> SessionProfile:
    """Replace a profile's tick-rule delta with Lee-Ready classification.

    This is the delta an order-flow hypothesis should be built on. The minute-bar
    tick rule is retained only as a cheap fallback and as a separate hypothesis
    in its own right -- it measures the direction of minute closes, which is a
    real thing, just not order flow.

    Affordable exactly where it matters: insider buying concentrates in small
    caps, and a full session of those runs 127 to 1,227 prints (XELB 739, GNSS
    1,227, RCG 127). A mega-cap session is millions and is not attempted.
    """
    delta, unsigned = lee_ready(trades, quotes)
    return SessionProfile(
        symbol=profile.symbol, day=profile.day, low=profile.low, high=profile.high,
        volume=profile.volume, vwap=profile.vwap, histogram=profile.histogram,
        delta=delta, unsigned_volume=unsigned, minute_count=profile.minute_count,
        rth_only=profile.rth_only, flow_method="lee_ready",
    )


def _require_one_method(profiles: Sequence[SessionProfile]) -> None:
    """Refuse to aggregate deltas produced by different classifiers.

    Measured on real prints, the two agree on sign only about a quarter of the
    time. Summing them would average a measurement against its own error and
    produce a number belonging to neither, so this raises rather than warns.
    """
    methods = {p.flow_method for p in profiles}
    if len(methods) > 1:
        raise ValueError(
            f"mixed flow classifiers in one window: {sorted(methods)}. "
            "tick_minute and lee_ready disagree on sign roughly three times in "
            "four; aggregating them is not meaningful."
        )


def cumulative_delta(profiles: Sequence[SessionProfile]) -> list[float]:
    """Running sum of session deltas, oldest first."""
    _require_one_method(profiles)
    out, running = [], 0.0
    for p in profiles:
        running += p.delta
        out.append(running)
    return out


def delta_ratio(profiles: Sequence[SessionProfile]) -> float | None:
    """Net signed volume as a share of total volume over the window.

    Scale-free, so it is comparable across a micro-cap and a mega-cap -- the
    same reason `bollinger.bandwidth` is expressed as a ratio.
    """
    _require_one_method(profiles)
    total = sum(p.volume for p in profiles)
    if total <= 0:
        return None
    return sum(p.delta for p in profiles) / total


# --- selectors ----------------------------------------------------------------
#
# Bound to a window of stored sessions plus the price being evaluated, so they
# compose with the daily selectors through `backtest.all_of`.

def above_poc(profiles: Sequence[SessionProfile], price: float) -> bool:
    vp = merge_profiles(profiles)
    return vp is not None and price > vp.poc


def below_value_area(profiles: Sequence[SessionProfile], price: float) -> bool:
    """Price below the value area: the classic mean-reversion setup, since the
    Market Profile claim is that price returns to accepted value."""
    vp = merge_profiles(profiles)
    return vp is not None and price < vp.value_area_low


def above_value_area(profiles: Sequence[SessionProfile], price: float) -> bool:
    vp = merge_profiles(profiles)
    return vp is not None and price > vp.value_area_high


def in_low_volume_node(profiles: Sequence[SessionProfile], price: float,
                       percentile: float = 0.25) -> bool:
    """Price sitting where little volume traded.

    The Market Profile claim is that price moves quickly through thin prices, so
    a signal firing in a volume void should behave differently from one firing
    in thick prior trade. Testing that difference is the point.
    """
    vp = merge_profiles(profiles)
    if vp is None or vp.total_volume <= 0:
        return False
    occupied = sorted(v for v in vp.buckets if v > 0)
    if not occupied:
        return False
    here = vp.volume_at(price)
    cut = occupied[max(int(len(occupied) * percentile) - 1, 0)]
    return here <= cut


def positive_delta(profiles: Sequence[SessionProfile], minimum: float = 0.10) -> bool:
    """Net buying pressure over the window, as a share of volume."""
    r = delta_ratio(profiles)
    return r is not None and r >= minimum


def delta_divergence(profiles: Sequence[SessionProfile]) -> bool:
    """Price made a lower low while cumulative delta did not.

    The textbook absorption pattern: sellers pushed price to a new low but were
    met, so the flow did not confirm the move. Needs at least two sessions of
    history on each side to mean anything.
    """
    if len(profiles) < 4:
        return False
    mid = len(profiles) // 2
    first, second = profiles[:mid], profiles[mid:]
    price_lower = min(p.low for p in second) < min(p.low for p in first)
    cd = cumulative_delta(profiles)
    delta_higher = min(cd[mid:]) > min(cd[:mid])
    return price_lower and delta_higher
