"""Technical indicators, computed from bars we already hold.

**Canonical parameters only — no sweeping.** Bollinger 20/2, RSI 14,
MACD 12/26/9, MA 50/200, ATR 14, Donchian 20. These are conventional values,
chosen precisely *because* they are conventional rather than because they
backtested well.

The reason is Sullivan, Timmermann & White (Journal of Finance, 1999). They took
the famous Brock/Lakonishok/LeBaron technical results, expanded the rule
universe, and applied White's Reality Check to 100 years of Dow data. The best
rule survived in-sample — and then failed completely out-of-sample, and showed no
outperformance at all on S&P futures once data-snooping was accounted for.
Technical rules are the textbook data-snooping case study, and their parameter
space is effectively unbounded: period × multiplier × entry × exit is thousands
of variants, each of which is a trial. Our own simulation showed 1,000 trials of
pure noise producing a best Sharpe of 2.33. Sweeping here would exhaust the
trial budget and make any genuine finding unrecognisable.

**Every function is causal.** A value at index i uses only bars 0..i. There is no
centring, no forward fill, and no lookahead — the mistakes that make published
TradingView strategies backtest beautifully and fail live. `None` is returned for
indices before enough history exists, rather than a partial value, so a strategy
cannot accidentally trade on a half-formed indicator.

**Bollinger's own position**, worth stating since it motivates the design: "There
is absolutely nothing about a tag of a band that in and of itself is a signal."
He is explicit that bands are not a standalone system. That is why every
indicator here exposes a `Selector` for use inside `all_of` — the interesting
hypotheses are conjunctions, not single indicators.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from .prices import Bar

# Conventional parameter sets. Change these only with a recorded reason.
BB_PERIOD, BB_STDEV = 20, 2.0
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
DONCHIAN_PERIOD = 20
MA_FAST, MA_SLOW = 50, 200

#: RSI levels in universal use.
RSI_OVERSOLD, RSI_OVERBOUGHT = 30.0, 70.0

#: A Bollinger "squeeze" is bandwidth in the lowest decile of its own history --
#: relative to the instrument, since absolute bandwidth is not comparable across
#: a $2 microcap and a $400 megacap.
SQUEEZE_PERCENTILE = 0.10


def _closes(bars: Sequence[Bar]) -> list[float]:
    return [b.close for b in bars]


def sma(values: Sequence[float], period: int) -> list[float | None]:
    """Simple moving average. None until `period` observations exist."""
    out: list[float | None] = [None] * len(values)
    if period <= 0:
        return out
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= period:
            total -= values[i - period]
        if i >= period - 1:
            out[i] = total / period
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average, seeded with the SMA of the first window.

    Seeding from an SMA rather than the first value avoids a long startup
    transient that would otherwise contaminate early signals.
    """
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    k = 2.0 / (period + 1)
    prev = statistics.fmean(values[:period])
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


@dataclass(frozen=True)
class BollingerBands:
    middle: list[float | None]
    upper: list[float | None]
    lower: list[float | None]
    bandwidth: list[float | None]
    percent_b: list[float | None]


def bollinger(bars: Sequence[Bar], period: int = BB_PERIOD,
              stdev: float = BB_STDEV) -> BollingerBands:
    """Bollinger Bands plus the two derived measures that carry the information.

    `bandwidth` = (upper - lower) / middle, which is scale-free and therefore
    comparable across instruments. `percent_b` places price within the bands:
    0 at the lower band, 1 at the upper, outside [0,1] beyond them.
    """
    closes = _closes(bars)
    mid = sma(closes, period)
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)
    width: list[float | None] = [None] * len(closes)
    pct: list[float | None] = [None] * len(closes)

    for i in range(len(closes)):
        m = mid[i]
        if m is None:
            continue
        window = closes[i - period + 1 : i + 1]
        sd = statistics.pstdev(window)
        u, lo = m + stdev * sd, m - stdev * sd
        upper[i], lower[i] = u, lo
        if m:
            width[i] = (u - lo) / m
        if u != lo:
            pct[i] = (closes[i] - lo) / (u - lo)
    return BollingerBands(mid, upper, lower, width, pct)


def rsi(bars: Sequence[Bar], period: int = RSI_PERIOD) -> list[float | None]:
    """Wilder's RSI, using his smoothing rather than a simple average."""
    closes = _closes(bars)
    out: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return out

    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


@dataclass(frozen=True)
class MACD:
    macd: list[float | None]
    signal: list[float | None]
    histogram: list[float | None]


def macd(bars: Sequence[Bar], fast: int = MACD_FAST, slow: int = MACD_SLOW,
         signal_period: int = MACD_SIGNAL) -> MACD:
    closes = _closes(bars)
    f, s = ema(closes, fast), ema(closes, slow)
    line: list[float | None] = [
        (f[i] - s[i]) if (f[i] is not None and s[i] is not None) else None
        for i in range(len(closes))
    ]
    # The signal line is an EMA of the MACD line, which only exists once the
    # slow EMA does -- so it is computed over the defined region and mapped back.
    defined = [(i, v) for i, v in enumerate(line) if v is not None]
    sig: list[float | None] = [None] * len(closes)
    if defined:
        sub = ema([v for _, v in defined], signal_period)
        for (idx, _), value in zip(defined, sub):
            sig[idx] = value
    hist: list[float | None] = [
        (line[i] - sig[i]) if (line[i] is not None and sig[i] is not None) else None
        for i in range(len(closes))
    ]
    return MACD(line, sig, hist)


def true_range(bars: Sequence[Bar]) -> list[float | None]:
    out: list[float | None] = [None] * len(bars)
    for i in range(1, len(bars)):
        b, p = bars[i], bars[i - 1]
        out[i] = max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close))
    return out


def atr(bars: Sequence[Bar], period: int = ATR_PERIOD) -> list[float | None]:
    """Average True Range, Wilder-smoothed.

    Reads highs and lows, so it inherits any phantom spike in them -- which is
    why `quality.validate` checks H/L before indicators are trusted.
    """
    tr = true_range(bars)
    out: list[float | None] = [None] * len(bars)
    vals = [t for t in tr if t is not None]
    if len(vals) < period:
        return out
    first = period  # index of the last bar in the seeding window
    prev = statistics.fmean(vals[:period])
    out[first] = prev
    for i in range(first + 1, len(bars)):
        t = tr[i]
        if t is None:
            continue
        prev = (prev * (period - 1) + t) / period
        out[i] = prev
    return out


@dataclass(frozen=True)
class Donchian:
    upper: list[float | None]
    lower: list[float | None]


def donchian(bars: Sequence[Bar], period: int = DONCHIAN_PERIOD) -> Donchian:
    """Donchian channel from the PRIOR `period` bars, excluding the current one.

    Including the current bar would make a breakout tautological -- today's high
    is always within a range that contains today's high. Excluding it is what
    makes "price exceeded the prior range" a real event.
    """
    up: list[float | None] = [None] * len(bars)
    lo: list[float | None] = [None] * len(bars)
    for i in range(period, len(bars)):
        window = bars[i - period : i]
        up[i] = max(b.high for b in window)
        lo[i] = min(b.low for b in window)
    return Donchian(up, lo)


def momentum(bars: Sequence[Bar], lookback: int = 252, skip: int = 21) -> list[float | None]:
    """Cross-sectional momentum, conventionally 12 months skipping the last one.

    The skip is not decoration: the most recent month exhibits short-term
    reversal, and including it materially weakens the effect Jegadeesh & Titman
    documented.
    """
    closes = _closes(bars)
    out: list[float | None] = [None] * len(closes)
    for i in range(lookback, len(closes)):
        start, end = closes[i - lookback], closes[i - skip]
        if start > 0:
            out[i] = end / start - 1
    return out


def percentile_rank(values: Sequence[float | None], i: int, lookback: int = 252) -> float | None:
    """Rank of values[i] within its own trailing history, in [0, 1].

    Causal by construction: only indices <= i are considered.
    """
    if i >= len(values) or values[i] is None:
        return None
    lo = max(0, i - lookback + 1)
    window = [v for v in values[lo : i + 1] if v is not None]
    if len(window) < 20:
        return None
    return sum(1 for v in window if v <= values[i]) / len(window)


# --- selectors ---------------------------------------------------------------
#
# These bind an indicator to a single bar index so a hypothesis can be expressed
# as a conjunction, e.g. all_of(insider_buy, bollinger_squeeze). Bollinger's own
# position is that a band tag is not a signal on its own; combinations are the
# point.

def bollinger_squeeze(bars: Sequence[Bar], i: int,
                      percentile: float = SQUEEZE_PERCENTILE) -> bool:
    """Bandwidth in the lowest decile of its own trailing history."""
    bb = bollinger(bars)
    r = percentile_rank(bb.bandwidth, i)
    return r is not None and r <= percentile


def bollinger_below_lower(bars: Sequence[Bar], i: int) -> bool:
    bb = bollinger(bars)
    return bb.percent_b[i] is not None and bb.percent_b[i] < 0.0


def rsi_oversold(bars: Sequence[Bar], i: int, level: float = RSI_OVERSOLD) -> bool:
    v = rsi(bars)[i]
    return v is not None and v <= level


def macd_bullish_cross(bars: Sequence[Bar], i: int) -> bool:
    """Histogram turned positive on this bar -- a crossing, not merely a state."""
    m = macd(bars)
    if i == 0 or m.histogram[i] is None or m.histogram[i - 1] is None:
        return False
    return m.histogram[i - 1] <= 0 < m.histogram[i]


def donchian_breakout(bars: Sequence[Bar], i: int) -> bool:
    d = donchian(bars)
    return d.upper[i] is not None and bars[i].close > d.upper[i]


def above_ma(bars: Sequence[Bar], i: int, period: int = MA_SLOW) -> bool:
    m = sma(_closes(bars), period)[i]
    return m is not None and bars[i].close > m
