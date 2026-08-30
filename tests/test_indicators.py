"""Tests for technical indicators.

The causality tests matter most. A non-causal indicator makes a backtest look
excellent and fail live — the exact failure mode that makes published
TradingView strategies unreliable.
"""

from __future__ import annotations

import statistics
from datetime import date, timedelta

import pytest

from tradezbotz.research.indicators import (
    above_anchored_vwap,
    anchored_vwap,
    anchored_vwap_from_day,
    atr,
    above_ma,
    bollinger,
    bollinger_squeeze,
    donchian,
    donchian_breakout,
    ema,
    macd,
    momentum,
    percentile_rank,
    rsi,
    rsi_oversold,
    relative_volume,
    sma,
    swept_high,
    swept_low,
    true_range,
)
from tradezbotz.research.prices import Bar


def bars_from(closes, highs=None, lows=None, start=date(2025, 1, 1)):
    out, day = [], start
    for i, c in enumerate(closes):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        h = highs[i] if highs else c * 1.01
        lo = lows[i] if lows else c * 0.99
        out.append(Bar(day, c, max(h, c), min(lo, c), c, 1_000_000))
        day += timedelta(days=1)
    return out


def rising(n=300, start=100.0, step=0.5):
    return [start + i * step for i in range(n)]


# --- moving averages ---------------------------------------------------------

def test_sma_is_none_until_the_window_fills():
    out = sma([1, 2, 3, 4, 5], 3)

    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(2.0)
    assert out[4] == pytest.approx(4.0)


def test_sma_uses_only_past_and_present():
    """Causality: appending future bars must not change an earlier value."""
    vals = [float(i) for i in range(50)]
    short = sma(vals[:30], 10)
    long = sma(vals, 10)

    assert short[29] == long[29]


def test_ema_seeds_from_sma_not_first_value():
    vals = [10.0] * 5 + [20.0] * 20
    out = ema(vals, 5)

    assert out[:4] == [None] * 4
    assert out[4] == pytest.approx(10.0)


def test_ema_is_causal():
    vals = [float(i % 7) for i in range(60)]

    assert ema(vals[:40], 12)[39] == ema(vals, 12)[39]


# --- bollinger ---------------------------------------------------------------

def test_bollinger_bands_bracket_price():
    bb = bollinger(bars_from(rising(60)))
    i = 50

    assert bb.lower[i] < bb.middle[i] < bb.upper[i]


def test_bollinger_is_none_before_the_period():
    bb = bollinger(bars_from(rising(40)), period=20)

    assert bb.middle[18] is None
    assert bb.middle[19] is not None


def test_percent_b_locates_price_within_bands():
    bb = bollinger(bars_from(rising(60)))
    i = 50

    assert 0.0 <= bb.percent_b[i] <= 1.5


def test_bandwidth_is_scale_free():
    """Bandwidth must be comparable between a $2 microcap and a $400 megacap,
    or a squeeze threshold means different things for different instruments."""
    cheap = bollinger(bars_from([2 + (i % 5) * 0.02 for i in range(60)]))
    dear = bollinger(bars_from([400 + (i % 5) * 4.0 for i in range(60)]))

    assert cheap.bandwidth[50] == pytest.approx(dear.bandwidth[50], rel=0.05)


def test_bollinger_is_causal():
    closes = rising(80)
    short = bollinger(bars_from(closes[:60]))
    long = bollinger(bars_from(closes))

    assert short.upper[59] == pytest.approx(long.upper[59])


# --- rsi ---------------------------------------------------------------------

def test_rsi_is_100_when_every_move_is_up():
    assert rsi(bars_from(rising(40)))[30] == pytest.approx(100.0)


def test_rsi_is_low_when_every_move_is_down():
    falling = [200 - i for i in range(40)]

    assert rsi(bars_from(falling))[30] < 5.0


def test_rsi_stays_in_range():
    import random
    rng = random.Random(3)
    closes, p = [], 100.0
    for _ in range(200):
        p *= 1 + rng.gauss(0, 0.02)
        closes.append(p)

    vals = [v for v in rsi(bars_from(closes)) if v is not None]

    assert vals and all(0.0 <= v <= 100.0 for v in vals)


def test_rsi_is_causal():
    closes = rising(60)

    assert rsi(bars_from(closes[:40]))[39] == pytest.approx(rsi(bars_from(closes))[39])


# --- macd --------------------------------------------------------------------

def test_macd_positive_in_an_uptrend():
    m = macd(bars_from(rising(120)))

    assert m.macd[100] > 0


def test_macd_histogram_is_line_minus_signal():
    m = macd(bars_from(rising(120)))
    i = 100

    assert m.histogram[i] == pytest.approx(m.macd[i] - m.signal[i])


def test_macd_is_causal():
    closes = rising(150)

    assert macd(bars_from(closes[:120])).macd[119] == pytest.approx(
        macd(bars_from(closes)).macd[119])


# --- atr / true range --------------------------------------------------------

def test_true_range_accounts_for_gaps():
    """A gap open means the true range exceeds the bar's own high-low."""
    bars = [Bar(date(2025, 1, 1), 100, 101, 99, 100, 1e6),
            Bar(date(2025, 1, 2), 120, 121, 119, 120, 1e6)]

    assert true_range(bars)[1] == pytest.approx(21.0)


def test_atr_is_positive_and_delayed():
    a = atr(bars_from(rising(60)))

    assert a[5] is None
    assert a[50] > 0


def test_atr_is_causal():
    closes = rising(80)

    assert atr(bars_from(closes[:60]))[59] == pytest.approx(atr(bars_from(closes))[59])


# --- donchian ----------------------------------------------------------------

def test_donchian_excludes_the_current_bar():
    """Including it would make breakout tautological: today's high is always
    inside a range that contains today's high."""
    closes = [100.0] * 30 + [200.0]
    highs = [101.0] * 30 + [201.0]
    bars = bars_from(closes, highs=highs)

    d = donchian(bars, period=20)

    assert d.upper[30] == pytest.approx(101.0), "prior range only"
    assert donchian_breakout(bars, 30) is True


def test_no_breakout_inside_the_range():
    bars = bars_from([100.0] * 40)

    assert donchian_breakout(bars, 35) is False


# --- momentum ----------------------------------------------------------------

def test_momentum_skips_the_recent_month():
    """The skip is load-bearing: the last month shows short-term reversal, and
    including it materially weakens the documented effect."""
    closes = rising(400)
    m = momentum(bars_from(closes), lookback=252, skip=21)
    i = 300

    assert m[i] == pytest.approx(closes[i - 21] / closes[i - 252] - 1)


def test_momentum_none_before_lookback():
    assert momentum(bars_from(rising(300)), lookback=252)[100] is None


# --- percentile rank ---------------------------------------------------------

def test_percentile_rank_is_causal():
    vals = [float(i) for i in range(300)]

    assert percentile_rank(vals, 100) == pytest.approx(1.0), "highest so far"


def test_percentile_rank_needs_enough_history():
    assert percentile_rank([1.0, 2.0, 3.0], 2) is None


# --- selectors ---------------------------------------------------------------

def test_squeeze_fires_when_volatility_compresses():
    import random
    rng = random.Random(5)
    closes, p = [], 100.0
    for i in range(400):
        vol = 0.03 if i < 340 else 0.0005      # sharp compression at the end
        p *= 1 + rng.gauss(0, vol)
        closes.append(p)
    bars = bars_from(closes)

    assert bollinger_squeeze(bars, 395) is True
    assert bollinger_squeeze(bars, 200) is False


def test_rsi_oversold_selector():
    falling = [200 - i * 2 for i in range(60)]

    assert rsi_oversold(bars_from(falling), 50) is True
    assert rsi_oversold(bars_from(rising(60)), 50) is False


def test_above_ma_selector():
    bars = bars_from(rising(300))

    assert above_ma(bars, 280, period=200) is True
    assert above_ma(bars, 50, period=200) is False, "None before the window fills"


def test_selectors_are_safe_on_short_series():
    """Must not raise on a symbol with too little history."""
    bars = bars_from(rising(5))

    assert bollinger_squeeze(bars, 4) is False
    assert rsi_oversold(bars, 4) is False
    assert donchian_breakout(bars, 4) is False
    assert above_ma(bars, 4) is False


# --- relative volume ---------------------------------------------------------

def test_relative_volume_uses_median_not_mean():
    """One earnings spike in the window must not mask the next real surge."""
    bars = [Bar(date(2025, 1, 1), 100, 101, 99, 100, 1_000) for _ in range(20)]
    bars[5] = Bar(date(2025, 1, 6), 100, 101, 99, 100, 100_000)   # outlier
    bars.append(Bar(date(2025, 2, 1), 100, 101, 99, 100, 2_000))

    assert relative_volume(bars, 20) == pytest.approx(2.0), "median ignores the spike"


def test_relative_volume_none_before_window():
    bars = bars_from(rising(30))
    assert relative_volume(bars, 5) is None


# --- anchored vwap -----------------------------------------------------------

def test_anchored_vwap_starts_at_the_anchor_bar():
    bars = bars_from(rising(30))
    v = anchored_vwap(bars, 10)

    assert v[9] is None, "nothing before the anchor"
    assert v[10] is not None


def test_anchored_vwap_uses_source_vwap_when_present():
    bars = [Bar(date(2025, 1, 1), 100, 110, 90, 100, 1_000, vwap=105.0),
            Bar(date(2025, 1, 2), 100, 110, 90, 100, 1_000, vwap=95.0)]

    v = anchored_vwap(bars, 0)

    assert v[0] == pytest.approx(105.0)
    assert v[1] == pytest.approx(100.0), "volume-weighted mean of 105 and 95"


def test_anchored_vwap_falls_back_to_typical_price():
    bars = [Bar(date(2025, 1, 1), 100, 120, 60, 90, 1_000)]

    assert anchored_vwap(bars, 0)[0] == pytest.approx(90.0), "(120+60+90)/3"


def test_anchored_vwap_weights_by_volume():
    """A huge-volume bar must dominate a thin one."""
    bars = [Bar(date(2025, 1, 1), 10, 10, 10, 10, 1),
            Bar(date(2025, 1, 2), 20, 20, 20, 20, 999_999)]

    assert anchored_vwap(bars, 0)[1] == pytest.approx(20.0, abs=0.001)


def test_anchored_vwap_is_causal():
    """Appending later bars must not move an earlier anchored value."""
    closes = rising(60)
    short = anchored_vwap(bars_from(closes[:40]), 10)
    long = anchored_vwap(bars_from(closes), 10)

    assert short[39] == pytest.approx(long[39])


def test_anchored_vwap_from_day_takes_next_session():
    """A Friday-evening filing anchors to Monday, not to a non-existent bar."""
    bars = bars_from(rising(20), start=date(2025, 1, 6))   # Mon 6 Jan onward
    v = anchored_vwap_from_day(bars, date(2025, 1, 4))     # a Saturday

    assert v[0] is not None, "anchored to the first session on or after"


def test_anchored_vwap_from_day_before_all_bars_is_all_none():
    bars = bars_from(rising(10), start=date(2025, 1, 6))

    assert anchored_vwap_from_day(bars, date(2030, 1, 1)) == [None] * 10


def test_above_anchored_vwap_selector():
    bars = bars_from(rising(40))

    assert above_anchored_vwap(bars, 35, anchor=10) is True, "uptrend sits above"
    falling = bars_from([200 - i * 2 for i in range(40)])
    assert above_anchored_vwap(falling, 35, anchor=10) is False


# --- liquidity sweeps --------------------------------------------------------

def sweep_bars(final, base_vol=1_000, n=25):
    """A flat 100-104 range, then one bar that does something interesting."""
    out = [Bar(date(2025, 1, 1) + timedelta(days=i), 102, 104, 100, 102, base_vol)
           for i in range(n)]
    out.append(final)
    return out


def test_swept_low_fires_on_a_stop_run_that_reclaims():
    bars = sweep_bars(Bar(date(2025, 3, 1), 101, 103, 95, 102, 3_000))

    assert swept_low(bars, 25) is True


def test_swept_low_ignores_a_genuine_breakdown():
    """Closing below the level is a breakdown, not a sweep. The reclaim is the
    entire signal -- without it this is just a down day."""
    bars = sweep_bars(Bar(date(2025, 3, 1), 101, 103, 95, 96, 3_000))

    assert swept_low(bars, 25) is False


def test_swept_low_requires_conviction_volume():
    bars = sweep_bars(Bar(date(2025, 3, 1), 101, 103, 95, 102, 1_000))

    assert swept_low(bars, 25) is False, "no volume surge, so noise"


def test_swept_low_ignores_a_bar_inside_the_range():
    bars = sweep_bars(Bar(date(2025, 3, 1), 101, 103, 100.5, 102, 5_000))

    assert swept_low(bars, 25) is False, "never reached the liquidity"


def test_swept_high_fires_on_a_failed_breakout():
    bars = sweep_bars(Bar(date(2025, 3, 1), 103, 110, 102, 103, 3_000))

    assert swept_high(bars, 25) is True


def test_swept_high_ignores_a_breakout_that_holds():
    bars = sweep_bars(Bar(date(2025, 3, 1), 103, 110, 102, 109, 3_000))

    assert swept_high(bars, 25) is False


def test_sweep_and_breakout_partition_the_same_event():
    """A bar piercing the prior high is either a breakout or a sweep, never
    both. If they overlapped, a breakout result could really be a sweep result."""
    held = sweep_bars(Bar(date(2025, 3, 1), 103, 110, 102, 109, 3_000))
    failed = sweep_bars(Bar(date(2025, 3, 1), 103, 110, 102, 103, 3_000))

    assert donchian_breakout(held, 25) is True
    assert swept_high(held, 25) is False
    assert donchian_breakout(failed, 25) is False
    assert swept_high(failed, 25) is True


def test_sweeps_are_safe_on_short_series():
    bars = bars_from(rising(5))

    assert swept_low(bars, 4) is False
    assert swept_high(bars, 4) is False
    assert relative_volume(bars, 4) is None
