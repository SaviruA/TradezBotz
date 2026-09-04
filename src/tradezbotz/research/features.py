"""Attach point-in-time indicator values to event payloads.

A `Selector` is `(payload, label) -> bool`. It never sees bars. That signature is
deliberate -- it keeps selection cheap and composable -- but it means every
indicator in `indicators.py` was unreachable from a backtest: there was no path
by which `rsi_oversold` could ever be asked about a specific event. This module
is that path. It computes each indicator once per (symbol, entry day) and writes
the answer into the payload, where `field_equals` and `threshold` can find it.

**The evaluation index is the bar BEFORE entry, and that is the whole point.**
`Label.entry_day` is the session whose *open* we trade. At the moment that order
is sent, the only thing known about the entry session is that it has not started.
Its close, high, low and volume are all in the future. So the bars handed to any
indicator are truncated to `day < entry_day` and the indicator is read at the
last of those.

Getting this wrong is not a small error. Evaluating `donchian_breakout` on the
entry bar itself asks "did price break out during the session I am about to buy
the open of" -- a question whose answer is only available after the fact, and one
that correlates directly with the return being measured. It would produce a
strong, entirely fictitious edge on every momentum feature at once. The
truncation below is the only thing preventing that, so it happens in one place
rather than being re-derived per feature.

**Cost.** Every feature is recomputed from raw bars per event, which is slow in
the way pure Python is slow. Two things keep it tractable: the trailing window is
capped at `FEATURE_BARS` (enough for the longest 252-session lookback, nothing
more), and results are memoised on (symbol, entry_day) -- insider filings cluster
heavily on the same name and day, so the memo carries most of the load.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Sequence

from . import indicators as ind
from .labeler import Label
from .prices import Bar, Series

#: Trailing bars retained for feature computation. The longest lookback here is
#: `above_ma` at 200 and `distance_from_high` at 252; 300 covers both with room
#: for the leading `None`s every windowed indicator emits.
FEATURE_BARS = 300

#: Calendar days to request so `FEATURE_BARS` sessions come back. Sessions run
#: about 7 per 10 calendar days; the margin absorbs holiday clusters.
FEATURE_LOOKBACK_DAYS = int(FEATURE_BARS * 10 / 7) + 30

#: Minimum prior bars before any feature is computed. Below this the windowed
#: indicators return `None` anyway, and a payload full of `False` from missing
#: history is indistinguishable from one where the condition genuinely failed --
#: so the features are simply omitted and `has_features` stays False.
MIN_FEATURE_BARS = 60

#: Every key this module writes, so a caller can tell "feature absent" from
#: "feature False" without guessing at names.
BOOLEAN_FEATURES = (
    "rsi_oversold",
    "bb_squeeze",
    "bb_below_lower",
    "macd_cross",
    "donchian_breakout",
    "above_ma_200",
    "swept_low",
    "swept_high",
    "near_high",
    "gain_over_10",
    "engulfing_bull",
    "ttm_squeeze",
    "connors_rsi2",
    "engulfing_reversal",
)

NUMERIC_FEATURES = ("distance_from_high", "rvol", "momentum_12_1")


def prior_bars(bars: Sequence[Bar], entry_day: date) -> tuple[Bar, ...]:
    """Bars fully known before the entry session opens.

    Strictly `day < entry_day`. The entry bar is excluded even though it exists
    in the series, because at decision time it does not.
    """
    out = [b for b in bars if b.day < entry_day]
    return tuple(out[-FEATURE_BARS:])


def features_at(bars: Sequence[Bar]) -> dict:
    """Every indicator, read at the last bar of `bars`.

    `bars` must already be truncated by `prior_bars`. This function has no way
    to check that and does not try: it evaluates at `len(bars) - 1` and trusts
    the caller, which is why the truncation lives in one place above.
    """
    i = len(bars) - 1
    if i < MIN_FEATURE_BARS:
        return {}

    dist = ind.distance_from_high(bars, i)
    rvol = ind.relative_volume(bars, i)
    mom = ind.momentum(bars)[i]

    # Liquidity, read from the same prior window as everything else here.
    #
    # Not an indicator -- a tradability measure, and the one the whole cost
    # problem turns on. 82 of 232 verdicts in the 5.5-year sweep were "costs
    # exceed edge", and the published insider result says the same thing:
    # abnormal returns "vanish and even become negative" once the tradable
    # dollar amount is held to a reasonable size, being "negatively correlated
    # with stock liquidity". Without these two fields there was no way to
    # express "the same signal, in names where a round trip does not eat it".
    window = bars[max(0, i - 19):i + 1]
    dollar_volumes = [b.close * b.volume for b in window if b.volume and b.close]

    out: dict = {
        "has_features": True,
        "entry_close": bars[i].close,
        "dollar_volume_20d": (sum(dollar_volumes) / len(dollar_volumes)
                              if dollar_volumes else 0.0),
        "rsi_oversold": ind.rsi_oversold(bars, i),
        "bb_squeeze": ind.bollinger_squeeze(bars, i),
        "bb_below_lower": ind.bollinger_below_lower(bars, i),
        "macd_cross": ind.macd_bullish_cross(bars, i),
        "donchian_breakout": ind.donchian_breakout(bars, i),
        "above_ma_200": ind.above_ma(bars, i),
        "swept_low": ind.swept_low(bars, i),
        "swept_high": ind.swept_high(bars, i),
        "near_high": ind.near_high(bars, i),
        "gain_over_10": ind.gain_over(bars, i),
        "engulfing_bull": ind.engulfing(bars, i, bullish=True),
        "ttm_squeeze": ind.ttm_squeeze(bars, i),
        "connors_rsi2": ind.connors_rsi2(bars, i),
        "engulfing_reversal": ind.engulfing_reversal(bars, i),
    }
    if dist is not None:
        out["distance_from_high"] = dist
    if rvol is not None:
        out["rvol"] = rvol
    if mom is not None:
        out["momentum_12_1"] = mom
    return out


class FeatureBuilder:
    """Enriches payloads in place, fetching each symbol's bars at most once.

    Holds one `Series` per symbol for the lifetime of the run. That is the
    memory/time trade this makes explicitly: the whole labelling window over a
    few thousand symbols is on the order of a few million bars, which fits, and
    the alternative is a cache round trip per event.
    """

    def __init__(self, cache, basis: str | None = None) -> None:
        self.cache = cache
        self.basis = basis
        self._series: dict[str, Series] = {}
        self._memo: dict[tuple[str, date], dict] = {}
        self.enriched = 0
        self.skipped_no_bars = 0
        self.skipped_short_history = 0

    def _bars(self, symbol: str, entry_day: date) -> Series:
        series = self._series.get(symbol)
        if series is None:
            kwargs = {} if self.basis is None else {"basis": self.basis}
            series = self.cache.get(
                symbol,
                entry_day - timedelta(days=FEATURE_LOOKBACK_DAYS * 4),
                entry_day,
                **kwargs,
            )
            self._series[symbol] = series
        return series

    def features(self, label: Label) -> dict:
        if not label.symbol or label.entry_day is None:
            return {}
        key = (label.symbol, label.entry_day)
        hit = self._memo.get(key)
        if hit is not None:
            return hit

        series = self._bars(label.symbol, label.entry_day)
        if not series.bars:
            self.skipped_no_bars += 1
            self._memo[key] = {}
            return {}
        window = prior_bars(series.bars, label.entry_day)
        computed = features_at(window)
        if not computed:
            self.skipped_short_history += 1
        else:
            self.enriched += 1
        self._memo[key] = computed
        return computed

    def enrich(self, payloads: Sequence[dict], labels: Sequence[Label]) -> list[dict]:
        """Return payloads with features merged in.

        New dicts rather than mutation: the payloads come out of the event store
        and represent what was filed. A feature is our computation about that
        filing, not part of it, and conflating the two is how a derived column
        ends up looking like source data three months later.
        """
        out = []
        for payload, label in zip(payloads, labels):
            merged = dict(payload)
            merged.update(self.features(label))
            out.append(merged)
        return out

    def summary(self) -> str:
        return (
            f"features: {self.enriched:,} (symbol, day) pairs computed; "
            f"{self.skipped_no_bars:,} had no cached bars, "
            f"{self.skipped_short_history:,} had under {MIN_FEATURE_BARS} "
            "sessions of prior history"
        )
