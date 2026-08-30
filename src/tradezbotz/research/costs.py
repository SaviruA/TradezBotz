"""Transaction costs, without which every result here is fiction.

The backtest measured returns as if fills were free, on the population where
fills are worst. Practitioners are blunt about this: small-cap spreads run five
to ten times wider than large-cap ones, and "the backtest always looks perfectly
liquid." A 6.3% mean abnormal return is a real edge; the same number minus a 3%
round trip is not.

**Three costs, measured separately** because they scale differently and because
lumping them into one fudge factor hides which one is binding:

  spread    crossing the bid-ask. Roughly fixed per trade, dominant at our size.
  impact    moving the price by trading. Scales with order size over volume.
  fees      commission and regulatory. Zero-commission at Alpaca; SEC/FINRA fees
            apply on sells and are small but not nil.

**Spread estimation uses EDGE** (Ardia, Guidotti & Kroencke, *Journal of
Financial Economics* 2024). Chosen over Roll (1984), Corwin-Schultz (2012) and
Abdi-Ranaldo (2017) for one reason that matters here specifically: those three
are biased when trading is infrequent, and infrequent trading is the defining
property of our universe. EDGE is derived without assuming continuous trading
and uses the full OHLC information set rather than closes or the high-low range
alone.

**What EDGE cannot do**, and it matters here: it cannot separate a wide spread
from a volatile price path. Both widen the observed high-low range, and the
estimator attributes all of it to the spread. A name that gaps on earnings reads
as expensive to trade for as long as that gap sits inside the estimation window.
So the estimate is an upper bound on cost, never a measurement of it, and where
real quotes exist they always win.

We call the authors' own implementation (`bidask`, MIT) rather than reimplement
it. The estimator is a GMM combination of two moment conditions with variance-
optimal weights; a sign error in a hand port would not fail loudly, it would
quietly bias every cost estimate in the study.

**The estimate is validated against ground truth, not trusted.** Where the
intraday store has NBBO quotes we can compute the realised effective spread and
compare. That is the same discipline applied to order flow, where a cheap
minute-bar classifier turned out to disagree with tick-level Lee-Ready on sign
three times in four. A cheap estimator is worth having only once you know how
far it drifts from the expensive one.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from .prices import Bar

#: Below this true spread, EDGE is not usable as a point estimate. Measured
#: against realised NBBO effective spreads on 2026-08-24, across every
#: estimation window from 21 to 504 sessions:
#:
#:     symbol   real bps   EDGE bps (w=252)   ratio
#:     AAPL          5.5               28.4    5.2x
#:     AMAT         16.1              153.9    9.6x
#:     AXR         139.2              176.0    1.3x
#:     XELB        223.6              333.2    1.5x
#:
#: This is the authors' own documented caveat, not a surprise: negative squared
#: spreads are truncated at zero, which overstates the estimate when the true
#: spread is 0.50% or smaller. Widening the window does not fix it -- AAPL reads
#: 28-81bps at every window against a real 5.5.
#:
#: **The direction of the error is the useful part.** EDGE errs high, so a
#: strategy that survives an EDGE-based cost survives the real one. It is a
#: conservative bound rather than a measurement, and it is close to correct in
#: precisely the illiquid regime our universe occupies. Treat an estimate below
#: this threshold as "cheap to trade, magnitude unknown", never as a number.
EDGE_RELIABLE_FLOOR_BPS = 50.0

#: Minimum bars before EDGE means anything. The estimator needs at least three
#: observations; the authors note that short windows produce more negative
#: estimates, and truncating those at zero biases the spread upward. 21 sessions
#: is a month of trading and keeps that bias small.
MIN_SPREAD_WINDOW = 21

#: Default estimation window, one trading year.
#:
#: 63 sessions was the first choice and it was worse. EDGE cannot distinguish a
#: wide spread from a violent price path, so a single earnings gap inside a short
#: window dominates the estimate. DOCS -- 3.1M shares a day at $25, unmistakably
#: liquid -- gapped 32.6% on 2026-08-07 and read 576bps at a 63-session window
#: against 295bps at 252. Measured against real NBBO spreads the longer window is
#: better on three of four names tested.
#:
#: The trade-off is real: a year-long window averages away a genuine liquidity
#: regime change. That is the lesser error when the output is a conservative
#: bound rather than a measurement.
SPREAD_WINDOW = 252

#: Exponent on participation rate in the impact model. 0.5 is the widely
#: replicated square-root law.
#:
#: Deliberately a parameter rather than a constant: the square-root law is
#: documented for large and medium caps, while European small caps have been
#: found to follow something closer to a *square* relation. Our universe is the
#: small-cap end, so the default is likely optimistic. `IMPACT_EXPONENT_SMALL`
#: exists to test that sensitivity rather than to assert an answer.
IMPACT_EXPONENT = 0.5
IMPACT_EXPONENT_SMALL = 1.0

#: Coefficient on the square-root impact term. Published calibrations cluster
#: around 0.5-1.0 times daily volatility; 1.0 is the conservative end.
IMPACT_COEFFICIENT = 1.0

#: SEC Section 31 fee plus FINRA TAF, charged on sales only. Small, but not zero,
#: and they are the only unavoidable per-trade cost on a zero-commission broker.
SELL_FEE_RATE = 0.0000278  # combined, order of magnitude

#: Above this participation rate an order is not realistically executable in one
#: session without moving the market against itself. Flagged rather than priced:
#: pretending a 30% participation fill is merely expensive is worse than saying
#: it is infeasible.
MAX_PARTICIPATION = 0.10


class CostError(RuntimeError):
    pass


def edge_spread(bars: Sequence[Bar], window: int = SPREAD_WINDOW) -> float | None:
    """Proportional bid-ask spread over the last `window` bars, via EDGE.

    Returns a fraction: 0.01 means a 1% spread. `None` when there is too little
    history to estimate rather than a guess, because a missing cost silently
    becomes a zero cost otherwise.
    """
    if len(bars) < MIN_SPREAD_WINDOW:
        return None
    try:
        from bidask import edge
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CostError(
            "The EDGE spread estimator needs the reference implementation:\n"
            "    pip install bidask\n"
            "It is the authors' own package (MIT) and depends only on numpy "
            "and pandas, which are already required."
        ) from exc

    w = bars[-window:] if window else bars
    value = edge(
        [b.open for b in w], [b.high for b in w],
        [b.low for b in w], [b.close for b in w],
    )
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    # EDGE truncates negative squared spreads at zero, so a zero here means
    # "indistinguishable from no spread", not "free to trade". Callers should
    # still apply a floor; see CostModel.floor_bps.
    return float(value)


def effective_spread(trades: Sequence[dict], quotes: Sequence[dict]) -> float | None:
    """Realised proportional effective spread from actual prints and quotes.

    The standard microstructure definition: 2|P - M| / M, where P is the trade
    price and M the prevailing midpoint. This is what a taker actually paid, and
    it is the ground truth EDGE is estimating.

    Doubling is not decoration -- it converts a one-way deviation from the
    midpoint into the round-trip quoted-spread equivalent, so the number is
    directly comparable to `edge_spread`.
    """
    from .microstructure import parse_ts
    import bisect

    if not trades or not quotes:
        return None
    q_ts = [parse_ts(q["t"]) for q in quotes]
    values: list[float] = []
    for t in trades:
        i = bisect.bisect_right(q_ts, parse_ts(t["t"])) - 1
        if i < 0:
            continue
        q = quotes[i]
        bid, ask = q.get("bp") or 0.0, q.get("ap") or 0.0
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2.0
        if mid > 0:
            values.append(2.0 * abs(t["p"] - mid) / mid)
    if not values:
        return None
    # Median, not mean: a single print through a stale quote produces an
    # enormous outlier, and on thin names that is common rather than rare.
    return statistics.median(values)


def amihud(bars: Sequence[Bar], window: int = SPREAD_WINDOW) -> float | None:
    """Amihud (2002) illiquidity: mean |return| per dollar of volume.

    Reported alongside the spread because it measures a different thing --
    price impact per dollar traded, rather than the cost of crossing. It is the
    standard cost-per-dollar-volume proxy and is available for every symbol,
    which makes it a useful screen even where the spread estimate is unstable.

    Scaled by 1e6 so the numbers are readable (the raw ratio is ~1e-7).
    """
    w = bars[-window:] if window else bars
    values = []
    for prev, cur in zip(w, w[1:]):
        dollar = cur.close * cur.volume
        if prev.close > 0 and dollar > 0:
            values.append(abs(cur.close / prev.close - 1.0) / dollar)
    if not values:
        return None
    return statistics.fmean(values) * 1e6


def participation_rate(shares: float, bars: Sequence[Bar],
                       window: int = 21) -> float | None:
    """Order size as a fraction of typical daily volume."""
    w = bars[-window:] if window else bars
    volumes = [b.volume for b in w if b.volume > 0]
    if not volumes:
        return None
    typical = statistics.median(volumes)
    return shares / typical if typical > 0 else None


def daily_volatility(bars: Sequence[Bar], window: int = SPREAD_WINDOW) -> float | None:
    w = bars[-window:] if window else bars
    rets = [
        cur.close / prev.close - 1.0
        for prev, cur in zip(w, w[1:])
        if prev.close > 0
    ]
    if len(rets) < 2:
        return None
    return statistics.stdev(rets)


def market_impact(participation: float, volatility: float,
                  exponent: float = IMPACT_EXPONENT,
                  coefficient: float = IMPACT_COEFFICIENT) -> float:
    """Square-root law: impact = coefficient * sigma * participation^exponent.

    One-way. A round trip pays it twice, and the exit is usually the worse of
    the two because it is not optional.
    """
    if participation <= 0 or volatility <= 0:
        return 0.0
    return coefficient * volatility * (participation ** exponent)


@dataclass(frozen=True)
class TradeCost:
    """Round-trip cost of one position, as a fraction of notional."""

    spread: float
    impact: float
    fees: float
    participation: float | None
    feasible: bool
    source: str          # "edge" or "quotes"
    notes: str = ""
    #: True when the spread came from EDGE in the regime where it reads high.
    #: The cost is then an upper bound, which is the safe direction for a cost
    #: model but is not a measurement and must not be reported as one.
    spread_is_upper_bound: bool = False

    @property
    def total(self) -> float:
        return self.spread + self.impact + self.fees

    @property
    def total_bps(self) -> float:
        return self.total * 10_000


@dataclass(frozen=True)
class CostModel:
    """Turns bars and an order size into a round-trip cost.

    Defaults are deliberately pessimistic. An optimistic cost model is worse
    than none: it produces a number that looks rigorous while still flattering
    the strategy, and it is far harder to argue with than an absent one.
    """

    #: Minimum spread charged regardless of estimate. EDGE truncates negative
    #: squared spreads to zero, and a zero-cost fill does not exist on any
    #: venue. One cent on a $5 stock is 20bp; this floor is deliberately modest.
    floor_bps: float = 5.0
    impact_exponent: float = IMPACT_EXPONENT
    impact_coefficient: float = IMPACT_COEFFICIENT
    spread_window: int = SPREAD_WINDOW
    max_participation: float = MAX_PARTICIPATION
    #: Whether the strategy crosses the spread on both legs. True is the honest
    #: default: a limit order that does not fill is not a saving, it is a missed
    #: trade, and a backtest that assumes passive fills is assuming away the
    #: adverse selection that makes them fill.
    crosses_spread: bool = True

    def estimate(self, bars: Sequence[Bar], shares: float = 0.0,
                 measured_spread: float | None = None) -> TradeCost:
        """Round-trip cost for a position of `shares`.

        `measured_spread` overrides the estimate where real quotes exist -- the
        estimator is a fallback for symbols we lack intraday data on, not a
        preference.
        """
        notes = []
        if measured_spread is not None:
            spread_est, source = measured_spread, "quotes"
        else:
            spread_est, source = edge_spread(bars, self.spread_window), "edge"
            if spread_est is None:
                spread_est = 0.0
                notes.append("no spread estimate; floor applied")

        spread_est = max(spread_est, self.floor_bps / 10_000)
        upper_bound = (
            source == "edge" and spread_est * 10_000 < EDGE_RELIABLE_FLOOR_BPS
        )
        if upper_bound:
            notes.append(
                f"EDGE below {EDGE_RELIABLE_FLOOR_BPS:.0f}bps reads high; "
                "treat as an upper bound, not a measurement"
            )
        # Crossing on entry and exit costs the full spread once (half each way).
        spread_cost = spread_est if self.crosses_spread else spread_est / 2

        part = participation_rate(shares, bars) if shares else 0.0
        vol = daily_volatility(bars, self.spread_window) or 0.0
        impact = 0.0
        feasible = True
        if part:
            # Twice: entry and exit both move the market.
            impact = 2 * market_impact(part, vol, self.impact_exponent,
                                       self.impact_coefficient)
            if part > self.max_participation:
                feasible = False
                notes.append(
                    f"participation {part:.1%} exceeds {self.max_participation:.0%}; "
                    "not executable in one session"
                )

        fees = SELL_FEE_RATE  # sells only, so once per round trip
        return TradeCost(
            spread=spread_cost, impact=impact, fees=fees,
            participation=part, feasible=feasible, source=source,
            notes="; ".join(notes), spread_is_upper_bound=upper_bound,
        )


def compare_spread_estimates(bars: Sequence[Bar], trades: Sequence[dict],
                             quotes: Sequence[dict]) -> dict[str, float | None]:
    """Measure how far the EDGE estimate drifts from the realised spread.

    Run this before trusting a cost model built on estimates. The same check on
    order flow found a cheap classifier disagreeing with the exact one on sign
    three times in four -- cheap estimators are not automatically approximate
    versions of expensive ones, and the only way to know is to measure.
    """
    est = edge_spread(bars)
    real = effective_spread(trades, quotes)
    out: dict[str, float | None] = {
        "edge": est, "effective": real,
        "edge_bps": est * 10_000 if est is not None else None,
        "effective_bps": real * 10_000 if real is not None else None,
    }
    if est is not None and real is not None and real > 0:
        out["ratio"] = est / real
        out["abs_error_bps"] = abs(est - real) * 10_000
    return out
