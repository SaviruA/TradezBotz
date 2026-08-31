"""Comparing two price sources to find bars we should not trust.

Free market data has no error bars. A single source will hand you a close price
for an illiquid small cap with exactly the same confidence it hands you one for
Apple, and nothing in the response distinguishes them.

Two independent sources do carry that information. Where they agree, the bar is
probably fine. Where they diverge materially, the name is almost certainly too
thinly traded for either to be reliable -- IEX-only feeds are documented to show
last trades well away from where a stock actually printed elsewhere.

This matters here more than it would elsewhere: insider purchases skew small-cap,
so the population we most want to measure is the population where free data is
weakest. Disagreement does not tell us which source is right. It tells us the
event is low-confidence, which is enough to report honestly and to exclude in a
sensitivity check.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum

from .prices import Series

#: Median relative close difference below which two sources effectively agree.
#: Tick rounding and minor adjustment differences live under this.
AGREE_THRESHOLD = 0.005  # 0.5%

#: Above this the bar is treated as untrustworthy. Set well under the ~15%
#: ghost-price errors reported for IEX on illiquid names, so real problems are
#: caught long before they reach that scale.
DISAGREE_THRESHOLD = 0.02  # 2%

#: Fewer overlapping sessions than this and the comparison says nothing.
MIN_OVERLAP_DAYS = 5


class Agreement(str, Enum):
    AGREE = "agree"
    SUSPECT = "suspect"
    DISAGREE = "disagree"
    INSUFFICIENT_OVERLAP = "insufficient_overlap"


@dataclass(frozen=True)
class Disagreement:
    symbol: str
    overlapping_days: int
    median_rel_diff: float | None
    max_rel_diff: float | None
    verdict: Agreement
    #: Highs and lows are compared separately because different indicators read
    #: different fields. Bollinger and RSI use closes; ATR, Donchian and every
    #: stop-loss read highs and lows. A source can agree on closes and still
    #: carry a phantom high that manufactures a breakout or trips a stop --
    #: which the Concretum provider comparison names as a top failure mode.
    median_high_diff: float | None = None
    median_low_diff: float | None = None

    @property
    def range_trustworthy(self) -> bool:
        """Whether highs and lows agree well enough for ATR, Donchian or stops.

        Deliberately separate from `trustworthy`: an indicator using only closes
        may be usable on a symbol whose highs disagree, and refusing both would
        discard usable data.
        """
        if self.median_high_diff is None or self.median_low_diff is None:
            return False
        return max(self.median_high_diff, self.median_low_diff) <= AGREE_THRESHOLD

    @property
    def trustworthy(self) -> bool:
        """Whether the bars are safe to build a return on.

        INSUFFICIENT_OVERLAP is deliberately not trustworthy: an unverified bar
        is not the same as a verified-good one, and quietly treating it as such
        is how unreliable data re-enters a study that meant to exclude it.
        """
        return self.verdict is Agreement.AGREE


def compare(primary: Series, secondary: Series) -> Disagreement:
    """Compare closes on the sessions both sources cover.

    Only overlapping days are considered. A missing day in one source is a
    coverage gap, not a price disagreement, and conflating them would blame the
    wrong thing -- `labeler.coverage_report` already tracks gaps.
    """
    b = {bar.day: bar for bar in secondary.bars}
    diffs, high_diffs, low_diffs = [], [], []
    for bar in primary.bars:
        other = b.get(bar.day)
        if other is None:
            continue
        if bar.close > 0:
            diffs.append(abs(bar.close - other.close) / bar.close)
        if bar.high > 0:
            high_diffs.append(abs(bar.high - other.high) / bar.high)
        if bar.low > 0:
            low_diffs.append(abs(bar.low - other.low) / bar.low)

    if len(diffs) < MIN_OVERLAP_DAYS:
        return Disagreement(
            symbol=primary.symbol, overlapping_days=len(diffs),
            median_rel_diff=None, max_rel_diff=None,
            verdict=Agreement.INSUFFICIENT_OVERLAP,
        )

    median = statistics.median(diffs)
    if median <= AGREE_THRESHOLD:
        verdict = Agreement.AGREE
    elif median <= DISAGREE_THRESHOLD:
        verdict = Agreement.SUSPECT
    else:
        verdict = Agreement.DISAGREE

    return Disagreement(
        symbol=primary.symbol,
        overlapping_days=len(diffs),
        median_rel_diff=median,
        max_rel_diff=max(diffs),
        verdict=verdict,
        median_high_diff=statistics.median(high_diffs) if high_diffs else None,
        median_low_diff=statistics.median(low_diffs) if low_diffs else None,
    )


def summarise(results: list[Disagreement]) -> dict[str, float | int]:
    """Aggregate verdicts, for reporting beside any backtest result.

    A strategy measured on symbols where two sources disagree is a strategy
    measured on prices that may not have existed.
    """
    if not results:
        return {"total": 0}
    counts = {v: sum(1 for r in results if r.verdict is v) for v in Agreement}
    return {
        "total": len(results),
        "agree": counts[Agreement.AGREE],
        "suspect": counts[Agreement.SUSPECT],
        "disagree": counts[Agreement.DISAGREE],
        "insufficient_overlap": counts[Agreement.INSUFFICIENT_OVERLAP],
        "trustworthy_rate": counts[Agreement.AGREE] / len(results),
        # Tracked separately: a symbol can agree on closes while its highs
        # disagree, which is safe for Bollinger and RSI but not for ATR,
        # Donchian or any stop-loss.
        "range_trustworthy": sum(1 for r in results if r.range_trustworthy),
        "range_trustworthy_rate":
            sum(1 for r in results if r.range_trustworthy) / len(results),
    }


# --- three-way adjudication ---------------------------------------------------
#
# Two sources can only ever say "one of you is wrong." Measured on 203 symbols,
# 45 disagreed materially and we spent weeks unable to say which vendor to
# believe -- so Alpaca's deeper history stayed unusable on principle rather than
# on evidence. A third independent source converts that standoff into a verdict.
#
# It is a referee, not a system of record. Yahoo is derived and unaudited; the
# only property being used is independence from the other two.

class Verdict(str, Enum):
    ALL_AGREE = "all_agree"
    #: Two agree and the third is the outlier -- the useful case.
    PRIMARY_OUTLIER = "primary_outlier"
    SECONDARY_OUTLIER = "secondary_outlier"
    REFEREE_OUTLIER = "referee_outlier"
    #: Not an error at all: the sources are quoting different things. One is
    #: total-return (dividend) adjusted and the others are price-only, so the
    #: series diverge by the accumulated distribution and reconverge at the
    #: present. Both are correct; they answer different questions.
    ADJUSTMENT_BASIS = "adjustment_basis"
    #: All three disagree with each other. No majority, so no verdict.
    NO_MAJORITY = "no_majority"
    INSUFFICIENT_OVERLAP = "insufficient_overlap"


#: A dividend-adjusted series is *below* a price-only one everywhere in the past
#: and converges to it at the present. Requiring both properties separates this
#: from a genuine error: a bad split factor is a roughly constant ratio that does
#: not converge, and can point either way.
CONVERGENCE_RATIO = 0.25


@dataclass(frozen=True)
class Adjudication:
    symbol: str
    primary_secondary: float | None
    primary_referee: float | None
    secondary_referee: float | None
    verdict: Verdict
    overlapping_days: int

    @property
    def trustworthy_source(self) -> str | None:
        """Which of the two working sources to believe for this symbol.

        None when there is no verdict. Deliberately per-symbol: measured against
        Yahoo, Massive was right on XELB (7.03 against Alpaca's 21.11) and wrong
        on BDX (240.97 against Alpaca's 181.20 and Yahoo's 189.44). Neither
        vendor is uniformly correct, so a global "trust X" rule would be wrong
        roughly half the time.
        """
        if self.verdict is Verdict.ALL_AGREE:
            return "both"
        if self.verdict is Verdict.PRIMARY_OUTLIER:
            return "secondary"
        if self.verdict is Verdict.SECONDARY_OUTLIER:
            return "primary"
        if self.verdict is Verdict.ADJUSTMENT_BASIS:
            # Both are right. Which to use is a modelling choice, not a data
            # quality one, so this deliberately does not name a winner.
            return "both"
        return None


def _median_diff(a: Series, b: Series) -> tuple[float | None, int]:
    other = {bar.day: bar for bar in b.bars}
    diffs = [
        abs(bar.close - other[bar.day].close) / bar.close
        for bar in a.bars
        if bar.day in other and bar.close > 0
    ]
    if len(diffs) < MIN_OVERLAP_DAYS:
        return None, len(diffs)
    return statistics.median(diffs), len(diffs)


def _signed_diffs(a: Series, b: Series) -> list[tuple[object, float]]:
    """Per-day (day, (b - a)/a), oldest first. Sign is the whole point here."""
    other = {bar.day: bar for bar in b.bars}
    return [
        (bar.day, (other[bar.day].close - bar.close) / bar.close)
        for bar in a.bars
        if bar.day in other and bar.close > 0
    ]


def _looks_like_dividend_adjustment(price_only: Series, candidate: Series) -> bool:
    """Whether `candidate` is the same series on a total-return basis.

    Two signatures must both hold, and requiring both is what separates this
    from a genuine error:

    1. **One-sided.** A dividend-adjusted history is *below* the price-only one
       on essentially every past day, because past prices are marked down by the
       distributions paid since. A bad split factor can point either way.
    2. **Convergent.** The gap is the sum of dividends *since that day*, so it
       shrinks to nothing at the present. A wrong split factor is a roughly
       constant ratio that never converges -- which is exactly the XELB case
       (a clean 3.004 throughout), and that one is a real error.
    """
    signed = _signed_diffs(price_only, candidate)
    if len(signed) < MIN_OVERLAP_DAYS * 4:
        return False
    values = [v for _, v in signed]
    below = sum(1 for v in values if v < 0) / len(values)
    if below < 0.9:
        return False

    gaps = [abs(v) for v in values]
    if statistics.median(gaps) <= AGREE_THRESHOLD:
        return False

    # Rank correlation of the gap against time. Dividends accumulate backwards
    # from today, so the gap declines monotonically and this sits near -1; a
    # wrong split factor is a constant ratio and sits near 0.
    #
    # Rank rather than a two-window ratio because the gap is a fraction of
    # *price*, and on a name whose price itself collapsed the raw ratio is
    # dominated by the price path. ARI pays ~98% of its price in distributions
    # over the window and also fell hard: the ratio test called it a vendor
    # fault, the rank test correctly reads the monotone decline.
    return _rank_correlation(list(range(len(gaps))), gaps) <= -0.7


def _rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Retained as a thin alias; the implementation moved to `clustering`."""
    from .clustering import rank_correlation
    return rank_correlation(xs, ys)



def adjudicate(primary: Series, secondary: Series, referee: Series) -> Adjudication:
    """Decide which of two disagreeing sources the third one backs.

    The rule is majority, not proximity: two sources agreeing within
    AGREE_THRESHOLD outvote a third that differs from both. Proximity would let
    a source that is merely *less* wrong win, which is not the same claim.
    """
    ps, n_ps = _median_diff(primary, secondary)
    pr, n_pr = _median_diff(primary, referee)
    sr, n_sr = _median_diff(secondary, referee)
    overlap = min(n_ps, n_pr, n_sr)

    if ps is None or pr is None or sr is None:
        return Adjudication(primary.symbol, ps, pr, sr,
                            Verdict.INSUFFICIENT_OVERLAP, overlap)

    close = AGREE_THRESHOLD
    if ps <= close and pr <= close and sr <= close:
        verdict = Verdict.ALL_AGREE
    elif (pr <= close < ps
          and _looks_like_dividend_adjustment(primary, secondary)):
        # Primary and referee agree, and the secondary's divergence has the
        # total-return signature. Checked BEFORE calling it an outlier: measured
        # on 20 disputed symbols this was every single one of them, and calling
        # a legitimately dividend-adjusted series "wrong" would have retired a
        # perfectly good source over a definitional difference.
        verdict = Verdict.ADJUSTMENT_BASIS
    elif sr <= close < ps and pr > close:
        # secondary and referee agree; primary is the odd one out
        verdict = Verdict.PRIMARY_OUTLIER
    elif pr <= close < ps and sr > close:
        verdict = Verdict.SECONDARY_OUTLIER
    elif ps <= close < pr and sr > close:
        verdict = Verdict.REFEREE_OUTLIER
    else:
        verdict = Verdict.NO_MAJORITY
    return Adjudication(primary.symbol, ps, pr, sr, verdict, overlap)


def summarise_adjudications(results: list[Adjudication]) -> dict[str, float | int]:
    """Aggregate verdicts, and report how often each vendor was the outlier.

    The headline number is `resolved`: how many previously-deadlocked symbols the
    referee actually settled. Everything else is diagnostics.
    """
    if not results:
        return {"total": 0}
    counts = {v: sum(1 for r in results if r.verdict is v) for v in Verdict}
    resolved = counts[Verdict.PRIMARY_OUTLIER] + counts[Verdict.SECONDARY_OUTLIER]
    benign = counts[Verdict.ALL_AGREE] + counts[Verdict.ADJUSTMENT_BASIS]
    return {
        "total": len(results),
        "all_agree": counts[Verdict.ALL_AGREE],
        "adjustment_basis": counts[Verdict.ADJUSTMENT_BASIS],
        "primary_outlier": counts[Verdict.PRIMARY_OUTLIER],
        "secondary_outlier": counts[Verdict.SECONDARY_OUTLIER],
        "referee_outlier": counts[Verdict.REFEREE_OUTLIER],
        "no_majority": counts[Verdict.NO_MAJORITY],
        "insufficient_overlap": counts[Verdict.INSUFFICIENT_OVERLAP],
        "resolved": resolved,
        "usable": benign + resolved,
    }
