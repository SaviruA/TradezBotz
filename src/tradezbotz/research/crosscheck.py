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
