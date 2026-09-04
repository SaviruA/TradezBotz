"""Is the edge broad, or is it five names?

`opportunistic buy + liquid` at h=60 reported a mean of +11.61% and a
winsorised mean of +2.65%. Those two numbers describe very different
strategies, and the gap between them is the whole question: a broad right tail
is an edge, and a handful of multi-baggers is an anecdote with a t-statistic
attached.

**Why the existing tests do not settle it.** Two-way clustering handles the
dependence between overlapping events on one symbol; it says nothing about
skewness. And at a 60-session horizon skewness is not a nuisance, it is the
dominant feature: Barber & Lyon (1997) and Lyon, Barber & Tsai (1999)
established that long-horizon buy-and-hold abnormal returns are strongly
positively skewed, that this makes the conventional t-statistic NEGATIVELY
biased, and that the resulting tests lose power in exactly the upper tail we
are testing. Their prescribed remedy is a skewness-adjusted bootstrapped t.

That bias runs in our favour, and saying so is the point: the honest reading is
not "our t is too low so the row is better than it looks", but "the standard
test is misspecified here in both directions, so use the one built for it".
LBT are blunt that misspecification "in nonrandom samples is pervasive", and an
insider-buy sample is about as nonrandom as they come.

So this module answers two separate questions:

  concentration -- how much of the total return comes from the few best trades,
                   which no significance test measures at all
  inference     -- Johnson's skewness-adjusted t, bootstrapped, which is the
                   published remedy for the horizon we are actually trading
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence

#: Trades whose contribution is reported individually.
TOP_N = (1, 5, 10)

#: Resamples for the bootstrap. 1,000 is the figure in Lyon, Barber & Tsai.
BOOTSTRAP_RESAMPLES = 1_000

#: Share of total return coming from the top 5 trades above which an edge is
#: called concentrated. Not a significance threshold -- a description. Five
#: trades carrying half the return is a fact a reader must be told, whatever
#: the t-statistic says.
CONCENTRATED_ABOVE = 0.50


@dataclass(frozen=True)
class Concentration:
    n: int
    mean: float
    median: float
    trimmed_mean: float
    #: share of the SUM of returns contributed by the top k trades
    top_share: dict[int, float]
    positive_share: float
    skew_adjusted_t: float
    bootstrap_ci: tuple[float, float]

    @property
    def concentrated(self) -> bool:
        return self.top_share.get(5, 0.0) >= CONCENTRATED_ABOVE

    def describe(self) -> str:
        tops = "  ".join(f"top {k}: {v:.1%}"
                         for k, v in sorted(self.top_share.items()))
        lines = [
            f"    n={self.n:,}  mean {self.mean:+.2%}  median "
            f"{self.median:+.2%}  10% trimmed {self.trimmed_mean:+.2%}  "
            f"{self.positive_share:.1%} positive",
            f"    share of total return from the best trades -- {tops}",
            f"    skewness-adjusted bootstrap t {self.skew_adjusted_t:+.2f}, "
            f"95% CI [{self.bootstrap_ci[0]:+.2%}, {self.bootstrap_ci[1]:+.2%}]"
            f"  (Lyon/Barber/Tsai 1999)",
        ]
        if self.concentrated:
            lines.append(
                f"    CONCENTRATED: five trades carry "
                f"{self.top_share.get(5, 0):.0%} of the total return. A "
                f"significance test cannot see this, and a portfolio that "
                f"missed those five earns nothing like the reported mean.")
        return "\n".join(lines)


def _johnson_t(xs: Sequence[float]) -> float:
    """Johnson's (1978) skewness-adjusted t.

    The conventional t assumes a symmetric sampling distribution. Under the
    positive skew that characterises long-horizon returns it is biased
    downward, which is why Lyon, Barber & Tsai adopt this correction before
    bootstrapping it.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean = statistics.fmean(xs)
    sd = statistics.pstdev(xs)
    if sd == 0:
        return 0.0
    gamma = sum((x - mean) ** 3 for x in xs) / (n * sd ** 3)
    s = mean / sd
    return math.sqrt(n) * (s + gamma * s ** 2 / 3 + gamma / (6 * n))


def _trimmed_mean(xs: Sequence[float], proportion: float = 0.10) -> float:
    ordered = sorted(xs)
    k = int(len(ordered) * proportion)
    core = ordered[k:len(ordered) - k] or ordered
    return statistics.fmean(core)


def analyse(returns: Sequence[float], *,
            resamples: int = BOOTSTRAP_RESAMPLES,
            seed: int = 0) -> Concentration | None:
    """Decompose a return series. None when there is nothing to decompose."""
    xs = [r for r in returns if r is not None]
    if len(xs) < 10:
        return None

    total = sum(xs)
    ordered = sorted(xs, reverse=True)
    top_share: dict[int, float] = {}
    for k in TOP_N:
        if k <= len(ordered):
            # Share of the SUM, which is the quantity a portfolio actually
            # earns. Against a total at or below zero the ratio is undefined
            # rather than enormous, and is reported as such.
            top_share[k] = (sum(ordered[:k]) / total) if total > 0 else float("nan")

    rng = random.Random(seed)
    means = []
    n = len(xs)
    for _ in range(resamples):
        means.append(statistics.fmean(rng.choices(xs, k=n)))
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(int(0.975 * resamples), resamples - 1)]

    return Concentration(
        n=n,
        mean=statistics.fmean(xs),
        median=statistics.median(xs),
        trimmed_mean=_trimmed_mean(xs),
        top_share=top_share,
        positive_share=sum(1 for x in xs if x > 0) / n,
        skew_adjusted_t=_johnson_t(xs),
        bootstrap_ci=(lo, hi),
    )
