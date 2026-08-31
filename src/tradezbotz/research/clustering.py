"""Dependence between observations, and what it does to a t-statistic.

Every backtest here has flagged CLUSTERED, and the flag was right: 1,836 trades
across 204 symbols with one $1.84 micro-cap contributing 154 of them are not
1,836 independent draws. The t-statistic and the Deflated Sharpe Ratio both
assume independence, so both were inflated by an unknown amount.

**There are two distinct dependencies here, and they need different fixes.**

  same symbol, many events   154 filings on one name trace a single price path.
                             Within-cluster dependence. Fixed by clustering the
                             standard error on the symbol.

  same date, many symbols    Form 4 filings bunch after earnings season, so many
                             events share a calendar day and therefore share
                             that day's market move. Cross-sectional dependence.
                             Fixed by clustering on the date, or by the
                             Kolari-Pynnonen correction.

Only fixing the first would have left the second untouched, and the second is
the more violent of the two. Kolari & Pynnonen (*Review of Financial Studies*,
2010) show the standard error of the mean is understated by a factor of roughly

    sqrt(1 + (N - 1) * rho)

so an average pairwise correlation of only **0.02** across 100 events overstates
the t-statistic by **1.73x**. A result at t = 3.0 is really at t = 1.7, which is
the difference between a finding and a coincidence.

**Two-way clustering** (Cameron, Gelbach & Miller) handles both at once:

    V = V_symbol + V_date - V_white

subtracting the plain heteroskedasticity-robust term because the observations
sharing both a symbol and a date are otherwise counted twice. The Monte Carlo
guidance is that it behaves well with at least 25 of each, which this universe
comfortably has on symbols and generally has on dates.

**The modification that matters most is not the t-statistic.** The Deflated
Sharpe Ratio takes an observation count, and we were handing it the raw trade
count -- which asserts independence, the very thing that is false. Passing the
*effective* sample size instead corrects the DSR and the t-statistic together,
and it is the DSR that decides significance here.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence

#: Below this many clusters, cluster-robust standard errors are themselves
#: unreliable -- the asymptotics assume many clusters, and with few the
#: correction is badly biased downward. The econometric literature turns to the
#: wild cluster bootstrap here; we report the problem rather than pretending the
#: correction worked.
MIN_CLUSTERS = 25

#: Cap on the estimated average pairwise correlation. Sample correlation
#: estimated from few overlapping dates is extremely noisy, and an outlier near
#: 1.0 would collapse the effective sample size to almost nothing. Capping keeps
#: a noisy estimate from producing an absurd correction.
MAX_RHO = 0.5


@dataclass(frozen=True)
class ClusterDiagnostics:
    """What the dependence structure of a sample actually is."""

    n_obs: int
    n_symbol_clusters: int
    n_date_clusters: int
    largest_cluster_share: float
    #: Average pairwise correlation implied by same-date co-movement.
    rho: float
    #: Independence-equivalent observation count.
    n_effective: float
    #: Standard error of the mean, naive and corrected.
    se_naive: float
    se_clustered: float

    @property
    def inflation(self) -> float:
        """How much the naive standard error understates the truth."""
        if self.se_naive <= 0:
            return 1.0
        return self.se_clustered / self.se_naive

    @property
    def enough_clusters(self) -> bool:
        """Whether the cluster-robust correction is itself trustworthy."""
        return (self.n_symbol_clusters >= MIN_CLUSTERS
                and self.n_date_clusters >= MIN_CLUSTERS)

    def summary(self) -> str:
        note = "" if self.enough_clusters else (
            f"  WARNING: fewer than {MIN_CLUSTERS} clusters on one dimension; "
            "the cluster-robust correction is itself unreliable here.\n"
        )
        return (
            f"  {self.n_obs:,} observations over {self.n_symbol_clusters} symbols "
            f"and {self.n_date_clusters} dates\n"
            f"  largest symbol {self.largest_cluster_share:.0%} of trades  "
            f"rho {self.rho:.4f}\n"
            f"  effective n {self.n_effective:,.0f} "
            f"({self.n_effective / self.n_obs:.0%} of nominal)  "
            f"SE inflation {self.inflation:.2f}x\n" + note
        )


def average_pairwise_correlation(values: Sequence[float],
                                 dates: Sequence[Hashable]) -> float:
    """Estimate average pairwise correlation from same-date co-movement.

    Events sharing a calendar day share that day's market move, so the variance
    of same-day group means carries the information. Using the standard
    intraclass correlation identity: if observations within a group have common
    correlation rho, the variance of a group mean of size m is

        var * (1 + (m - 1) * rho) / m

    so rho can be recovered by comparing observed between-group variance against
    what independence would predict.

    Returns 0.0 when there is nothing to estimate from -- no groups with more
    than one member, or no variance. Zero is the neutral value: it makes every
    correction below a no-op rather than silently inventing dependence.
    """
    groups: dict[Hashable, list[float]] = defaultdict(list)
    for v, d in zip(values, dates):
        groups[d].append(v)

    multi = [g for g in groups.values() if len(g) > 1]
    if not multi or len(values) < 3:
        return 0.0

    overall_var = statistics.pvariance(values)
    if overall_var <= 0:
        return 0.0

    # Weighted average of the within-group correlation implied by each group.
    weighted, weight_total = 0.0, 0.0
    for g in multi:
        m = len(g)
        mean = statistics.fmean(g)
        # Between-group deviation squared, versus the independence expectation.
        within = statistics.pvariance(g)
        # rho from the ratio of within-group variance to overall: identical
        # observations give within == 0 and rho == 1; independent draws give
        # within == overall and rho == 0.
        implied = 1.0 - (within / overall_var)
        weighted += implied * m
        weight_total += m

    if weight_total <= 0:
        return 0.0
    rho = weighted / weight_total
    # Negative correlation is possible in sample but not meaningful as a
    # dependence inflation; clamp to the usable range.
    return max(0.0, min(rho, MAX_RHO))


def kolari_pynnonen_factor(n: int, rho: float) -> float:
    """The KP deflation factor for a test statistic.

    sqrt((1 - rho) / (1 + (n - 1) * rho)), from Kolari & Pynnonen (RFS 2010).
    Multiply a naive t-statistic by this to correct it for cross-sectional
    correlation. Equals 1.0 when rho is zero, so applying it unconditionally is
    safe.
    """
    if n <= 1 or rho <= 0:
        return 1.0
    denom = 1.0 + (n - 1) * rho
    if denom <= 0:
        return 1.0
    return math.sqrt(max(0.0, (1.0 - rho)) / denom)


def effective_sample_size(n: int, rho: float, cluster_size: float) -> float:
    """Independence-equivalent observation count, via the Kish design effect.

        n_eff = n / (1 + (m - 1) * rho)

    where **m is the average cluster size, not the sample size**. That
    distinction is the whole correctness of this function. Using `n` in place of
    `m` treats every observation as correlated with every other, which is not
    what clustering means -- dependence here is block-diagonal, strong inside a
    symbol or a date and zero across them. The first version of this function
    made that mistake and reported an effective n of 3 out of 1,000, which would
    have made the Deflated Sharpe Ratio reject everything forever.

    Kolari-Pynnonen's own (N-1) term is not a counterexample: their N is the
    number of firms sharing a single event date, which *is* one cluster.
    """
    if n <= 1 or rho <= 0 or cluster_size <= 1:
        return float(max(n, 0))
    design_effect = 1.0 + (cluster_size - 1) * rho
    return n / design_effect if design_effect > 0 else float(n)


def effective_from_inflation(n: int, inflation: float) -> float:
    """Effective sample size implied by a measured standard-error inflation.

    Preferred over the design-effect formula wherever a cluster-robust standard
    error has actually been computed. The design effect assumes one uniform rho
    across a single clustering dimension; a measured inflation reflects whatever
    the real dependence structure is, including both dimensions at once and any
    imbalance in cluster sizes.

    Since SE scales as 1/sqrt(n), an inflation factor k means the sample carries
    the information of n / k^2 independent observations.
    """
    if n <= 1 or inflation <= 1:
        return float(max(n, 0))
    return n / (inflation ** 2)


def cluster_robust_variance(values: Sequence[float],
                            groups: Sequence[Hashable]) -> float:
    """Cluster-robust variance of the sample mean.

    The standard sandwich for a mean: sum the squared *cluster totals* of
    demeaned values rather than the squared individual deviations. Observations
    inside a cluster are allowed to be arbitrarily correlated, which is exactly
    the assumption we need for repeat filings on one symbol.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = statistics.fmean(values)
    totals: dict[Hashable, float] = defaultdict(float)
    for v, g in zip(values, groups):
        totals[g] += v - mean
    g_count = len(totals)
    if g_count < 2:
        # One cluster carries no information about between-cluster variation.
        return float("inf")
    # Finite-sample correction, as in standard cluster-robust practice.
    correction = g_count / (g_count - 1)
    return correction * sum(t * t for t in totals.values()) / (n * n)


def two_way_cluster_robust_variance(values: Sequence[float],
                                    groups_a: Sequence[Hashable],
                                    groups_b: Sequence[Hashable]) -> float:
    """Cameron-Gelbach-Miller two-way cluster-robust variance of the mean.

    V = V_a + V_b - V_intersection. The subtraction is not optional: without it
    observations sharing both a symbol and a date are counted twice, which
    overstates the correction rather than understating it.

    The estimator is not guaranteed positive in finite samples. When it comes
    out negative the conventional fallback is the larger one-way estimate, which
    is the conservative choice.
    """
    v_a = cluster_robust_variance(values, groups_a)
    v_b = cluster_robust_variance(values, groups_b)
    pairs = [(a, b) for a, b in zip(groups_a, groups_b)]
    v_ab = cluster_robust_variance(values, pairs)
    if any(math.isinf(v) for v in (v_a, v_b, v_ab)):
        return max(v for v in (v_a, v_b) if not math.isinf(v)) if not (
            math.isinf(v_a) and math.isinf(v_b)) else float("inf")
    combined = v_a + v_b - v_ab
    if combined <= 0 or math.isnan(combined):
        return max(v_a, v_b)
    return combined


def diagnose(values: Sequence[float], symbols: Sequence[Hashable],
             dates: Sequence[Hashable]) -> ClusterDiagnostics:
    """Full dependence diagnosis for one set of trade returns."""
    n = len(values)
    if n < 2:
        return ClusterDiagnostics(n, 0, 0, 0.0, 0.0, float(n), 0.0, 0.0)

    symbol_counts: dict[Hashable, int] = defaultdict(int)
    for s in symbols:
        symbol_counts[s] += 1
    largest = max(symbol_counts.values()) / n if symbol_counts else 0.0

    rho = average_pairwise_correlation(values, dates)
    se_naive = statistics.stdev(values) / math.sqrt(n)
    var_clustered = two_way_cluster_robust_variance(values, symbols, dates)
    se_clustered = math.sqrt(var_clustered) if var_clustered > 0 else se_naive
    # Never report a corrected SE tighter than the naive one: dependence inflates
    # uncertainty, and a sample that happens to produce a smaller clustered
    # estimate has not thereby become more informative than an independent one.
    se_clustered = max(se_clustered, se_naive)

    return ClusterDiagnostics(
        n_obs=n,
        n_symbol_clusters=len(symbol_counts),
        n_date_clusters=len(set(dates)),
        largest_cluster_share=largest,
        rho=rho,
        # From the measured inflation rather than the design-effect formula:
        # it reflects the actual two-dimensional structure and any imbalance in
        # cluster sizes, instead of assuming one uniform rho.
        n_effective=effective_from_inflation(n, se_clustered / se_naive
                                             if se_naive > 0 else 1.0),
        se_naive=se_naive,
        se_clustered=se_clustered,
    )
