"""Event-study backtesting, wired to the trial registry.

Turns labelled events into an answer, with the statistics that make the answer
mean something.

**Selectors compose, deliberately.** A signal with no standalone edge can still
carry information in combination -- insider buying *conditioned on* elevated
sentiment is a different hypothesis from either alone. Dropping a weak signal
destroys that combination before it can be measured, so nothing is filtered out
in advance; `all_of`, `any_of` and `threshold` exist to make combinations as easy
to test as singles.

**Every run registers a trial, including the ones that look bad.** Testing
everything raises the trial count and therefore the Deflated Sharpe bar. That is
the honest cost. The dishonest alternative is testing everything and reporting
only the winners, which is what actually corrupts a result.

**Why the Sharpe here is per-trade, not per-day.** The labeller stores the return
at each horizon, not the daily path between entry and exit, so a genuine daily
portfolio series cannot be reconstructed without refetching every intermediate
bar. Treating each trade as one observation is the standard event-study approach
and it feeds the DSR coherently: n_obs is the trade count, and the question
becomes "is this trade-level edge distinguishable from selection noise across N
trials?" -- which is exactly what we want to know. `sharpe_annualised` is derived
from observed trade frequency for human intuition only; never feed it to the DSR.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .concentration import analyse as analyse_concentration
from .labeler import Coverage, Label
from .trials import TrialRegistry, assess

#: An event is (payload, label). Returns True to take the trade.
Selector = Callable[[dict, Label], bool]

TRADING_DAYS = 252


# --- selector combinators ----------------------------------------------------

def all_of(*selectors: Selector) -> Selector:
    """Conjunction: every condition must hold. The usual way to test whether
    two weak signals are stronger together."""
    def sel(payload: dict, label: Label) -> bool:
        return all(s(payload, label) for s in selectors)
    return sel


def any_of(*selectors: Selector) -> Selector:
    """Disjunction: widens the population rather than narrowing it."""
    def sel(payload: dict, label: Label) -> bool:
        return any(s(payload, label) for s in selectors)
    return sel


def negate(selector: Selector) -> Selector:
    """The complement -- useful as a control group. If a signal works, its
    inverse should not."""
    def sel(payload: dict, label: Label) -> bool:
        return not selector(payload, label)
    return sel


def field_equals(key: str, value: object) -> Selector:
    def sel(payload: dict, label: Label) -> bool:
        return payload.get(key) == value
    return sel


def threshold(key: str, minimum: float) -> Selector:
    def sel(payload: dict, label: Label) -> bool:
        v = payload.get(key)
        return isinstance(v, (int, float)) and v >= minimum
    return sel


def everything(payload: dict, label: Label) -> bool:
    """Baseline: trade every labelled event. Any real signal must beat this."""
    return True


# --- results -----------------------------------------------------------------

@dataclass(frozen=True)
class BacktestResult:
    hypothesis: str
    horizon: int
    trial_id: int
    n_events: int
    n_trades: int
    mean_return: float
    median_return: float
    stdev: float
    hit_rate: float
    sharpe_per_trade: float
    sharpe_annualised: float
    t_stat: float
    skew: float
    kurtosis: float
    deflated_sharpe: float
    n_trials: int
    significant: bool
    n_symbols: int = 0
    top_symbol_share: float = 0.0
    mean_return_winsorised: float = 0.0
    #: Whether the edge is broad or is a handful of names, plus the
    #: skewness-adjusted inference that long horizons require. None when
    #: there were too few trades to decompose.
    concentration: object | None = None
    #: Mean return after round-trip transaction costs. Zero when no cost model
    #: was supplied, in which case `costed` is False and the gross figure is the
    #: only one that exists.
    mean_return_net: float = 0.0
    #: Median round-trip cost actually charged, in basis points.
    median_cost_bps: float = 0.0
    costed: bool = False
    #: t-statistic with a two-way cluster-robust standard error (symbol x date).
    #: This is the one to read. `t_stat` above assumes independent draws, which
    #: on this data is false: on simulated noise with our dependence structure
    #: the naive statistic rejected a true null 57% of the time at alpha = 5%.
    t_stat_clustered: float = 0.0
    #: Independence-equivalent observation count, derived from the measured
    #: standard-error inflation. This is what the DSR is given.
    n_effective: float = 0.0
    #: Average pairwise correlation implied by same-date co-movement.
    rho: float = 0.0
    #: How much the naive standard error understates the clustered one.
    se_inflation: float = 1.0
    #: False when either clustering dimension has too few groups for the
    #: correction itself to be trustworthy.
    clusters_sufficient: bool = True
    coverage: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def outlier_dependent(self) -> bool:
        """Whether the result leans on a few extreme observations.

        If capping returns at ±15% halves the mean, the edge lives in the tail
        rather than the population -- and on this data a tail observation is as
        likely to be a phantom print as a real move.
        """
        if self.mean_return == 0:
            return False
        return abs(self.mean_return_winsorised / self.mean_return) < 0.5

    @property
    def survives_costs(self) -> bool:
        """Whether the edge is still positive once fills are paid for.

        The single most important property here, and the one that was missing
        while every result was computed as though fills were free. Small-cap
        spreads run five to ten times wider than large-cap ones, and this is a
        small-cap strategy: a gross edge smaller than the round trip is not a
        weak edge, it is a loss.

        False when no cost model was supplied -- an uncosted result has not
        demonstrated survival, and treating "unmeasured" as "passed" is exactly
        the failure this field exists to prevent.
        """
        return self.costed and self.mean_return_net > 0

    @property
    def cost_ratio(self) -> float:
        """Share of the gross edge consumed by costs. >1 means costs exceed it."""
        if not self.costed or self.mean_return == 0:
            return 0.0
        return 1.0 - (self.mean_return_net / self.mean_return)

    @property
    def clustered(self) -> bool:
        """Whether observations are concentrated on few symbols.

        Overlapping events on one symbol are not independent draws: 154 filings
        on a $1.84 micro-cap trace one price path, not 154 outcomes.

        This is now a *description*, not a disqualification. It used to mean
        "this result is unusable". Since the standard error is cluster-robust
        and the DSR receives the effective sample size, the dependence is
        corrected for rather than merely flagged -- a clustered result can be
        believed now, it simply needed a far larger raw edge to get here.
        """
        if self.n_trades < 30 or self.n_symbols == 0:
            return False
        return (self.n_trades / self.n_symbols) > 5 or self.top_symbol_share > 0.15

    @property
    def dependence_severe(self) -> bool:
        """Whether dependence has consumed most of the nominal sample.

        `n_effective == 0` is the extreme, not an absence of one: it means every
        observation fell in a single cluster, so there is no between-cluster
        variation to estimate a standard error from at all. Reading that as
        "not severe" would let the most degenerate possible sample through
        unflagged, which is the opposite of what this is for.
        """
        if self.n_trades < 30:
            return False
        if self.n_effective <= 0:
            return True
        return (self.n_effective / self.n_trades) < 0.25

    def summary(self) -> str:
        return (
            f"{self.hypothesis}  h={self.horizon}\n"
            f"  trades {self.n_trades:,} of {self.n_events:,} labelled events\n"
            f"  mean {self.mean_return:+.3%}  median {self.median_return:+.3%}  "
            f"hit {self.hit_rate:.1%}\n"
            f"  t-stat {self.t_stat:+.2f}  Sharpe/trade {self.sharpe_per_trade:+.3f}\n"
            f"  DSR {self.deflated_sharpe:.3f} across {self.n_trials} trials  "
            f"-> {'SIGNIFICANT' if self.significant else 'not significant'}"
            + (
                f"\n  net {self.mean_return_net:+.3%} after "
                f"{self.median_cost_bps:.0f}bps costs "
                f"({self.cost_ratio:.0%} of the edge)"
                f"{'' if self.survives_costs else '  -- DOES NOT SURVIVE COSTS'}"
                if self.costed else
                "\n  UNCOSTED: gross returns only. Fills are not free, and this "
                "is a small-cap strategy."
            )
            + (
                f"\n  clustered: {self.n_trades:,} trades over {self.n_symbols} "
                f"symbols, top symbol {self.top_symbol_share:.0%}"
                if self.clustered else ""
            )
            + (
                f"\n  dependence: rho {self.rho:.3f}, SE inflated "
                f"{self.se_inflation:.2f}x, effective n {self.n_effective:,.0f} "
                f"({self.n_effective / max(self.n_trades, 1):.0%} of nominal)"
                f"\n  t-stat clustered {self.t_stat_clustered:+.2f}  "
                f"(naive {self.t_stat:+.2f})"
                if self.n_effective else
                "\n  dependence: every observation fell in a single cluster; "
                "no standard error can be estimated and no evidence is carried."
                if self.n_trades >= 30 else ""
            )
            + (
                "\n  WARNING: too few clusters for the correction itself to be "
                "reliable. The econometric answer here is a wild cluster "
                "bootstrap, which is not implemented."
                if not self.clusters_sufficient and self.n_trades >= 30 else ""
            )
        )


#: Winsorisation bound for the sensitivity check. Welch ("Stock Return
#: Outliers") recommends ±10-15% for CRSP stocks, finding that winsorised
#: standard deviations and betas predict their own future realisations better
#: than unwinsorised ones, with no gain from deleting observations instead.
#:
#: We report winsorised *alongside* raw rather than replacing it. On the small
#: caps where insider buying concentrates, a 15% five-day move is frequently
#: genuine, so silently capping would discard real outcomes. A large gap
#: between the two means the result rests on a handful of extreme observations
#: -- which is a finding about fragility, not a number to quietly fix.
WINSOR_LIMIT = 0.15


def winsorise(xs: Sequence[float], limit: float = WINSOR_LIMIT) -> list[float]:
    """Cap values at ±limit, keeping every observation."""
    return [max(-limit, min(limit, x)) for x in xs]


def _moments(xs: Sequence[float]) -> tuple[float, float]:
    """Sample skew and kurtosis. The DSR needs both: fat tails and negative
    skew both make an extreme Sharpe easier to reach by luck."""
    n = len(xs)
    if n < 4:
        return 0.0, 3.0
    mean = statistics.fmean(xs)
    sd = statistics.pstdev(xs)
    if sd == 0:
        return 0.0, 3.0
    m3 = sum((x - mean) ** 3 for x in xs) / n
    m4 = sum((x - mean) ** 4 for x in xs) / n
    return m3 / sd**3, m4 / sd**4


def run(
    labels: Iterable[Label],
    payloads: Iterable[dict],
    *,
    hypothesis: str,
    rationale: str,
    registry: TrialRegistry,
    selector: Selector = everything,
    horizon: int = 5,
    partition: str = "train",
    trades_per_year: float | None = None,
    costs: "Callable[[Label], float] | None" = None,
    dataset: str = "",
) -> BacktestResult:
    """Measure one hypothesis over labelled events.

    The trial is registered *before* the result is known, so an experiment
    abandoned midway still counts against N.
    """
    # `dataset` makes the trial's identity include the data it ran against, so a
    # nightly re-run of the same sweep updates one trial rather than appending a
    # new one and inflating N. See TrialRegistry.register.
    trial_id = registry.register(
        hypothesis,
        rationale,
        params={"horizon": horizon, "partition": partition},
        split=partition,
        dataset=dataset,
    )

    pairs = list(zip(payloads, labels))
    coverage = {c.value: 0 for c in Coverage}
    returns: list[float] = []
    symbols: list[str] = []
    entry_days: list[object] = []
    cost_per_trade: list[float] = []
    for payload, label in pairs:
        coverage[label.coverage.value] += 1
        if horizon not in label.returns:
            continue
        if selector(payload, label):
            returns.append(label.returns[horizon])
            symbols.append(label.symbol)
            # Entry day, not event day: two filings entering on the same session
            # share that session's market move regardless of when they were
            # disclosed. That shared move is the cross-sectional dependence.
            entry_days.append(label.entry_day)
            if costs is not None:
                # Charged per trade rather than as one average, because cost
                # varies by an order of magnitude across this universe -- a
                # mega-cap round trip is ~5bps and a micro-cap one ~300bps.
                # A single average would flatter the illiquid names, which are
                # exactly the ones carrying the signal.
                cost_per_trade.append(costs(label))

    n = len(returns)
    if n < 2:
        registry.abandon(trial_id, f"only {n} qualifying trades")
        return BacktestResult(
            hypothesis=hypothesis, horizon=horizon, trial_id=trial_id,
            n_events=len(pairs), n_trades=n, mean_return=0.0, median_return=0.0,
            stdev=0.0, hit_rate=0.0, sharpe_per_trade=0.0, sharpe_annualised=0.0,
            t_stat=0.0, skew=0.0, kurtosis=3.0, deflated_sharpe=0.0,
            n_trials=registry.count(), significant=False, coverage=coverage,
            notes="insufficient trades to measure",
        )

    from collections import Counter
    counts = Counter(symbols)
    n_symbols = len(counts)
    top_share = (counts.most_common(1)[0][1] / len(returns)) if counts else 0.0

    from .clustering import diagnose

    # Dependence structure first: it changes both the t-statistic and the
    # observation count the DSR is entitled to assume.
    cluster = diagnose(returns, symbols, entry_days)

    mean = statistics.fmean(returns)
    mean_w = statistics.fmean(winsorise(returns))

    # Decomposed on the NET series when we have one, because the question is
    # about the return a portfolio would actually keep. Computed here because
    # the per-trade returns do not survive onto the result -- reconstructing
    # them later would mean re-running the whole labelling pass.
    _net_series = ([r - c for r, c in zip(returns, cost_per_trade)]
                   if cost_per_trade and len(cost_per_trade) == len(returns)
                   else list(returns))
    concentration = analyse_concentration(_net_series)
    costed = bool(cost_per_trade) and len(cost_per_trade) == len(returns)
    if costed:
        mean_net = statistics.fmean(
            r - c for r, c in zip(returns, cost_per_trade)
        )
        median_cost_bps = statistics.median(cost_per_trade) * 10_000
    else:
        mean_net, median_cost_bps = 0.0, 0.0
    sd = statistics.stdev(returns)
    skew, kurt = _moments(returns)
    sharpe_trade = mean / sd if sd > 0 else 0.0
    t_stat = sharpe_trade * math.sqrt(n)
    # The statistic to actually read. Dependence inflates the naive one; on
    # simulated noise carrying our structure it rejected a true null 57% of the
    # time against a nominal 5%, and the clustered version brought that to 6.7%.
    t_clustered = mean / cluster.se_clustered if cluster.se_clustered > 0 else 0.0

    # Annualisation is presentational only. Without a supplied trade frequency,
    # assume each position is held `horizon` sessions and capital recycles.
    per_year = trades_per_year or (TRADING_DAYS / max(horizon, 1))
    sharpe_annual = sharpe_trade * math.sqrt(per_year)

    # The DSR is handed the EFFECTIVE observation count, not the trade count.
    # It assumes independent draws, and passing a number that asserts far more
    # independence than the data has is the same error as the naive t-stat --
    # except the DSR is what decides significance here, so it matters more.
    verdict = assess(
        registry, observed_sharpe_annual=sharpe_annual,
        n_obs=max(int(cluster.n_effective), 2),
        skew=skew, kurtosis=kurt, periods_per_year=int(per_year) or 1,
    )

    registry.complete(
        trial_id, sharpe=sharpe_trade, n_obs=n, n_trades=n,
        skew=skew, kurtosis=kurt,
        notes=f"mean={mean:.5f} hit={sum(1 for r in returns if r > 0)/n:.3f}",
    )

    return BacktestResult(
        hypothesis=hypothesis, horizon=horizon, trial_id=trial_id,
        n_events=len(pairs), n_trades=n,
        mean_return=mean, median_return=statistics.median(returns), stdev=sd,
        hit_rate=sum(1 for r in returns if r > 0) / n,
        sharpe_per_trade=sharpe_trade, sharpe_annualised=sharpe_annual,
        t_stat=t_stat, skew=skew, kurtosis=kurt,
        deflated_sharpe=float(verdict["deflated_sharpe"]),
        n_trials=int(verdict["n_trials"]),
        significant=bool(verdict["significant"]),
        n_symbols=n_symbols, top_symbol_share=top_share,
        mean_return_winsorised=mean_w,
        t_stat_clustered=t_clustered, n_effective=cluster.n_effective,
        rho=cluster.rho, se_inflation=cluster.inflation,
        clusters_sufficient=cluster.enough_clusters,
        mean_return_net=mean_net, median_cost_bps=median_cost_bps, costed=costed,
        coverage=coverage, concentration=concentration,
    )


def compare(results: Sequence[BacktestResult]) -> str:
    """Tabulate several hypotheses side by side.

    Always show the baseline and the control alongside a candidate: a signal
    that does not beat "trade everything" has not demonstrated anything, and one
    whose negation performs equally well is measuring the population, not the
    signal.
    """
    if not results:
        return "no results"
    header = (
        f"{'hypothesis':<34}{'trades':>8}{'mean':>9}{'hit':>7}"
        f"{'t':>7}{'DSR':>7}{'syms':>6}{'wins':>9}  verdict"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.hypothesis[:33]:<34}{r.n_trades:>8,}{r.mean_return:>+9.3%}"
            f"{r.hit_rate:>7.1%}{r.t_stat:>+7.2f}{r.deflated_sharpe:>7.3f}"
            f"{r.n_symbols:>6}{r.mean_return_winsorised:>+9.3%}"
            f"  {'CLUSTERED' if r.clustered else ('SIGNIFICANT' if r.significant else '-')}"
            f"{'  OUTLIER-DEP' if r.outlier_dependent else ''}"
        )
    return "\n".join(lines)
