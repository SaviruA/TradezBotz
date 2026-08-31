"""Run the whole backlog, then weed out on the results.

The standing rule is that no strategy is dropped before it is tested -- not
because it looks unpromising, not because a paper found it weak, and not because
whoever is holding the keyboard has an opinion. Opinions belong in the
`prior` field of a `Candidate`, where they can be *checked against the outcome*
rather than quietly deciding it.

This module exists because that rule is easy to state and easy to erode. Skipping
a test is invisible: nothing in the output shows a strategy that was never run.
An explicit registry of candidates makes an omission a deletion someone has to
make on purpose.

**Every run costs trial budget, and that is the honest price.** A sweep of N
candidates across H horizons registers N x H trials, and the Deflated Sharpe bar
rises with every one. Testing everything therefore makes each individual finding
*harder* to establish, not easier. That is not a flaw in the method; it is what
honest search costs, and the alternative -- testing everything and reporting only
what looked good -- is the thing that corrupts a result.

**Controls are counted too.** A control is not a strategy anyone would deploy,
so there is a defensible argument that it does not compete for selection and
should not inflate N. We count it anyway. The conservative choice raises the bar
against ourselves, and being wrong in that direction costs a missed edge rather
than a false one.

**Weeding out happens here, after measurement.** `verdict` applies the gates in
the order they can disqualify: no edge, eaten by costs, indistinguishable from
its own control, dependent on a handful of outliers, or not significant once
trial count and dependence are accounted for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .backtest import BacktestResult, Selector, everything, negate, run
from .labeler import Label
from .trials import TrialRegistry

#: Horizons every candidate is measured at. Kept small deliberately: each extra
#: horizon multiplies the trial count, and four is enough to see whether an edge
#: decays, persists, or only exists at one arbitrary holding period.
DEFAULT_HORIZONS = (1, 5, 20)


@dataclass(frozen=True)
class Candidate:
    """One hypothesis queued for measurement."""

    name: str
    selector: Selector
    rationale: str
    #: What we expect, recorded BEFORE the run. This is the honest home for an
    #: opinion: written down where the result can contradict it, rather than
    #: acted on by quietly not running the test.
    prior: str = ""
    #: Skip only for a mechanical reason -- data that does not exist yet -- never
    #: because the idea looks weak. The reason is required and is printed in the
    #: report, so an omission is visible rather than silent.
    blocked_by: str = ""

    @property
    def runnable(self) -> bool:
        return not self.blocked_by


class Verdict:
    KEEP = "keep"
    NO_EDGE = "no edge"
    EATEN_BY_COSTS = "costs exceed edge"
    MATCHES_CONTROL = "control performs as well"
    OUTLIER_DEPENDENT = "rests on a few outliers"
    NOT_SIGNIFICANT = "not significant after trials and dependence"
    TOO_FEW_TRADES = "too few trades to measure"


@dataclass(frozen=True)
class Assessment:
    """One candidate's result at one horizon, with the reason it survived or not."""

    name: str
    horizon: int
    result: BacktestResult
    control: BacktestResult | None
    verdict: str

    @property
    def kept(self) -> bool:
        return self.verdict == Verdict.KEEP


#: A signal whose control does nearly as well is measuring the population, not
#: the signal. Ratio of control mean to signal mean above this fails the check.
CONTROL_TOLERANCE = 0.7

#: Below this many trades the statistics say nothing, whatever they print.
MIN_TRADES = 30


def judge(result: BacktestResult, control: BacktestResult | None) -> str:
    """Apply the disqualifying gates, in the order they can disqualify.

    Order matters for readability rather than correctness: a strategy with no
    gross edge should be reported as having no edge, not as failing a cost test
    it never reached.
    """
    if result.n_trades < MIN_TRADES:
        return Verdict.TOO_FEW_TRADES
    if result.mean_return <= 0:
        return Verdict.NO_EDGE
    if result.costed and not result.survives_costs:
        return Verdict.EATEN_BY_COSTS
    if control is not None and control.n_trades >= MIN_TRADES:
        # A control that keeps most of the signal's edge means the edge belongs
        # to the population. This caught a real bug once: a labeller
        # misalignment made negate(BUY) return results identical to BUY.
        if control.mean_return > 0 and (
            control.mean_return / result.mean_return >= CONTROL_TOLERANCE
        ):
            return Verdict.MATCHES_CONTROL
    if result.outlier_dependent:
        return Verdict.OUTLIER_DEPENDENT
    if not result.significant:
        return Verdict.NOT_SIGNIFICANT
    return Verdict.KEEP


def sweep(
    candidates: Sequence[Candidate],
    labels: Sequence[Label],
    payloads: Sequence[dict],
    *,
    registry: TrialRegistry,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    costs: Callable[[Label], float] | None = None,
    with_controls: bool = True,
    partition: str = "train",
) -> list[Assessment]:
    """Measure every runnable candidate at every horizon.

    Blocked candidates are skipped but reported, so the difference between
    "tested and failed" and "never tested" stays visible in the output.
    """
    out: list[Assessment] = []
    for cand in candidates:
        if not cand.runnable:
            continue
        for horizon in horizons:
            result = run(
                labels, payloads,
                hypothesis=cand.name, rationale=cand.rationale,
                registry=registry, selector=cand.selector,
                horizon=horizon, partition=partition, costs=costs,
            )
            control = None
            if with_controls and cand.selector is not everything:
                control = run(
                    labels, payloads,
                    hypothesis=f"{cand.name} [control]",
                    rationale=f"complement of {cand.name}; if this performs "
                              "equally the edge belongs to the population",
                    registry=registry, selector=negate(cand.selector),
                    horizon=horizon, partition=partition, costs=costs,
                )
            out.append(Assessment(cand.name, horizon, result, control,
                                  judge(result, control)))
    return out


def report(assessments: Sequence[Assessment],
           candidates: Sequence[Candidate] = ()) -> str:
    """Tabulate everything measured, survivors first, with the blocked listed.

    Deliberately prints failures and blocked candidates rather than only what
    survived. A report that shows only winners is the exact artefact this whole
    apparatus exists to avoid producing.
    """
    lines: list[str] = []
    header = (
        f"{'candidate':<30}{'h':>3}{'trades':>8}{'mean':>9}{'net':>9}"
        f"{'t(cl)':>7}{'DSR':>7}  verdict"
    )
    lines.append(header)
    lines.append("-" * len(header))

    ordered = sorted(assessments, key=lambda a: (not a.kept, -a.result.mean_return))
    for a in ordered:
        r = a.result
        net = f"{r.mean_return_net:+.2%}" if r.costed else "  --  "
        lines.append(
            f"{a.name[:29]:<30}{a.horizon:>3}{r.n_trades:>8,}"
            f"{r.mean_return:>+9.2%}{net:>9}"
            f"{r.t_stat_clustered:>+7.2f}{r.deflated_sharpe:>7.3f}  {a.verdict}"
        )

    kept = [a for a in assessments if a.kept]
    lines.append("")
    lines.append(f"{len(kept)} of {len(assessments)} measurements survived "
                 f"every gate.")

    blocked = [c for c in candidates if not c.runnable]
    if blocked:
        lines.append("")
        lines.append(f"{len(blocked)} candidates NOT measured -- these are "
                     "untested, not rejected:")
        for c in blocked:
            lines.append(f"  {c.name:<28} blocked by: {c.blocked_by}")

    if kept:
        lines.append("")
        lines.append("Survivors still owe an out-of-sample test. The holdout is "
                     "locked for exactly this moment.")
    return "\n".join(lines)


def priors_vs_outcomes(assessments: Sequence[Assessment],
                       candidates: Sequence[Candidate]) -> str:
    """Compare what we expected against what happened.

    The point of recording a prior is to be shown wrong by it. A prior that is
    never checked is just an opinion that got to act without being measured.
    """
    by_name = {c.name: c for c in candidates if c.prior}
    if not by_name:
        return "no priors recorded"
    lines = ["prior vs outcome:", ""]
    for name, cand in by_name.items():
        rows = [a for a in assessments if a.name == name]
        if not rows:
            lines.append(f"  {name:<28} NOT MEASURED -- prior stands unchecked")
            continue
        kept = sum(1 for a in rows if a.kept)
        lines.append(f"  {name:<28} expected: {cand.prior[:44]}")
        lines.append(f"  {'':<28} actual  : survived {kept}/{len(rows)} horizons")
    return "\n".join(lines)
