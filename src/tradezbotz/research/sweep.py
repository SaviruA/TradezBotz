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

from dataclasses import dataclass
from typing import Callable, Sequence

from .backtest import BacktestResult, Selector, negate, run
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
    #: Whether a complement control means anything for this candidate. False for
    #: a baseline that already selects everything: its complement is the empty
    #: set, so the "control" would be a registered trial measuring nothing.
    controlled: bool = True

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
    COST_NOT_MEASURED = "cost gate rests on the fallback constant"
    THIN_COVERAGE = "population too thinly covered to generalise"


@dataclass(frozen=True)
class Assessment:
    """One candidate's result at one horizon, with the reason it survived or not."""

    name: str
    horizon: int
    result: BacktestResult
    control: BacktestResult | None
    verdict: str
    #: Why no control was run, when none was. Empty when one was run or when
    #: controls were switched off for the whole sweep. Recorded because "the
    #: control was not run" and "the control was run and matched" are opposite
    #: readings of the same blank column.
    control_note: str = ""
    #: Share of the event population that was labellable, and share of trades
    #: charged the fallback cost. Carried on the assessment rather than printed
    #: once, because a reader comparing two rows needs to know they were
    #: measured on the same footing.
    coverage: float = 1.0
    fallback_share: float = 0.0

    @property
    def kept(self) -> bool:
        return self.verdict == Verdict.KEEP


#: A signal whose control does nearly as well is measuring the population, not
#: the signal. Ratio of control mean to signal mean above this fails the check.
CONTROL_TOLERANCE = 0.7

#: Below this many trades the statistics say nothing, whatever they print.
MIN_TRADES = 30

#: Share of trades whose cost came from the fallback constant rather than from a
#: spread estimate, above which the cost gate is not a measurement.
#:
#: `survives_costs` decides KEEP. If most trades were charged a flat 93bps that
#: somebody chose, then KEEP is a statement about that constant rather than
#: about the strategy. A quarter is already generous.
MAX_FALLBACK_SHARE = 0.25

#: Share of the candidate population that must actually have been labellable.
#: A result computed on 0.3% of events is a result about whichever symbols
#: happened to be in the price cache, and the first candidate to clear the
#: trade floor would be selected by data availability rather than by edge.
MIN_COVERAGE = 0.20

#: Partition name for the sealed out-of-sample window. Matched by string because
#: that is what `run` records on the trial, and a typo here should fail loudly
#: rather than quietly sweeping the holdout under a name nothing guards.
HOLDOUT = "holdout"


class SweepError(RuntimeError):
    pass


def _require_holdout_declared(candidates: Sequence[Candidate],
                              registry: TrialRegistry) -> None:
    """Refuse a holdout sweep for anything not already declared a finalist.

    `splits.Split` seals the holdout behind `unlock_holdout`, which records the
    access per hypothesis. Nothing connected that seal to the sweep: passing
    `partition="holdout"` measured the holdout and recorded trials against it
    with no access logged at all, which is the exact failure the split exists to
    prevent -- and worse than peeking manually, because a sweep touches every
    candidate at once.

    Requiring a per-candidate declaration is the point. The holdout can answer
    "does this specific finalist hold up", once. It cannot answer "which of my
    forty candidates works", because that question is selection, and running it
    here consumes the only clean data left.
    """
    undeclared = [
        c.name for c in candidates
        if c.runnable and registry.holdout_accesses(c.name) == 0
    ]
    if not undeclared:
        return
    shown = ", ".join(undeclared[:5])
    more = f" (+{len(undeclared) - 5} more)" if len(undeclared) > 5 else ""
    raise SweepError(
        f"refusing to sweep the holdout: {len(undeclared)} candidates have no "
        f"recorded access -- {shown}{more}. Declare each finalist first with "
        "Split.unlock_holdout(registry, name, reason). If that feels like too "
        "much friction for the number of candidates involved, that is the "
        "signal: the holdout is for confirming a finalist, not for selecting "
        "among a backlog."
    )


def judge(result: BacktestResult, control: BacktestResult | None,
          *, fallback_share: float = 0.0, coverage: float = 1.0) -> str:
    """Apply the disqualifying gates, in the order they can disqualify.

    Order matters for readability rather than correctness: a strategy with no
    gross edge should be reported as having no edge, not as failing a cost test
    it never reached.

    The two gates before any statistic is read are about whether the numbers
    mean anything at all:

    **Coverage.** A result computed on a fraction of the population describes
    whichever symbols happened to be in the price cache. Without this gate the
    first candidate to clear the trade floor is selected by data availability,
    and it would be reported identically to one selected by edge.

    **Cost provenance.** `survives_costs` decides KEEP, and when most trades are
    charged the fallback constant, KEEP is a claim about that constant. Both are
    reported as distinct verdicts rather than folded into "not significant",
    because the fix for each is to get more data, not to abandon the hypothesis.
    """
    if coverage < MIN_COVERAGE:
        return Verdict.THIN_COVERAGE
    if result.n_trades < MIN_TRADES:
        return Verdict.TOO_FEW_TRADES
    if result.costed and fallback_share > MAX_FALLBACK_SHARE:
        return Verdict.COST_NOT_MEASURED
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
    dataset: str = "",
    coverage: float = 1.0,
    fallback_share: float = 0.0,
) -> list[Assessment]:
    """Measure every runnable candidate at every horizon.

    Blocked candidates are skipped but reported, so the difference between
    "tested and failed" and "never tested" stays visible in the output.

    Sweeping the holdout is refused unless every candidate in it has already
    been declared a finalist through `splits.unlock_holdout`. The check runs
    before any measurement, so a sweep that would have touched the holdout
    improperly touches none of it.
    """
    if partition == HOLDOUT:
        _require_holdout_declared(candidates, registry)

    pairs = list(zip(payloads, labels))
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
                dataset=dataset,
            )
            control, note = None, ""
            if with_controls and cand.controlled:
                # Count the complement before running it. An empty or tiny
                # complement is not a control that failed, it is a control that
                # could never have said anything -- and running it anyway would
                # register a trial, raising the Deflated Sharpe bar for every
                # other candidate in exchange for no information.
                #
                # This replaces an `is not everything` identity test, which only
                # recognised the literal baseline function and missed any
                # selector that happens to admit everything -- `all_of(buy)`
                # where every event is a buy, say.
                complement = negate(cand.selector)
                n_complement = sum(
                    1 for p, lab in pairs
                    if horizon in lab.returns and complement(p, lab)
                )
                if n_complement >= MIN_TRADES:
                    control = run(
                        labels, payloads,
                        hypothesis=f"{cand.name} [control]",
                        rationale=f"complement of {cand.name}; if this performs "
                                  "equally the edge belongs to the population",
                        registry=registry, selector=negate(cand.selector),
                        horizon=horizon, partition=partition, costs=costs,
                        dataset=dataset,
                    )
                else:
                    note = (f"complement holds {n_complement} trades, below the "
                            f"{MIN_TRADES} floor; no control run")
            elif with_controls:
                note = "control not meaningful for this candidate"
            out.append(Assessment(
                cand.name, horizon, result, control,
                judge(result, control, fallback_share=fallback_share,
                      coverage=coverage),
                note, coverage, fallback_share))
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
        f"{'t(cl)':>7}{'DSR':>7}{'cov':>6}  verdict"
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
            f"{r.t_stat_clustered:>+7.2f}{r.deflated_sharpe:>7.3f}"
            f"{a.coverage:>6.0%}  {a.verdict}"
        )

    kept = [a for a in assessments if a.kept]
    lines.append("")
    lines.append(f"{len(kept)} of {len(assessments)} measurements survived "
                 f"every gate.")

    # Provenance, stated once and prominently. A reader who does not know that
    # the cost figures came from a constant will read `net` as a measurement.
    if assessments:
        cov = assessments[0].coverage
        fb = assessments[0].fallback_share
        lines.append("")
        lines.append(f"coverage {cov:.1%} of the event population; "
                     f"{fb:.0%} of trades charged the fallback cost constant.")
        if cov < MIN_COVERAGE:
            lines.append(f"  Below the {MIN_COVERAGE:.0%} floor: every row above "
                         "describes whichever symbols happened to be cached, so "
                         "nothing here is a finding about a strategy.")
        if fb > MAX_FALLBACK_SHARE:
            lines.append(f"  Above the {MAX_FALLBACK_SHARE:.0%} fallback ceiling: "
                         "the cost gate is reporting a chosen constant, not a "
                         "measured spread.")

    # A missing control reads as "no control needed" unless it says otherwise,
    # and those are opposite claims about the same empty column.
    uncontrolled = [a for a in assessments if a.control is None and a.control_note]
    if uncontrolled:
        lines.append("")
        lines.append(f"{len(uncontrolled)} measurements ran without a control:")
        for a in uncontrolled[:10]:
            lines.append(f"  {a.name[:28]:<30}h={a.horizon:<3} {a.control_note}")
        if len(uncontrolled) > 10:
            lines.append(f"  ... and {len(uncontrolled) - 10} more")

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
