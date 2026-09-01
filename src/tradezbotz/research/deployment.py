"""The bar a candidate must clear before money is placed on it.

**Why this file exists and why it exists now.** Deciding what counts as good
enough *after* seeing which candidate won is how a threshold becomes a
rationalisation. The only honest moment to set it is while the results are
unknown, and that window is closing: `measure` runs nightly from now on.

**The enforcement is not that the numbers cannot change. It is that changing
them leaves a record.** The criteria live in a version-controlled JSON file, so
lowering a threshold is a commit with a diff and a message, sitting next to the
result that motivated it. That is a far stronger guarantee than a constant
buried in code, and a far more honest one than pretending the bar is immutable.

**Every criterion here is stricter than the corresponding research gate**, and
deliberately. `sweep.judge` decides whether a measurement means anything;
this decides whether it justifies risk. A result can be perfectly sound and
still not worth trading.

The one criterion with no counterpart in the research gates is the live
haircut. A backtest Sharpe of 4 arriving live at 0.5 is the documented base
rate rather than bad luck -- biases stack, each shaving the number. So the
requirement is not that the strategy is profitable as measured, but that it
survives losing half its gross edge. If it is not viable at half, it is not
viable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

#: Where the committed criteria live. JSON rather than Python so a change is a
#: data diff a reviewer can read, not a code change they must interpret.
DEFAULT_CRITERIA_PATH = Path("docs/deployment-criteria.json")


class DeploymentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Criteria:
    """Thresholds a candidate must clear to be deployable.

    Defaults are the proposed values. `confirmed` gates the whole thing: until
    the operator has set the three figures only they can set, nothing passes,
    however good it looks.
    """

    # --- statistical, all stricter than the research gates -------------------
    min_deflated_sharpe: float = 0.95
    min_trades: int = 200
    min_coverage: float = 0.40
    max_fallback_share: float = 0.10
    max_control_ratio: float = 0.50
    require_holdout_confirmation: bool = True

    #: Fraction of gross edge assumed lost between backtest and live. The
    #: strategy must still clear costs after this is removed.
    assumed_live_haircut: float = 0.50

    # --- the operator's, and undefaultable -----------------------------------
    #: Total capital at risk. Drives position size and therefore market impact,
    #: so it is an input to the cost model rather than a footnote.
    capital_at_risk: float = 0.0
    #: Peak-to-trough loss at which the system halts itself without asking.
    #: Without this there is no plan for being wrong, only for being right.
    max_drawdown_halt: float = 0.0
    #: Phase 3b money: the smallest real stake whose only job is measuring
    #: realised slippage against the model. Deliberately separate from
    #: `capital_at_risk`, because they answer different questions and
    #: conflating them corrupts the backtest.
    #:
    #: Sizing the BACKTEST off a $100 stake would charge impact for ~$10 orders,
    #: which is approximately zero -- and every net return would then flatter
    #: any size you later actually traded. The backtest must be pessimistic
    #: relative to reality, never optimistic, so it is sized off the capital you
    #: might REACH rather than the capital you start with.
    live_test_capital: float = 0.0
    #: Positions to hold during the live test. One or two, not ten: at $10 a
    #: position only 24% of cached names are buyable as a whole share, and only
    #: 56% of listed US equities are fractionable at all.
    live_test_positions: int = 1
    #: How many positions may be open at once. Needed to turn total capital
    #: into a position size, and a position size is what determines market
    #: impact -- so this is a cost-model input, not portfolio housekeeping.
    max_concurrent_positions: int = 10
    #: Position ceiling as a share of average daily volume. Overrides
    #: `costs.MAX_PARTICIPATION` (10%), which is the point: the research default
    #: is what is merely executable, this is what we are willing to do.
    max_participation: float = 0.05

    #: Set true only when the three figures above have been chosen deliberately.
    confirmed: bool = False
    #: Free text: who signed off, when, and on what basis.
    signed_off: str = ""

    @property
    def position_notional(self) -> float:
        """Capital per position, which is what the cost model actually needs.

        Total capital alone cannot price impact: impact is a function of how
        much of a day's volume ONE order consumes. Splitting evenly is crude and
        stated as such -- a real allocator would size by conviction or
        volatility -- but a crude stated size beats the implied size of zero
        that the backtest used before this existed.
        """
        if self.capital_at_risk <= 0 or self.max_concurrent_positions <= 0:
            return 0.0
        return self.capital_at_risk / self.max_concurrent_positions

    def unset_operator_values(self) -> list[str]:
        missing = []
        if self.capital_at_risk <= 0:
            missing.append("capital_at_risk")
        if self.max_drawdown_halt <= 0:
            missing.append("max_drawdown_halt")
        return missing


@dataclass(frozen=True)
class GateResult:
    candidate: str
    horizon: int
    passed: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [f"{head}  {self.candidate}  h={self.horizon}"]
        for f in self.failures:
            lines.append(f"    x {f}")
        for n in self.notes:
            lines.append(f"    - {n}")
        return "\n".join(lines)


def load(path: str | Path = DEFAULT_CRITERIA_PATH) -> Criteria:
    p = Path(path)
    if not p.exists():
        raise DeploymentError(
            f"no deployment criteria at {p}. The bar must be committed before "
            "it can be applied -- see docs/DEPLOYMENT-CRITERIA.md."
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    known = {f for f in Criteria.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise DeploymentError(
            f"unknown criteria in {p}: {sorted(unknown)}. A typo here would "
            "silently drop a threshold, so it is refused rather than ignored."
        )
    return Criteria(**raw)


def save(criteria: Criteria, path: str | Path = DEFAULT_CRITERIA_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(criteria), indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")


def evaluate(assessment, criteria: Criteria, *,
             holdout_confirmed: bool = False) -> GateResult:
    """Judge one measured candidate against the committed bar.

    Takes a `sweep.Assessment`. Returns every reason it failed rather than the
    first: a candidate short on three criteria is a different situation from one
    short on a single threshold, and stopping at the first hides that.
    """
    failures: list[str] = []
    notes: list[str] = []
    r = assessment.result

    if not criteria.confirmed:
        failures.append(
            "criteria are not confirmed: the operator has not signed off on "
            "capital, drawdown halt and position limits. Nothing deploys until "
            "they have, however good the numbers look.")
    missing = criteria.unset_operator_values()
    if missing:
        failures.append(f"undefined operator values: {', '.join(missing)}")

    if not assessment.kept:
        failures.append(f"research verdict is not KEEP: {assessment.verdict}")

    if r.deflated_sharpe < criteria.min_deflated_sharpe:
        failures.append(
            f"deflated Sharpe {r.deflated_sharpe:.3f} below "
            f"{criteria.min_deflated_sharpe:.2f}")
    if r.n_trades < criteria.min_trades:
        failures.append(
            f"{r.n_trades:,} trades below {criteria.min_trades:,}")
    if assessment.coverage < criteria.min_coverage:
        failures.append(
            f"coverage {assessment.coverage:.1%} below "
            f"{criteria.min_coverage:.0%}")
    if assessment.fallback_share > criteria.max_fallback_share:
        failures.append(
            f"{assessment.fallback_share:.0%} of trades priced by the fallback "
            f"constant, above {criteria.max_fallback_share:.0%}")

    if assessment.control is not None and r.mean_return > 0:
        ratio = assessment.control.mean_return / r.mean_return
        if ratio > criteria.max_control_ratio:
            failures.append(
                f"control keeps {ratio:.0%} of the edge, above "
                f"{criteria.max_control_ratio:.0%}")
    elif assessment.control is None:
        notes.append("no control was run; the population comparison is missing")

    # The haircut. Not "is it profitable" but "is it still profitable after
    # losing half its gross edge", which is the documented base rate for the
    # gap between a backtest and its live counterpart.
    if not r.costed:
        failures.append("uncosted: a gross edge on this universe is not a result")
    else:
        cost = r.mean_return - r.mean_return_net
        haircut_edge = r.mean_return * (1.0 - criteria.assumed_live_haircut)
        if haircut_edge - cost <= 0:
            failures.append(
                f"does not survive a {criteria.assumed_live_haircut:.0%} live "
                f"haircut: {haircut_edge:+.2%} gross against {cost:.2%} costs")
        else:
            notes.append(
                f"survives the haircut with {haircut_edge - cost:+.2%} per trade")

    if criteria.require_holdout_confirmation and not holdout_confirmed:
        failures.append("holdout not yet confirmed for this candidate")

    return GateResult(assessment.name, assessment.horizon,
                      not failures, tuple(failures), tuple(notes))


def report(results: Sequence[GateResult], criteria: Criteria) -> str:
    lines = ["deployment gate", "=" * 60]
    if not criteria.confirmed:
        lines.append("CRITERIA UNCONFIRMED -- nothing can pass. See "
                     "docs/DEPLOYMENT-CRITERIA.md.")
        lines.append("")
    passed = [r for r in results if r.passed]
    for r in sorted(results, key=lambda x: (not x.passed, x.candidate)):
        lines.append(r.summary())
    lines.append("")
    lines.append(f"{len(passed)} of {len(results)} measured candidates are "
                 "deployable.")
    if criteria.signed_off:
        lines.append(f"criteria signed off: {criteria.signed_off}")
    return "\n".join(lines)
