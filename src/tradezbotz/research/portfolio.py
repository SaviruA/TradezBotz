"""Turning a signal into something a person could actually hold.

Everything upstream of this is an EVENT STUDY: every qualifying event is a
trade, unweighted, with no capital constraint. That is the right instrument for
"does this signal exist" and it is not a strategy. The gap is not cosmetic --
measured concurrency ran to a median of 591 distinct symbols at h=5 against an
assumed 10, and at h=60 it is worse. At $25,000 that is $42 a position, which
is not a position.

**A capacity constraint is not a detail bolted on at the end. It changes which
trades you take, and therefore what you measured.** Once slots are finite, some
signals are refused, and *which* ones are refused is a new selection decision.
That decision has to be registered as a trial like any other, because choosing
the ranking that produces the best backtest is precisely the search this whole
apparatus exists to charge for.

**Arbitrary order is the control, and it is the important one.** A ranked
portfolio must be compared against taking the same number of positions in
whatever order they arrived. If ranking does not beat arrival order, the
ranking carries nothing and any improvement came from the capacity constraint
changing the population -- which is a different finding, and one that would
otherwise be misread as skill.

**Rejected signals are counted and reported.** A strategy that takes 5% of its
signals is not the strategy that was measured upstream, however similar the
per-trade return looks. Without that number the two are indistinguishable in a
report.

One approximation, stated rather than hidden: slots are released on calendar
days scaled by 7/5 rather than on a real session calendar, because a `Label`
carries an entry day and a horizon in sessions but no trading calendar. This is
a capacity model, not a pricing model -- an exit landing a day early or late
changes when a slot frees, never what a trade earned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Sequence

from .labeler import Label

#: (payload, label) -> score. Higher is taken first.
Ranker = Callable[[dict, Label], float]


@dataclass(frozen=True)
class PortfolioResult:
    horizon: int
    max_positions: int
    n_signals: int
    n_taken: int
    n_rejected_no_slot: int
    returns: list[float] = field(default_factory=list)
    #: Distinct symbols held on each day a position was open.
    peak_concurrent: int = 0

    @property
    def participation(self) -> float:
        """Share of signals actually tradeable at this capacity."""
        return self.n_taken / self.n_signals if self.n_signals else 0.0

    @property
    def mean_return(self) -> float:
        return sum(self.returns) / len(self.returns) if self.returns else 0.0

    def describe(self) -> str:
        return (
            f"portfolio h={self.horizon} cap={self.max_positions}: took "
            f"{self.n_taken:,} of {self.n_signals:,} signals "
            f"({self.participation:.1%}), refused "
            f"{self.n_rejected_no_slot:,} for want of a slot; mean "
            f"{self.mean_return:+.2%}, peak {self.peak_concurrent} concurrent"
        )


def _exit_day(entry: date, horizon: int) -> date:
    # Sessions to calendar days. See the module note: this decides when a slot
    # frees, never what a trade earned.
    return entry + timedelta(days=int(round(horizon * 7 / 5)))


def simulate(payloads: Sequence[dict], labels: Sequence[Label], horizon: int, *,
             max_positions: int, rank: Ranker | None = None) -> PortfolioResult:
    """Walk the signals forward, holding at most `max_positions` at a time.

    `rank` scores signals competing for the same slot; higher goes first. None
    means arrival order, which is the control rather than a lesser option.

    A symbol already held is not re-entered. Doubling into a name is a
    different strategy with different risk, and allowing it silently would let
    one issuer consume the whole book.
    """
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")

    eligible: list[tuple[date, dict, Label]] = []
    for payload, label in zip(payloads, labels):
        if label.entry_day is None or horizon not in label.returns:
            continue
        if not label.symbol:
            continue
        eligible.append((label.entry_day, payload, label))
    eligible.sort(key=lambda row: row[0])

    by_day: dict[date, list[tuple[dict, Label]]] = {}
    for day, payload, label in eligible:
        by_day.setdefault(day, []).append((payload, label))

    open_until: dict[str, date] = {}
    returns: list[float] = []
    taken = rejected = peak = 0

    for day in sorted(by_day):
        for symbol, until in list(open_until.items()):
            if until <= day:
                del open_until[symbol]

        todays = by_day[day]
        if rank is not None:
            # Stable within equal scores, so an unranked tie keeps arrival
            # order rather than depending on dict iteration.
            todays = sorted(todays, key=lambda pl: -rank(pl[0], pl[1]))

        for payload, label in todays:
            symbol = label.symbol.upper()
            if symbol in open_until:
                continue
            if len(open_until) >= max_positions:
                rejected += 1
                continue
            open_until[symbol] = _exit_day(label.entry_day, horizon)
            returns.append(label.returns[horizon])
            taken += 1
        peak = max(peak, len(open_until))

    return PortfolioResult(
        horizon=horizon,
        max_positions=max_positions,
        n_signals=len(eligible),
        n_taken=taken,
        n_rejected_no_slot=rejected,
        returns=returns,
        peak_concurrent=peak,
    )


def compare(payloads: Sequence[dict], labels: Sequence[Label], horizon: int, *,
            max_positions: int, rank: Ranker,
            rank_name: str = "ranked") -> str:
    """Ranked against arrival order at the same capacity.

    The only comparison that isolates the ranking. Against the uncapped event
    study a ranked portfolio differs in two ways at once -- fewer trades AND a
    different selection -- and attributing the difference to the ranking would
    be unfounded.
    """
    ranked = simulate(payloads, labels, horizon,
                      max_positions=max_positions, rank=rank)
    arrival = simulate(payloads, labels, horizon,
                       max_positions=max_positions, rank=None)
    spread = ranked.mean_return - arrival.mean_return
    lines = [
        f"  {rank_name:<16} {ranked.describe()}",
        f"  {'arrival order':<16} {arrival.describe()}",
        f"  ranking is worth {spread:+.2%} a trade against arrival order at "
        f"the same capacity",
    ]
    if spread <= 0:
        lines.append(
            "  The ranking carries nothing here: any difference from the "
            "uncapped study came from the capacity constraint changing the "
            "population, not from choosing well.")
    return "\n".join(lines)
