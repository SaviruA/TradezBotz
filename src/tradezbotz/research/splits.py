"""Train / validation / holdout partitioning, with the holdout locked.

A holdout is only worth having while it stays untouched. The failure mode is
never a deliberate decision to peek -- it is looking "just once" to check
something, then a second time after a tweak, until the holdout has quietly
become another validation set and there is nothing clean left to confirm on.

So `train` and `validation` are ordinary properties, while `holdout` raises
unless the caller passes an explicit hypothesis and reason. Every access is
recorded in the trial registry. A finalist should appear there exactly once; a
second entry for the same hypothesis is itself a finding about the process.

The split is chronological, never random. Shuffling market data across time
leaks the future into the past through overlapping horizons and shared regimes,
and a randomly-split backtest can look excellent while being untradeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .trials import TrialRegistry


class HoldoutLocked(RuntimeError):
    pass


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def __str__(self) -> str:
        return f"{self.start} .. {self.end}"


@dataclass(frozen=True)
class Split:
    """Chronological three-way partition of the labelling window.

    Default shape reflects what two years of free price data can support: most
    of it for fitting, a slice for selection among candidates, and the most
    recent months sealed. The holdout is last precisely because the recent
    regime is the one a live strategy will meet first.
    """

    train: DateRange
    validation: DateRange
    _holdout: DateRange

    @property
    def holdout(self) -> DateRange:
        raise HoldoutLocked(
            "The holdout is sealed. Use unlock_holdout(registry, hypothesis, "
            "reason) for a declared finalist. Reaching for it during exploration "
            "is how a holdout silently becomes a second validation set."
        )

    def unlock_holdout(
        self, registry: "TrialRegistry", hypothesis: str, reason: str
    ) -> DateRange:
        """Open the holdout for one declared finalist, recording the access."""
        if not hypothesis.strip() or not reason.strip():
            raise HoldoutLocked("both a hypothesis and a reason are required")
        n = registry.record_holdout_access(hypothesis.strip(), reason.strip())
        if n > 1:
            print(
                f"WARNING: holdout opened {n} times for '{hypothesis}'. "
                "After the first look it is no longer an out-of-sample test, "
                "and results from it should be reported as validation, not "
                "confirmation."
            )
        return self._holdout

    def of(self, day: date) -> str:
        """Which partition a date belongs to."""
        if self.train.contains(day):
            return "train"
        if self.validation.contains(day):
            return "validation"
        if self._holdout.contains(day):
            return "holdout"
        return "outside"

    def describe(self) -> str:
        return (
            f"train      {self.train}  ({self.train.days}d)\n"
            f"validation {self.validation}  ({self.validation.days}d)\n"
            f"holdout    {self._holdout}  ({self._holdout.days}d)  [sealed]"
        )


def chronological_split(
    start: date,
    end: date,
    *,
    validation_frac: float = 0.2,
    holdout_frac: float = 0.25,
) -> Split:
    """Partition [start, end] in time order: train, then validation, then holdout.

    Holdout defaults to the most recent 25% -- roughly six months of a two-year
    window. Smaller than that and it cannot support a meaningful number of
    trade-sparse events; much larger and there is too little left to fit on.
    """
    if end <= start:
        raise ValueError("end must be after start")
    if not 0 < validation_frac < 1 or not 0 < holdout_frac < 1:
        raise ValueError("fractions must be between 0 and 1")
    if validation_frac + holdout_frac >= 1:
        raise ValueError("train would be empty")

    total = (end - start).days
    holdout_days = int(total * holdout_frac)
    validation_days = int(total * validation_frac)

    holdout_start = end - timedelta(days=holdout_days)
    validation_start = holdout_start - timedelta(days=validation_days)

    return Split(
        train=DateRange(start, validation_start - timedelta(days=1)),
        validation=DateRange(validation_start, holdout_start - timedelta(days=1)),
        _holdout=DateRange(holdout_start, end),
    )


def filter_events(events, split: Split, partition: str) -> list:
    """Keep only events whose observation date falls in `partition`.

    Filtering on `observed_at` rather than the transaction date keeps the split
    consistent with the point-in-time rule the event store enforces.
    """
    from datetime import datetime

    out = []
    for e in events:
        observed = e["observed_at"] if isinstance(e, dict) else e.observed_at
        if isinstance(observed, str):
            observed = datetime.fromisoformat(observed)
        if split.of(observed.date()) == partition:
            out.append(e)
    return out
