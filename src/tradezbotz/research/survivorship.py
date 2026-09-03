"""Bounding the survivorship bias instead of noting that it exists.

The coverage report has always said what share of the population went
unmeasured, and warned that returns are "biased upward by an amount nothing
here can estimate". That last clause was wrong, and it let the largest known
bias in the system sit as a caveat rather than a number.

**The estimate.** An event goes unmeasured mostly because its symbol has no
cached bars. That is not missing at random: a company that stopped trading is
far likelier to be unpriced than one that did not, and a company that stopped
trading did so for a reason. Shumway (1997) established the bias in CRSP data;
Shumway & Warther (1999) put the corrected return for a PERFORMANCE-RELATED
NASDAQ delisting at roughly -55%, and found the NASDAQ delisting bias 4.7x
larger than the NYSE/AMEX bias documented earlier. This universe is microcaps,
overwhelmingly NASDAQ and OTC, so -55% is the applicable figure and -30% would
be the wrong one.

**What this does not claim.** It is a BOUND, not a correction. The unmeasured
delisted events are not silently assigned a return and folded into the result;
they are used to compute the worst case the measurement is consistent with, and
both numbers are reported. Quietly applying it would replace a known bias with
an assumed one, which is not an improvement -- it just moves where the guess
lives.

The arithmetic is a linear map, which is what makes it usable per candidate:

    bounded(m) = w * m + (1 - w) * d

where `d` is the delisting return and `w` the share of the extended population
that was actually measured. `w` depends only on counts, so one map applies to
every candidate measured on that population.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Shumway & Warther (1999), performance-related NASDAQ delistings. Applied
#: rather than the -30% NYSE/AMEX figure because this universe is microcaps.
DELISTING_RETURN = -0.55

#: Classifications whose unmeasured events are treated as delistings. "unknown"
#: is deliberately EXCLUDED from the base case and reported separately: it
#: certainly contains delistings, but folding it in would overstate a bound
#: that is already the pessimistic end of the range.
DELISTED = "delisted"


@dataclass(frozen=True)
class Bound:
    """A linear map from a measured mean to its worst-case population mean."""

    measured: int
    unmeasured_delisted: int
    unmeasured_unknown: int
    delisting_return: float

    @property
    def weight(self) -> float:
        """Share of the extended population that was actually measured."""
        total = self.measured + self.unmeasured_delisted
        return self.measured / total if total else 1.0

    @property
    def binding(self) -> bool:
        return self.unmeasured_delisted > 0

    def apply(self, mean: float) -> float:
        w = self.weight
        return w * mean + (1.0 - w) * self.delisting_return

    def describe(self, examples: tuple[float, ...] = (0.02, 0.04, 0.06)) -> str:
        if not self.binding:
            return ("survivorship bound: no unmeasured delisted events; the "
                    "measured population is the population")
        lines = [
            f"survivorship bound: {self.unmeasured_delisted:,} delisted events "
            f"went unmeasured against {self.measured:,} measured, so a mean "
            f"survives at weight {self.weight:.3f}.",
            f"  Worst case charges them the delisting return "
            f"({self.delisting_return:.0%}, Shumway & Warther 1999, "
            f"performance-related NASDAQ):",
        ]
        for m in examples:
            lines.append(f"    a measured {m:+.2%} bounds to {self.apply(m):+.2%}")
        lines.append(
            "  This is a BOUND, not a correction -- nothing here is adjusted, "
            "because replacing a known bias with an assumed one moves the guess "
            "rather than removing it.")
        if self.unmeasured_unknown:
            lines.append(
                f"  {self.unmeasured_unknown:,} unmeasured events are of "
                f"UNKNOWN classification and are excluded from the bound. They "
                f"certainly contain delistings, so the true worst case is "
                f"beyond this one.")
        return "\n".join(lines)


def bound(buckets: dict[str, list[int]],
          delisting_return: float = DELISTING_RETURN) -> Bound:
    """Build the bound from a {classification: [seen, labelled]} breakdown.

    `seen` counts events whose symbol carries that classification; `labelled`
    counts those that produced at least one return.
    """
    measured = sum(ok for _seen, ok in buckets.values())
    seen_d, ok_d = buckets.get(DELISTED, [0, 0])
    seen_u, ok_u = buckets.get("unknown", [0, 0])
    return Bound(
        measured=measured,
        unmeasured_delisted=max(seen_d - ok_d, 0),
        unmeasured_unknown=max(seen_u - ok_u, 0),
        delisting_return=delisting_return,
    )
