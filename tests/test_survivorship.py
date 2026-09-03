"""Tests for the survivorship bound.

The coverage report used to say returns were "biased upward by an amount
nothing here can estimate", and only said it when delisted labelling fell below
HALF of listed labelling. At 78.4% against 87.9% it stayed silent -- while the
bias was still worth several points of return on the baseline.

These tests fix two things: that the arithmetic is right, and that the bound
stays a bound. Silently applying it would replace a known bias with an assumed
one, which moves the guess rather than removing it.
"""

from __future__ import annotations

import pytest

from tradezbotz.research.survivorship import DELISTING_RETURN, Bound, bound


def _buckets(**kw):
    """{classification: [seen, labelled]}."""
    return {k: list(v) for k, v in kw.items()}


# --- the real numbers from the run that motivated this ----------------------

REAL = _buckets(delisted=(6_637, 5_206), listed=(28_444, 25_009),
                otc=(1_290, 535), unknown=(3_629, 3_059))


def test_the_bound_is_material_on_the_observed_population():
    """1,431 unmeasured delisted events against 33,809 measured. At -55% this
    takes a +3.46% baseline down near +1%, which is most of the effect."""
    b = bound(REAL)

    assert b.unmeasured_delisted == 1_431
    assert b.apply(0.0346) < 0.015


def test_the_weight_is_the_measured_share_of_the_extended_population():
    b = bound(REAL)

    assert b.weight == pytest.approx(33_809 / (33_809 + 1_431))


def test_a_gap_too_small_for_the_old_warning_still_produces_a_bound():
    """The regression in spirit: delisted labelling at 78% of a population is
    nowhere near the old 50% trigger, and was silently fine."""
    b = bound(REAL)

    assert b.binding
    assert "survivorship bound" in b.describe()


# --- the arithmetic ---------------------------------------------------------

def test_the_map_is_linear_and_weighted_by_counts():
    b = Bound(measured=90, unmeasured_delisted=10, unmeasured_unknown=0,
              delisting_return=-0.50)

    assert b.weight == pytest.approx(0.9)
    assert b.apply(0.10) == pytest.approx(0.9 * 0.10 + 0.1 * -0.50)


def test_a_fully_measured_population_is_unchanged():
    b = bound(_buckets(listed=(100, 100), delisted=(50, 50)))

    assert not b.binding
    assert b.apply(0.07) == pytest.approx(0.07)
    assert "no unmeasured delisted events" in b.describe()


def test_the_bound_always_moves_a_positive_mean_downward():
    b = bound(REAL)

    for m in (0.001, 0.02, 0.05, 0.20):
        assert b.apply(m) < m


def test_the_default_is_the_nasdaq_figure_not_the_nyse_one():
    """-30% is the NYSE/AMEX number. This universe is microcaps, overwhelmingly
    NASDAQ and OTC, where Shumway & Warther found the bias 4.7x larger."""
    assert DELISTING_RETURN == -0.55


def test_the_delisting_return_is_overridable_for_sensitivity():
    lenient = bound(REAL, delisting_return=-0.30)
    strict = bound(REAL, delisting_return=-0.55)

    assert lenient.apply(0.0346) > strict.apply(0.0346)


# --- it stays a bound -------------------------------------------------------

def test_unknown_classification_is_excluded_from_the_bound_but_reported():
    """Folding "unknown" in would overstate a figure that is already the
    pessimistic end of the range -- but omitting it silently would understate
    the true worst case."""
    b = bound(REAL)

    assert b.unmeasured_unknown == 570
    assert b.weight == pytest.approx(33_809 / (33_809 + 1_431))
    assert "excluded from the bound" in b.describe()
    assert "beyond this one" in b.describe()


def test_the_description_says_plainly_that_nothing_is_adjusted():
    assert "BOUND, not a correction" in bound(REAL).describe()


def test_the_description_cites_its_source():
    """A number this consequential must carry its provenance to the reader."""
    assert "Shumway & Warther 1999" in bound(REAL).describe()


# --- degenerate inputs ------------------------------------------------------

def test_an_empty_breakdown_is_not_a_division_by_zero():
    b = bound({})

    assert b.weight == 1.0
    assert not b.binding
    assert b.apply(0.05) == pytest.approx(0.05)


def test_a_population_with_no_delisted_bucket_is_not_binding():
    b = bound(_buckets(listed=(100, 80)))

    assert not b.binding
    assert b.apply(0.05) == pytest.approx(0.05)


def test_labelled_exceeding_seen_cannot_produce_a_negative_count():
    b = bound(_buckets(delisted=(10, 12)))

    assert b.unmeasured_delisted == 0
