"""Tests for the five standard valuation multiples.

Most of these guard sign and definedness rather than arithmetic. The arithmetic
is one division; the ways these go wrong quietly are (a) returning a negative
multiple that sorts as "cheapest", (b) treating an untagged balance sheet as a
debt-free one, and (c) getting the capex sign backwards so a cash-burning
company reads as a cash generator.
"""

from __future__ import annotations

from datetime import date

import pytest

from tradezbotz.research.fundamentals import (
    BAND_MEGA,
    BAND_MICRO,
    COVERAGE_BY_BAND,
    FundamentalsError,
    Snapshot,
    guard_single_band,
    size_band,
)

AS_OF = date(2025, 1, 1)


def snap(**kw) -> Snapshot:
    base = dict(
        cik="1", as_of=AS_OF, revenue_ttm=1_000_000_000.0,
        gross_profit_ttm=400_000_000.0, net_income_ttm=100_000_000.0,
        operating_income_ttm=150_000_000.0, shares_outstanding=100_000_000.0,
        depreciation_amortisation_ttm=50_000_000.0,
        operating_cash_flow_ttm=200_000_000.0, capex_ttm=50_000_000.0,
        cash=100_000_000.0, debt=300_000_000.0, debt_reported=True,
    )
    base.update(kw)
    return Snapshot(**base)


PRICE = 50.0   # 100M shares -> $5B market cap


# --- arithmetic -------------------------------------------------------------

def test_the_five_multiples_compute():
    s = snap()

    assert s.market_cap(PRICE) == pytest.approx(5_000_000_000.0)
    assert s.price_to_earnings(PRICE) == pytest.approx(50.0)
    assert s.price_to_sales(PRICE) == pytest.approx(5.0)
    assert s.ebitda_ttm == pytest.approx(200_000_000.0)
    assert s.free_cash_flow_ttm == pytest.approx(150_000_000.0)
    assert s.price_to_free_cash_flow(PRICE) == pytest.approx(33.333, rel=1e-3)
    assert s.enterprise_value(PRICE) == pytest.approx(5_200_000_000.0)
    assert s.ev_to_ebitda(PRICE) == pytest.approx(26.0)


def test_capex_is_subtracted_whichever_sign_it_arrives_with():
    """XBRL reports capex as a positive outflow, but filers are not perfectly
    consistent. Adding it would turn a cash burner into a cash generator."""
    positive = snap(capex_ttm=50_000_000.0)
    negative = snap(capex_ttm=-50_000_000.0)

    assert positive.free_cash_flow_ttm == negative.free_cash_flow_ttm
    assert positive.free_cash_flow_ttm == pytest.approx(150_000_000.0)


# --- definedness, which is the whole argument -------------------------------

def test_negative_earnings_give_no_pe_rather_than_a_negative_one():
    """A negative P/E sorts below every cheap profitable company, so a naive
    'lowest P/E' screen fills with loss-makers. That is the failure mode, not
    the number itself."""
    assert snap(net_income_ttm=-50_000_000.0).price_to_earnings(PRICE) is None


def test_negative_free_cash_flow_gives_no_multiple():
    burning = snap(operating_cash_flow_ttm=10_000_000.0, capex_ttm=90_000_000.0)

    assert burning.free_cash_flow_ttm == pytest.approx(-80_000_000.0)
    assert burning.price_to_free_cash_flow(PRICE) is None


def test_negative_ebitda_gives_no_multiple():
    assert snap(operating_income_ttm=-300_000_000.0).ev_to_ebitda(PRICE) is None


def test_missing_da_blocks_ebitda_and_everything_downstream():
    """D&A is the binding constraint: 45.8% of small filers tag it. EBITDA is
    non-GAAP, so it cannot be recovered from anything else."""
    s = snap(depreciation_amortisation_ttm=None)

    assert s.ebitda_ttm is None
    assert s.ev_to_ebitda(PRICE) is None


def test_missing_capex_blocks_fcf():
    s = snap(capex_ttm=None)

    assert s.free_cash_flow_ttm is None
    assert s.price_to_free_cash_flow(PRICE) is None


# --- the untagged-balance-sheet trap ----------------------------------------

def test_an_untagged_balance_sheet_is_not_a_debt_free_one():
    """The trap XBRL sets: a filer with no debt tag may have no debt or may
    simply not have tagged it, and XBRL does not distinguish them. Treating
    absent as zero understates enterprise value for exactly the filers whose
    tagging is weakest, producing a screen that rewards poor disclosure."""
    untagged = snap(debt=None, debt_reported=False)

    assert untagged.enterprise_value(PRICE) is None
    assert untagged.ev_to_ebitda(PRICE) is None


def test_a_genuinely_debt_free_company_still_gets_an_enterprise_value():
    """The other half of the same distinction: reported zero is a measurement."""
    debt_free = snap(debt=0.0, debt_reported=True)

    assert debt_free.enterprise_value(PRICE) == pytest.approx(4_900_000_000.0)


def test_missing_cash_blocks_enterprise_value():
    assert snap(cash=None).enterprise_value(PRICE) is None


# --- size bands -------------------------------------------------------------

@pytest.mark.parametrize("cap,expected", [
    (50_000_000, "micro"),
    (1_000_000_000, "small"),
    (5_000_000_000, "mid"),
    (50_000_000_000, "large"),
    (3_000_000_000_000, "mega"),
])
def test_size_bands(cap, expected):
    assert size_band(cap) == expected


def test_size_band_is_none_without_a_market_cap():
    assert size_band(None) is None
    assert size_band(0) is None
    assert snap(shares_outstanding=None).market_cap(PRICE) is None


def test_mixing_size_bands_is_refused():
    """Costs differ by an order of magnitude across this range, so a pooled
    result is an average of two economies weighted by tag availability."""
    with pytest.raises(FundamentalsError) as exc:
        guard_single_band([BAND_MICRO, BAND_MEGA])

    assert "mixes size bands" in str(exc.value)


def test_one_band_with_gaps_is_allowed():
    """None means 'no market cap known', not 'a different band'. Refusing on it
    would block every study with an untagged share count."""
    guard_single_band([BAND_MEGA, None, BAND_MEGA])


# --- the recorded measurement -----------------------------------------------

def test_coverage_table_records_that_size_does_not_fix_ev_ebitda():
    """The finding that changed the plan: moving up the size distribution fixes
    P/E dramatically and does not fix EV/EBITDA."""
    small = COVERAGE_BY_BAND["revenue <$100M"]
    largest = COVERAGE_BY_BAND["over $10B"]

    assert largest["pe"] / small["pe"] > 3.0, "P/E coverage more than triples"
    assert largest["ev_ebitda"] < 0.5, "EV/EBITDA stays a minority even at the top"


def test_coverage_peaks_below_the_largest_band():
    """'Bigger is better' is false past mid-cap: three of four metrics peak in
    the $1B-$10B band, because the largest filers include banks and insurers
    for whom operating income and capex do not mean what the formula assumes."""
    mid = COVERAGE_BY_BAND["$1B-$10B"]
    largest = COVERAGE_BY_BAND["over $10B"]

    for metric in ("fcf", "ebitda", "ev_ebitda"):
        assert mid[metric] > largest[metric], f"{metric} should peak at $1B-$10B"
