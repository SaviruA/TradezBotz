"""Tests for XBRL fundamentals.

`visible()` carries the weight. Without point-in-time filtering every ratio is
contaminated by figures that did not exist at the decision point, including
restatements of periods long past -- and nothing in the number would reveal it.
"""

from __future__ import annotations

from datetime import date

import pytest

from tradezbotz.research.fundamentals import (
    Fact,
    Snapshot,
    extract,
    latest,
    trailing_twelve_months,
    visible,
)


def fact(val, end, filed, start=None, form="10-Q"):
    return Fact(value=val, start=date.fromisoformat(start) if start else None,
                end=date.fromisoformat(end), filed=date.fromisoformat(filed),
                form=form)


QUARTERS = [
    fact(100, "2024-03-31", "2024-05-01", "2024-01-01"),
    fact(110, "2024-06-30", "2024-08-01", "2024-04-01"),
    fact(120, "2024-09-30", "2024-11-01", "2024-07-01"),
    fact(130, "2024-12-31", "2025-02-01", "2024-10-01"),
]


# --- point in time -----------------------------------------------------------------

def test_facts_filed_after_the_date_are_invisible():
    seen = visible(QUARTERS, date(2024, 8, 15))

    assert len(seen) == 2, "only the two filed by mid-August"
    assert all(f.filed <= date(2024, 8, 15) for f in seen)


def test_a_restatement_does_not_leak_backwards():
    """The same period refiled later must not be visible before it was filed."""
    original = fact(100, "2024-03-31", "2024-05-01", "2024-01-01")
    restated = fact(80, "2024-03-31", "2025-02-01", "2024-01-01", form="10-K/A")

    early = latest([original, restated], date(2024, 6, 1))
    late = latest([original, restated], date(2025, 6, 1))

    assert early.value == 100, "the restatement had not happened yet"
    assert late.value == 80, "once filed, the newer version wins"


def test_latest_breaks_ties_on_filing_date():
    a = fact(1, "2024-03-31", "2024-05-01")
    b = fact(2, "2024-03-31", "2024-09-01")

    assert latest([a, b], date(2025, 1, 1)).value == 2


def test_nothing_visible_yet_is_none():
    assert latest(QUARTERS, date(2020, 1, 1)) is None


# --- trailing twelve months ----------------------------------------------------------

def test_ttm_sums_four_quarters():
    assert trailing_twelve_months(QUARTERS, date(2025, 3, 1)) == 460


def test_ttm_needs_four_quarters():
    assert trailing_twelve_months(QUARTERS[:3], date(2025, 3, 1)) is None


def test_ttm_respects_what_was_filed():
    """At mid-2024 only two quarters had been filed, so there is no TTM."""
    assert trailing_twelve_months(QUARTERS, date(2024, 8, 15)) is None


def test_ttm_prefers_an_annual_figure():
    annual = fact(500, "2024-12-31", "2025-02-15", "2024-01-01", form="10-K")

    assert trailing_twelve_months(QUARTERS + [annual], date(2025, 3, 1)) == 500


def test_stale_data_does_not_count_as_ttm():
    """Four quarters ending two years ago are not a trailing twelve months."""
    assert trailing_twelve_months(QUARTERS, date(2027, 6, 1)) is None


def test_a_restated_quarter_is_not_double_counted():
    restated = fact(200, "2024-12-31", "2025-03-01", "2024-10-01")

    total = trailing_twelve_months(QUARTERS + [restated], date(2025, 6, 1))

    assert total == 100 + 110 + 120 + 200, "the newer filing replaces, not adds"


# --- concept extraction ---------------------------------------------------------------

def facts_doc(concept, rows):
    return {"facts": {"us-gaap": {concept: {"units": {"USD": rows}}}}}


def test_extract_finds_the_first_present_concept():
    doc = facts_doc("Revenues", [
        {"val": 5, "end": "2024-03-31", "filed": "2024-05-01", "form": "10-Q"}])

    out = extract(doc, ("RevenueFromContractWithCustomerExcludingAssessedTax",
                        "Revenues"))

    assert len(out) == 1 and out[0].value == 5


def test_extract_does_not_merge_two_concepts():
    """Two tags for the same quantity are not guaranteed to agree; concatenating
    them would produce a series that never appeared on any filing."""
    doc = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"val": 9, "end": "2024-03-31", "filed": "2024-05-01"}]}},
        "Revenues": {"units": {"USD": [
            {"val": 5, "end": "2024-03-31", "filed": "2024-05-01"}]}},
    }}}

    out = extract(doc, ("RevenueFromContractWithCustomerExcludingAssessedTax",
                        "Revenues"))

    assert [f.value for f in out] == [9], "first concept only"


def test_extract_skips_malformed_rows():
    doc = facts_doc("Revenues", [
        {"val": 5, "end": "2024-03-31", "filed": "2024-05-01"},
        {"val": "oops", "end": "2024-06-30", "filed": "2024-08-01"},
        {"end": "2024-09-30", "filed": "2024-11-01"},
    ])

    assert len(extract(doc, ("Revenues",))) == 1


def test_extract_of_an_absent_concept_is_empty():
    assert extract({"facts": {"us-gaap": {}}}, ("Revenues",)) == []


# --- ratios ------------------------------------------------------------------------------

def snap(**kw):
    base = dict(cik="1", as_of=date(2025, 1, 1), revenue_ttm=1000.0,
                gross_profit_ttm=400.0, net_income_ttm=-50.0,
                operating_income_ttm=-30.0, shares_outstanding=100.0,
                revenue_prior_ttm=800.0)
    base.update(kw)
    return Snapshot(**base)


def test_price_to_sales():
    assert snap().price_to_sales(20.0) == pytest.approx(2.0)


def test_price_to_sales_works_when_earnings_are_negative():
    """The entire argument for P/S: 74% of small caps have negative earnings,
    so P/E is undefined for most of our universe."""
    s = snap(net_income_ttm=-500.0)

    assert s.profitable is False
    assert s.price_to_sales(20.0) is not None


def test_gross_margin():
    assert snap().gross_margin == pytest.approx(0.4)


def test_revenue_growth():
    assert snap().revenue_growth == pytest.approx(0.25)


def test_value_growth_score():
    # P/S 2.0 over 25% growth
    assert snap().value_growth_score(20.0) == pytest.approx(2.0 / 25.0)


def test_value_growth_score_refuses_negative_growth():
    """Dividing by negative growth would rank shrinking companies as cheapest."""
    assert snap(revenue_prior_ttm=2000.0).value_growth_score(20.0) is None


def test_ratios_are_none_without_the_inputs():
    empty = snap(revenue_ttm=None, gross_profit_ttm=None,
                 shares_outstanding=None, revenue_prior_ttm=None)

    assert empty.price_to_sales(20.0) is None
    assert empty.gross_margin is None
    assert empty.revenue_growth is None
    assert empty.value_growth_score(20.0) is None


def test_zero_revenue_does_not_divide_by_zero():
    assert snap(revenue_ttm=0.0).price_to_sales(20.0) is None
    assert snap(revenue_ttm=0.0).gross_margin is None
