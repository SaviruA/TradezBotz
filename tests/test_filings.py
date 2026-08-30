"""Tests for 8-K and 424B parsing.

The point-in-time property carries the weight, as it does everywhere else here:
a filing is knowable when the SEC accepted it, never when the event inside it
happened. Getting that wrong would let a backtest trade on a material agreement
days before anyone could have read about it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradezbotz.research.filings import (
    IMMEDIATE_DILUTION,
    ITEM_NAMES,
    ITEMS_DISTRESS,
    ITEMS_MANAGEMENT,
    ITEMS_RESULTS,
    ITEMS_ROUTINE,
    LARGE_ISSUER_HEAVY,
    is_distress_8k,
    is_immediate_dilution,
    is_informative_8k,
    is_management_change,
    is_results_8k,
    parse_424b,
    parse_8k,
)

HEADER = """<SEC-DOCUMENT>0001213900-26-093981.txt : 20260826
<SEC-HEADER>0001213900-26-093981.hdr.sgml : 20260826
<ACCEPTANCE-DATETIME>20260826061700
ACCESSION NUMBER:		0001213900-26-093981
CONFORMED SUBMISSION TYPE:	8-K
	COMPANY DATA:
		COMPANY CONFORMED NAME:			TEST CORP
		CENTRAL INDEX KEY:			0001083220
	FILING VALUES:
		FORM TYPE:		8-K
{items}
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<TEXT>
{body}
</TEXT>
</DOCUMENT>
"""


def filing(items_header="", body="Item 2.02 Results of Operations\nItem 9.01 Exhibits"):
    return HEADER.format(items=items_header, body=body)


# --- 8-K item extraction --------------------------------------------------------

def test_items_are_extracted_from_the_body():
    ev = parse_8k(filing())

    assert ev is not None
    assert ev.items == ("2.02", "9.01")
    assert ev.cik == "1083220", "leading zeros stripped"
    assert ev.company == "TEST CORP"


def test_administrative_items_are_separated_from_informative_ones():
    """9.01 accompanies nearly every 8-K. Counting it as content would make
    'an 8-K was filed' look informative when it is not."""
    ev = parse_8k(filing(body="Item 5.03 Bylaws\nItem 9.01 Exhibits"))

    assert set(ev.items) == {"5.03", "9.01"}
    assert ev.informative_items == ()
    assert is_informative_8k(ev.to_event().payload) is False


def test_a_real_item_survives_alongside_routine_ones():
    ev = parse_8k(filing(body="Item 1.01 Agreement\nItem 5.03 Bylaws\nItem 9.01 Exhibits"))

    assert ev.informative_items == ("1.01",)
    assert is_informative_8k(ev.to_event().payload) is True


def test_invented_item_codes_are_rejected():
    """Exhibits routinely carry strings like 'Item 3.05' from other documents.
    Without the whitelist those become phantom events."""
    ev = parse_8k(filing(body="Item 2.02 Results\nItem 3.05 Not A Real Code"))

    assert ev.items == ("2.02",)
    assert "3.05" not in ITEM_NAMES


def test_item_mismatch_is_recorded_not_resolved():
    """Header says two items, body shows one. Guessing which is right would be
    worse than saying they disagree."""
    header = "ITEM INFORMATION:\tResults\nITEM INFORMATION:\tOther Events"

    ev = parse_8k(filing(items_header=header, body="Item 2.02 Results only"))

    assert ev.items == ("2.02",)
    assert ev.item_mismatch is True


def test_matching_counts_are_not_flagged():
    header = "ITEM INFORMATION:\tResults\nITEM INFORMATION:\tExhibits"

    ev = parse_8k(filing(items_header=header))

    assert ev.item_mismatch is False


def test_a_filing_with_no_items_is_skipped():
    assert parse_8k(filing(body="No items here at all")) is None


def test_a_filing_with_no_acceptance_stamp_is_skipped():
    """Without an acceptance time there is no defensible observed_at, and
    inventing one would break the point-in-time guarantee."""
    raw = filing().replace("<ACCEPTANCE-DATETIME>20260826061700", "")

    assert parse_8k(raw) is None


# --- point in time ---------------------------------------------------------------

def test_observed_at_comes_from_the_acceptance_stamp():
    ev = parse_8k(filing())

    assert ev.observed_at.tzinfo is not None, "must be timezone-aware"
    assert ev.observed_at.year == 2026 and ev.observed_at.month == 8


def test_event_never_occurs_after_it_was_observed():
    """The invariant the event store enforces."""
    ev = parse_8k(filing())
    event = ev.to_event()

    assert event.occurred_at <= event.observed_at


# --- selectors -------------------------------------------------------------------

def test_results_selector():
    results = parse_8k(filing(body="Item 2.02 Results")).to_event().payload
    other = parse_8k(filing(body="Item 8.01 Other")).to_event().payload

    assert is_results_8k(results) is True
    assert is_results_8k(other) is False


def test_management_change_selector():
    p = parse_8k(filing(body="Item 5.02 Departure of Officer")).to_event().payload

    assert is_management_change(p) is True


def test_distress_selector_covers_delisting_and_restatement():
    for code in ("3.01", "4.02", "1.03", "2.06"):
        p = parse_8k(filing(body=f"Item {code} Something")).to_event().payload
        assert is_distress_8k(p) is True, code


def test_selectors_are_safe_on_an_empty_payload():
    assert is_results_8k({}) is False
    assert is_distress_8k({}) is False
    assert is_informative_8k({}) is False
    assert is_immediate_dilution({}) is False


# --- 424B offerings ---------------------------------------------------------------

def offering_filing(form="424B4"):
    return HEADER.format(items="", body="prospectus supplement").replace(
        "CONFORMED SUBMISSION TYPE:	8-K", f"CONFORMED SUBMISSION TYPE:	{form}"
    )


def test_424b4_is_immediate_dilution():
    ev = parse_424b(offering_filing("424B4"), "424B4")

    assert ev.form_type == "424B4"
    assert ev.immediate is True
    assert is_immediate_dilution(ev.to_event().payload) is True


def test_424b2_is_not_flagged_as_immediate():
    """Measured on one day, 447 of 501 424B filings were 424B2 and were
    overwhelmingly Bank of America structured notes. Flagging those as dilution
    would bury the two real 424B4s and libel the largest bank in the country."""
    ev = parse_424b(offering_filing("424B2"), "424B2")

    assert ev.immediate is False
    assert "424B2" in LARGE_ISSUER_HEAVY
    assert "424B2" not in IMMEDIATE_DILUTION


def test_424b3_is_ingested_but_not_immediate():
    """ATM programmes are a standing marginal seller, not a single hit."""
    ev = parse_424b(offering_filing("424B3"), "424B3")

    assert ev is not None
    assert ev.immediate is False
    assert "ATM" in ev.to_event().payload["description"]


def test_offering_event_shape():
    event = parse_424b(offering_filing("424B4"), "424B4").to_event()

    assert event.kind == "offering"
    assert event.source == "sec_424b"
    assert event.payload["form_type"] == "424B4"


def test_item_groups_do_not_overlap_with_routine():
    """A code cannot be both a signal and administrative noise."""
    for group in (ITEMS_RESULTS, ITEMS_MANAGEMENT, ITEMS_DISTRESS):
        assert not (group & ITEMS_ROUTINE)
