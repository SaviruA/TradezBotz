"""Tests for the SEC quarterly bulk loader."""

from __future__ import annotations

import io
import zipfile
from datetime import date

import pytest

from tradezbotz.research.bulk import (
    events_from_archive,
    parse_sec_date,
    quarters_between,
)
from tradezbotz.research.edgar import ET_TZ

SUBMISSION = (
    "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERTRADINGSYMBOL\n"
    "0001-25-000001\t31-MAR-2025\t4\t0000320193\tAAPL\n"
    "0001-25-000002\t02-APR-2025\t4\t0000789019\tMSFT\n"
    "0001-25-000003\t31-MAR-2025\t3\t0000320193\tAAPL\n"      # Form 3: excluded
    "0001-25-000004\t31-MAR-2025\t4\t0000111111\t\n"          # no ticker: excluded
)

OWNERS = (
    "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
    "0001-25-000001\t0001111\tDOE JANE\tOfficer,Director\tChief Executive Officer\n"
    "0001-25-000002\t0002222\tROE JOHN\tTenPercentOwner\t\n"
)

TRANS = (
    "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES"
    "\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
    "0001-25-000001\t900001\t28-MAR-2025\tP\t5000\t150.25\tA\n"
    "0001-25-000001\t900002\t28-MAR-2025\tP\t5000\t150.25\tA\n"   # identical line
    "0001-25-000002\t900003\t01-APR-2025\tS\t200\t410.00\tD\n"
    "0001-25-000003\t900004\t28-MAR-2025\tP\t100\t10.00\tA\n"     # Form 3: excluded
)


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "q.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("SUBMISSION.tsv", SUBMISSION)
        zf.writestr("REPORTINGOWNER.tsv", OWNERS)
        zf.writestr("NONDERIV_TRANS.tsv", TRANS)
    return path


def test_parses_form4_transactions(archive):
    events = list(events_from_archive(archive))

    assert len(events) == 3
    assert {e.symbol for e in events} == {"AAPL", "MSFT"}


def test_form3_and_form5_are_excluded(archive):
    accessions = {e.payload["accession"] for e in events_from_archive(archive)}

    assert "0001-25-000003" not in accessions


def test_identical_lines_stay_distinct_via_sec_surrogate_key(archive):
    """The SEC's own NONDERIV_TRANS_SK separates lines that are identical on
    every visible field, so the bulk path needs no document-order index."""
    aapl = [e for e in events_from_archive(archive) if e.symbol == "AAPL"]

    assert len(aapl) == 2
    assert len({e.external_id for e in aapl}) == 2


def test_observed_at_forces_next_session_entry(archive):
    """Bulk data has no acceptance time. Stamping the 22:00 ET cutoff makes the
    labeller take the next session's open -- later than reality at worst, which
    understates returns rather than inventing them."""
    e = next(iter(events_from_archive(archive)))
    et = e.observed_at.astimezone(ET_TZ)

    assert (et.hour, et.minute) == (22, 0)


def test_ordering_invariant_holds(archive):
    for e in events_from_archive(archive):
        assert e.occurred_at <= e.observed_at


def test_relationship_flags_are_parsed(archive):
    by_symbol = {e.symbol: e.payload for e in events_from_archive(archive)}

    assert by_symbol["AAPL"]["is_officer"] is True
    assert by_symbol["AAPL"]["is_director"] is True
    assert by_symbol["AAPL"]["is_ten_percent"] is False
    assert by_symbol["MSFT"]["is_ten_percent"] is True
    assert by_symbol["MSFT"]["is_officer"] is False


def test_notional_is_computed(archive):
    aapl = next(e for e in events_from_archive(archive) if e.symbol == "AAPL")

    assert aapl.payload["notional"] == pytest.approx(5000 * 150.25)


def test_precision_is_recorded_as_provenance(archive):
    """Downstream must be able to tell date-only rows from timed ones."""
    assert all(e.payload["precision"] == "date_only" for e in events_from_archive(archive))


def test_before_cutoff_excludes_the_labelling_window(archive):
    """Bulk and per-filing paths mint different external_id formats, so an
    overlap would store the same transaction twice instead of deduplicating."""
    events = list(events_from_archive(archive, before=date(2025, 4, 1)))

    assert {e.symbol for e in events} == {"AAPL"}, "the 02-APR filing is excluded"


# --- helpers -----------------------------------------------------------------

def test_parse_sec_date():
    assert parse_sec_date("31-MAR-2025") == date(2025, 3, 31)
    assert parse_sec_date("01-JAN-2020") == date(2020, 1, 1)
    assert parse_sec_date("") is None
    assert parse_sec_date("2025-03-31") is None
    assert parse_sec_date("31-XXX-2025") is None


def test_quarters_between_covers_the_range():
    qs = quarters_between(date(2024, 2, 1), date(2024, 11, 1))

    assert qs == [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]


def test_quarters_between_crosses_year_boundaries():
    qs = quarters_between(date(2023, 11, 1), date(2024, 2, 1))

    assert qs == [(2023, 4), (2024, 1)]


def test_five_years_is_about_twenty_downloads():
    qs = quarters_between(date(2021, 8, 1), date(2026, 8, 1))

    assert 19 <= len(qs) <= 22


def test_quarters_between_is_oldest_first():
    """cmd_ingest_bulk reverses this. Pinning the direction here so a future
    change to either side cannot silently reintroduce the bug where a
    time-boxed run spends its whole budget on deep baselines and never reaches
    the labelling window."""
    qs = quarters_between(date(2024, 1, 1), date(2025, 6, 1))

    assert qs[0] == (2024, 1)
    assert qs[-1] == (2025, 2)
    assert qs == sorted(qs)
