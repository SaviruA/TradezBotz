"""Tests for bulk acceptance-time retrieval and precision upgrade."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from tradezbotz.research.edgar import ET_TZ
from tradezbotz.research.eventstore import Event
from tradezbotz.research.submissions import (
    SubmissionsCache,
    SubmissionsClient,
    parse_acceptance,
    upgrade_precision,
)


# --- timezone semantics ------------------------------------------------------

def test_acceptance_json_is_parsed_as_utc_not_eastern():
    """Verified against live data: the submissions API reports UTC while the
    filing header reports ET. Both describe the same instant.

        submissions JSON : 2026-08-27T22:30:30.000Z
        .txt header      : 20260827183030  (18:30:30 ET)

    Reading the JSON as Eastern would shift everything by 4-5 hours and roll
    post-close filings past the 22:00 ET cutoff into the wrong session.
    """
    parsed = parse_acceptance("2026-08-27T22:30:30.000Z")

    assert parsed == datetime(2026, 8, 27, 22, 30, 30, tzinfo=UTC)
    assert parsed.astimezone(ET_TZ).hour == 18


def test_malformed_acceptance_returns_none():
    assert parse_acceptance("") is None
    assert parse_acceptance("not-a-date") is None


# --- cache -------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    c = SubmissionsCache(tmp_path / "subs.db")
    yield c
    c.close()


def test_cache_round_trips(cache):
    t = datetime(2026, 8, 27, 22, 30, 30, tzinfo=UTC)
    cache.put("320193", {"0001-26-000001": t})

    assert cache.get("0001-26-000001") == t
    assert cache.has_cik("320193") is True
    assert cache.count() == 1


def test_cache_reports_unknown_accession(cache):
    assert cache.get("nope") is None
    assert cache.has_cik("999") is False


# --- client ------------------------------------------------------------------

class FakeEdgar:
    def __init__(self, payload):
        self.payload = payload
        self.urls: list[str] = []

    def _get(self, url):
        self.urls.append(url)
        return json.dumps(self.payload)


PAYLOAD = {
    "filings": {
        "recent": {
            "accessionNumber": ["0001-26-000001", "0001-26-000002", "0001-26-000003"],
            "acceptanceDateTime": [
                "2026-08-27T22:30:30.000Z",
                "2026-08-20T22:30:16.000Z",
                "",                       # missing: skipped, not guessed
            ],
            "form": ["4", "4", "4"],
        }
    }
}


def test_extracts_acceptance_times():
    c = SubmissionsClient(FakeEdgar(PAYLOAD))

    times = c.acceptance_times("320193")

    assert len(times) == 2, "the blank timestamp is skipped"
    assert times["0001-26-000001"].astimezone(ET_TZ).hour == 18


def test_cik_is_zero_padded_in_the_url():
    edgar = FakeEdgar(PAYLOAD)

    SubmissionsClient(edgar).acceptance_times("320193")

    assert "CIK0000320193.json" in edgar.urls[0]


def test_load_ciks_skips_already_cached(cache):
    edgar = FakeEdgar(PAYLOAD)
    c = SubmissionsClient(edgar, cache)

    assert c.load_ciks(["320193", "320193", "789019"]) == 2, "dedupes within the call"
    assert c.load_ciks(["320193", "789019"]) == 0, "second pass costs nothing"


def test_bad_json_yields_no_times():
    class Broken:
        def _get(self, url):
            return "<html>not json</html>"

    assert SubmissionsClient(Broken()).acceptance_times("1") == {}


# --- precision upgrade -------------------------------------------------------

def bulk_event(accession="0001-26-000001", filing_day=date(2026, 8, 27)):
    """A date-only event as the bulk loader produces it: stamped 22:00 ET."""
    observed = datetime.combine(
        filing_day, datetime.min.time().replace(hour=22), tzinfo=ET_TZ
    )
    return Event(
        source="sec_form4",
        external_id=f"{accession}:sk1",
        kind="insider_transaction",
        symbol="AAPL",
        observed_at=observed,
        occurred_at=observed,
        payload={
            "accession": accession,
            "transaction_date": "2026-08-26",
            "precision": "date_only",
        },
    )


def test_upgrade_replaces_the_conservative_stamp(cache):
    # Accepted 18:30 ET, comfortably before the 22:00 cutoff.
    cache.put("320193", {"0001-26-000001": datetime(2026, 8, 27, 22, 30, 30, tzinfo=UTC)})

    out = list(upgrade_precision([bulk_event()], cache))[0]

    assert out.payload["precision"] == "timed"
    assert out.observed_at.astimezone(ET_TZ).hour == 18
    assert out.observed_at < bulk_event().observed_at, "real time beats the placeholder"


def test_upgrade_preserves_identity_so_storage_stays_idempotent(cache):
    cache.put("320193", {"0001-26-000001": datetime(2026, 8, 27, 22, 30, 30, tzinfo=UTC)})
    original = bulk_event()

    out = list(upgrade_precision([original], cache))[0]

    assert out.external_id == original.external_id
    assert out.source == original.source


def test_upgrade_keeps_the_ordering_invariant(cache):
    cache.put("320193", {"0001-26-000001": datetime(2026, 8, 27, 22, 30, 30, tzinfo=UTC)})

    out = list(upgrade_precision([bulk_event()], cache))[0]

    assert out.occurred_at <= out.observed_at


def test_uncached_event_passes_through_unchanged(cache):
    """A missing timestamp costs precision, never correctness -- the event keeps
    its conservative 22:00 stamp rather than being dropped or guessed."""
    original = bulk_event(accession="0009-26-999999")

    out = list(upgrade_precision([original], cache))[0]

    assert out.observed_at == original.observed_at
    assert out.payload["precision"] == "date_only"


def test_late_acceptance_still_rolls_to_next_session(cache):
    """23:30 ET is past the Form 4 cutoff, so it disseminates next morning --
    the upgrade must not bypass that rule."""
    cache.put("320193", {"0001-26-000001": datetime(2026, 8, 28, 3, 30, tzinfo=UTC)})

    out = list(upgrade_precision([bulk_event()], cache))[0]
    et = out.observed_at.astimezone(ET_TZ)

    assert et.date() == date(2026, 8, 28)
    assert et.hour == 6
