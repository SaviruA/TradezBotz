"""Exact filing timestamps in bulk, from EDGAR's per-issuer submissions API.

The quarterly Form 345 archives carry complete transaction data but only a
filing *date*. Our entry rule needs the dissemination *time*, and until now the
only way to get it was fetching each filing individually -- roughly 212,000
requests for a two-year window.

`data.sec.gov/submissions/CIK##########.json` supplies `acceptanceDateTime` for
every filing a CIK has made. Two properties make this cheap:

  * there are only ~4,200 distinct issuer CIKs per quarter, and
  * each `recent` block holds ~1,000 filings, reaching back 5-11 years even for
    heavy filers like Apple, NVIDIA and Tesla

So one request per issuer covers the whole window: ~7,000 requests instead of
~212,000, about fifteen minutes instead of seven and a half hours, for the same
precision.

**Timezone, verified against live data.** The two sources disagree in
representation and agree in fact:

    submissions JSON : 2026-08-27T22:30:30.000Z   (UTC)
    .txt header      : 20260827183030             (18:30:30 ET)

Reading the JSON as Eastern would shift every timestamp by four or five hours
and push post-close filings past the 22:00 ET cutoff, rolling them to the wrong
session. The `Z` is real: parse as UTC and let `_disseminated_at` convert.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .edgar import EdgarClient, _disseminated_at, _occurred_at
from .eventstore import Event

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS acceptance (
    accession   TEXT PRIMARY KEY,
    accepted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fetched_ciks (
    cik        TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    filings    INTEGER NOT NULL
);
"""


class SubmissionsCache:
    """Persists accession -> acceptance time so reruns cost nothing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(CACHE_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def has_cik(self, cik: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM fetched_ciks WHERE cik = ?", (cik,)
            ).fetchone()
            is not None
        )

    def put(self, cik: str, times: dict[str, datetime]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO acceptance VALUES (?,?)",
            [(acc, t.isoformat()) for acc, t in times.items()],
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO fetched_ciks VALUES (?,?,?)",
            (cik, datetime.now(timezone.utc).isoformat(), len(times)),
        )
        self._conn.commit()

    def get(self, accession: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT accepted_at FROM acceptance WHERE accession = ?", (accession,)
        ).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM acceptance").fetchone()[0])


def parse_acceptance(raw: str) -> datetime | None:
    """Parse `2026-08-27T22:30:30.000Z` as UTC. See the module note on timezones."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


class SubmissionsClient:
    def __init__(self, edgar: EdgarClient, cache: SubmissionsCache | None = None) -> None:
        self.edgar = edgar          # reused for its rate limiter and User-Agent
        self.cache = cache

    def acceptance_times(self, cik: str) -> dict[str, datetime]:
        """All accession -> acceptance time pairs for one CIK.

        Only the `recent` block is read. It reaches back years for even the
        busiest filers, so paging into the older archives would spend requests
        on filings far outside any window we can label.
        """
        cik = str(cik).lstrip("0") or "0"
        padded = cik.zfill(10)
        import json

        raw = self.edgar._get(SUBMISSIONS_URL.format(cik=padded))
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        recent = (body.get("filings") or {}).get("recent") or {}
        accessions = recent.get("accessionNumber") or []
        accepted = recent.get("acceptanceDateTime") or []
        out: dict[str, datetime] = {}
        for acc, ts in zip(accessions, accepted):
            parsed = parse_acceptance(ts)
            if parsed:
                out[acc] = parsed
        return out

    def load_ciks(self, ciks: Iterable[str], *, on_progress=None) -> int:
        """Warm the cache for a set of issuer CIKs. Returns CIKs fetched."""
        fetched = 0
        seen: set[str] = set()
        for raw_cik in ciks:
            cik = str(raw_cik or "").lstrip("0")
            if not cik or cik in seen:
                continue
            seen.add(cik)
            if self.cache and self.cache.has_cik(cik):
                continue
            try:
                times = self.acceptance_times(cik)
            except Exception:  # noqa: BLE001 - one dead CIK must not stop the run
                times = {}
            if self.cache:
                self.cache.put(cik, times)
            fetched += 1
            if on_progress:
                on_progress(cik, fetched, len(times))
        return fetched


def upgrade_precision(
    events: Iterable[Event], cache: SubmissionsCache
) -> Iterator[Event]:
    """Replace date-only `observed_at` with the real dissemination time.

    Events whose accession is not in the cache pass through unchanged, keeping
    their conservative 22:00 ET stamp. That is the safe direction: a missing
    timestamp costs precision, never correctness.
    """
    for event in events:
        accession = event.payload.get("accession")
        accepted = cache.get(accession) if accession else None
        if accepted is None:
            yield event
            continue

        observed = _disseminated_at(accepted)
        tdate = event.payload.get("transaction_date")
        occurred = event.occurred_at
        if tdate:
            from datetime import date as _date

            occurred = _occurred_at(_date.fromisoformat(tdate), observed)

        yield Event(
            source=event.source,
            external_id=event.external_id,
            kind=event.kind,
            symbol=event.symbol,
            observed_at=observed,
            occurred_at=occurred,
            payload={**event.payload, "precision": "timed"},
            revision=event.revision,
        )
