"""Point-in-time event store.

This module exists to enforce one rule:

    A backtest evaluating time T may only see rows whose `observed_at` <= T.

`occurred_at` -- when the underlying thing actually happened -- is recorded for
analysis but is never a query key for backtests. For our signals it is routinely
days (Form 4) or weeks (congressional PTRs) earlier than the moment the
information became public, and querying on it silently manufactures returns that
were never available to anyone.

The store is append-only. Revisions of an event are added as new rows rather
than overwriting, so a later correction cannot retroactively rewrite what the
market saw at the time.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    source        TEXT    NOT NULL,
    external_id   TEXT    NOT NULL,
    revision      INTEGER NOT NULL,
    observed_at   TEXT    NOT NULL,
    occurred_at   TEXT,
    symbol        TEXT,
    kind          TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    ingested_at   TEXT    NOT NULL,
    PRIMARY KEY (source, external_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_events_observed ON events(observed_at);
CREATE INDEX IF NOT EXISTS idx_events_symbol   ON events(symbol, observed_at);

-- Which source-days have been fully ingested. Events are deduped on insert, so
-- this is not needed for correctness -- it exists so a run sliced across many
-- short sessions does not re-download filings it already holds. Re-fetching a
-- day of EDGAR costs ~1000 requests; skipping it costs one lookup.
CREATE TABLE IF NOT EXISTS ingested_days (
    source       TEXT    NOT NULL,
    day          TEXT    NOT NULL,
    events       INTEGER NOT NULL,
    completed_at TEXT    NOT NULL,
    PRIMARY KEY (source, day)
);
"""


class EventStoreError(RuntimeError):
    pass


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise EventStoreError(
            f"{label} must be timezone-aware. Naive datetimes are the most common "
            "source of off-by-one-session lookahead bias."
        )


def _utc(value: datetime, label: str) -> str:
    _require_aware(value, label)
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    """A single observation, stamped with when it became knowable."""

    source: str
    external_id: str
    kind: str
    observed_at: datetime
    payload: dict[str, Any]
    occurred_at: datetime | None = None
    symbol: str | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.source or not self.external_id:
            raise EventStoreError("source and external_id are required")
        # Checked before any comparison: mixing naive and aware datetimes raises
        # an unhelpful TypeError, and a naive timestamp is itself the bug.
        _require_aware(self.observed_at, "observed_at")
        if self.occurred_at is not None:
            _require_aware(self.occurred_at, "occurred_at")
        if self.occurred_at and self.occurred_at > self.observed_at:
            raise EventStoreError(
                f"{self.external_id}: occurred_at is after observed_at, which would "
                "mean the event was visible before it happened"
            )


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def raw_query(self, sql: str, params: tuple = ()):
        """Read-only escape hatch for aggregate scans over the whole store.

        `as_of` is the right interface for anything that reasons about one
        point in time, and nothing here should bypass its revision handling to
        answer such a question. This exists for the other shape: a single pass
        that builds an index, where going through `as_of` would deserialise
        several million payloads in Python to read two fields out of each.

        Refuses anything but SELECT. The store's whole value is that history
        cannot be rewritten under a backtest, and a general-purpose execute()
        on it would be a loaded gun pointed at that guarantee.
        """
        if not sql.lstrip().upper().startswith("SELECT"):
            raise EventStoreError(
                "raw_query is read-only: the event store's guarantee is that "
                "history cannot be rewritten under a running backtest"
            )
        return self._conn.execute(sql, params)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record(self, event: Event) -> bool:
        """Insert an event. Returns False if this revision already exists.

        Idempotent by (source, external_id, revision) so re-running an ingest
        over an overlapping date range is safe.
        """
        try:
            self._conn.execute(
                "INSERT INTO events (source, external_id, revision, observed_at, "
                "occurred_at, symbol, kind, payload, ingested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    event.source,
                    event.external_id,
                    event.revision,
                    _utc(event.observed_at, "observed_at"),
                    _utc(event.occurred_at, "occurred_at") if event.occurred_at else None,
                    event.symbol,
                    event.kind,
                    json.dumps(event.payload, sort_keys=True, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def record_many(self, events: Sequence[Event]) -> int:
        return sum(1 for e in events if self.record(e))

    def as_of(
        self,
        when: datetime,
        *,
        kind: str | None = None,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield everything knowable at `when` -- and nothing else.

        Where several revisions of an event are visible by `when`, the latest
        visible revision wins. Later revisions stay hidden until their own
        observed_at, so corrections do not leak backwards.
        """
        sql = [
            "SELECT e.* FROM events e",
            "JOIN (SELECT source, external_id, MAX(revision) AS rev FROM events",
            "      WHERE observed_at <= :when GROUP BY source, external_id) latest",
            "  ON e.source = latest.source AND e.external_id = latest.external_id",
            " AND e.revision = latest.rev",
            "WHERE e.observed_at <= :when",
        ]
        params: dict[str, Any] = {"when": _utc(when, "when")}
        if kind:
            sql.append("AND e.kind = :kind")
            params["kind"] = kind
        if symbol:
            sql.append("AND e.symbol = :symbol")
            params["symbol"] = symbol.upper()
        if since:
            sql.append("AND e.observed_at >= :since")
            params["since"] = _utc(since, "since")
        sql.append("ORDER BY e.observed_at")

        for row in self._conn.execute(" ".join(sql), params):
            record = dict(row)
            record["payload"] = json.loads(record["payload"])
            yield record

    # -- ingest checkpointing ------------------------------------------------

    def mark_day_ingested(self, source: str, day: date, events: int) -> None:
        """Record that `day` is fully pulled, so a resumed run can skip it."""
        self._conn.execute(
            "INSERT OR REPLACE INTO ingested_days VALUES (?,?,?,?)",
            (source, day.isoformat(), events, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def day_ingested(self, source: str, day: date) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM ingested_days WHERE source = ? AND day = ?",
                (source, day.isoformat()),
            ).fetchone()
            is not None
        )

    def days_ingested(self, source: str) -> set[date]:
        return {
            date.fromisoformat(r[0])
            for r in self._conn.execute(
                "SELECT day FROM ingested_days WHERE source = ?", (source,)
            )
        }

    def count(self, kind: str | None = None) -> int:
        if kind:
            cur = self._conn.execute("SELECT COUNT(*) FROM events WHERE kind = ?", (kind,))
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM events")
        return int(cur.fetchone()[0])
