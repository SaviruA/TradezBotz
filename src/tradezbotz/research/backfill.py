"""Resumable price backfill.

At 5 requests/minute a few thousand symbols is a many-hour job, so it must
survive the things that actually kill long jobs: a dropped SSH session, a VM
reboot, a vendor outage, an accidental Ctrl-C.

Design consequences:

* **Checkpoint after every symbol.** A hard kill loses at most one symbol's work.
* **Symbols are the unit, not events.** Many events share a ticker; fetching per
  event would multiply a ten-hour job into weeks.
* **Failures are recorded and retried, not fatal.** One bad ticker must not end
  a twelve-hour run. Symbols that fail `max_attempts` times are parked as
  `failed` and reported, so they are visible rather than silently absent.
* **SIGTERM/SIGINT stop cleanly at the next symbol boundary**, which is what
  `systemctl stop` and Ctrl-C send. Half-written state is worse than no state.
"""

from __future__ import annotations

import signal
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .prices import PriceError, PriceSource

CHECKPOINT_SCHEMA = """
CREATE TABLE IF NOT EXISTS backfill (
    symbol     TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    bars       INTEGER,
    is_active  INTEGER,
    last_error TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_backfill_status ON backfill(status);
"""

PENDING = "pending"
DONE = "done"
FAILED = "failed"

DEFAULT_MAX_ATTEMPTS = 3


@dataclass
class Progress:
    total: int = 0
    done: int = 0
    failed: int = 0
    fetched_this_run: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done - self.failed)

    @property
    def rate_per_hour(self) -> float:
        elapsed = time.monotonic() - self.started_at
        if elapsed <= 0 or self.fetched_this_run == 0:
            return 0.0
        return self.fetched_this_run / elapsed * 3600.0

    @property
    def eta(self) -> timedelta | None:
        rate = self.rate_per_hour
        if rate <= 0:
            return None
        return timedelta(hours=self.remaining / rate)

    def __str__(self) -> str:
        eta = self.eta
        eta_s = "unknown" if eta is None else str(timedelta(seconds=int(eta.total_seconds())))
        return (
            f"{self.done}/{self.total} done, {self.failed} failed, "
            f"{self.remaining} left, {self.rate_per_hour:.0f}/h, ETA {eta_s}"
        )


class BackfillRunner:
    def __init__(
        self,
        source: PriceSource,
        checkpoint_path: str | Path,
        *,
        start: date,
        end: date,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.source = source
        self.start = start
        self.end = end
        self.max_attempts = max_attempts
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(CHECKPOINT_SCHEMA)
        self._conn.commit()
        self._stop = False

    def close(self) -> None:
        self._conn.close()

    # -- queue ---------------------------------------------------------------

    def enqueue(self, symbols: Iterable[str]) -> int:
        """Add symbols to the queue. Already-known symbols keep their status,
        so re-enqueueing after new events arrive does not redo finished work."""
        added = 0
        now = datetime.now(timezone.utc).isoformat()
        for raw in symbols:
            symbol = (raw or "").strip().upper()
            if not symbol:
                continue
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO backfill (symbol, status, updated_at) "
                "VALUES (?, ?, ?)",
                (symbol, PENDING, now),
            )
            added += cur.rowcount
        self._conn.commit()
        return added

    def pending(self, limit: int | None = None) -> list[str]:
        sql = (
            "SELECT symbol FROM backfill WHERE status = ? "
            "   OR (status = ? AND attempts < ?) ORDER BY symbol"
        )
        params: list = [PENDING, FAILED, self.max_attempts]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [r[0] for r in self._conn.execute(sql, params)]

    def progress(self) -> Progress:
        rows = dict(
            self._conn.execute("SELECT status, COUNT(*) FROM backfill GROUP BY status")
        )
        total = sum(rows.values())
        return Progress(total=total, done=rows.get(DONE, 0), failed=rows.get(FAILED, 0))

    def failures(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT symbol, attempts, last_error FROM backfill "
                "WHERE status = ? ORDER BY symbol",
                (FAILED,),
            )
        )

    # -- execution -----------------------------------------------------------

    def request_stop(self, *_: object) -> None:
        """Ask the run loop to finish the current symbol and exit."""
        self._stop = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):
                pass  # not on the main thread; caller drives stop directly

    def run(
        self,
        *,
        limit: int | None = None,
        on_progress: Callable[[str, Progress], None] | None = None,
    ) -> Progress:
        symbols = self.pending(limit)
        prog = self.progress()
        prog.started_at = time.monotonic()

        for symbol in symbols:
            if self._stop:
                break
            try:
                series = self.source.daily_bars(symbol, self.start, self.end)
            except Exception as exc:  # noqa: BLE001
                # Deliberately broad: one malformed ticker must not end a
                # twelve-hour run. Failures are parked and reported, not hidden.
                self._record_failure(symbol, exc)
                prog.failed += 1
            else:
                self._record_success(symbol, len(series.bars), series.is_active)
                prog.done += 1
            prog.fetched_this_run += 1
            if on_progress:
                on_progress(symbol, prog)

        return prog

    def _record_success(self, symbol: str, bars: int, is_active: bool | None) -> None:
        self._conn.execute(
            "UPDATE backfill SET status = ?, bars = ?, is_active = ?, last_error = NULL, "
            "attempts = attempts + 1, updated_at = ? WHERE symbol = ?",
            (
                DONE,
                bars,
                None if is_active is None else int(is_active),
                datetime.now(timezone.utc).isoformat(),
                symbol,
            ),
        )
        self._conn.commit()

    def _record_failure(self, symbol: str, exc: BaseException) -> None:
        self._conn.execute(
            "UPDATE backfill SET status = ?, attempts = attempts + 1, last_error = ?, "
            "updated_at = ? WHERE symbol = ?",
            (
                FAILED,
                f"{type(exc).__name__}: {exc}"[:500],
                datetime.now(timezone.utc).isoformat(),
                symbol,
            ),
        )
        self._conn.commit()


def symbols_from_events(events: Iterable[dict]) -> list[str]:
    """Distinct, sorted symbols from an event iterable."""
    seen = {(e.get("symbol") or "").upper() for e in events}
    return sorted(s for s in seen if s)
