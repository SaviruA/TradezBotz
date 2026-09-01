"""Did the pipeline actually run, and did it actually do anything?

Two failure modes, and the second is the dangerous one.

**A run that does not happen.** GitHub schedules workflows on a best-effort
basis: runs are delayed under load, occasionally dropped, and scheduled
workflows are disabled outright after 60 days without repository activity. None
of that produces a failure notification, because nothing failed -- nothing ran.
The documented answer is a dead man's switch: monitor *expected execution*
rather than failures, and alert when the heartbeat does not arrive.

**A run that happens and silently does nothing.** This one is ours rather than
GitHub's. Eight of twenty steps carry `continue-on-error: true`, which is
correct in itself -- a failing intraday fetch must not discard an hour of EDGAR
ingestion -- but it means a run can report success having accomplished almost
nothing. Every optional step could fail on every run, for weeks, and the badge
would stay green.

`RunLog` addresses both by recording what each run completed. The next run reads
it, so a gap is detectable retroactively without any external service, and a
step that has been failing for days is visible as a streak rather than as one
line buried in a log nobody opened.

**Retroactive is weaker than external, and that is a deliberate trade.** A true
dead man's switch needs a third party to notice silence. This notices at the
next run, which is no help if runs have stopped entirely -- but it needs no
service, no secret and no account, and it catches the far more common case of a
pipeline that is running while quietly broken.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

#: Hours after which a missing run is a problem worth failing on. The schedule
#: is daily, so 36 allows one late or delayed run before complaining.
STALE_AFTER_HOURS = 36

#: Consecutive failures of one step before it stops being noise and becomes a
#: finding. One failed fetch is weather; three in a row is a broken step.
FAILURE_STREAK_ALERT = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE TABLE IF NOT EXISTS step_outcomes (
    run_id     TEXT NOT NULL,
    step       TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (run_id, step)
);
CREATE INDEX IF NOT EXISTS idx_step_outcomes_step ON step_outcomes(step);
"""


@dataclass(frozen=True)
class StepHealth:
    step: str
    last_outcome: str
    streak: int

    @property
    def alerting(self) -> bool:
        return self.last_outcome != "success" and self.streak >= FAILURE_STREAK_ALERT


class RunLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self, run_id: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, started_at, finished_at) "
            "VALUES (?,?,COALESCE((SELECT finished_at FROM runs WHERE run_id=?),NULL))",
            (run_id, datetime.now(timezone.utc).isoformat(), run_id))
        self._conn.commit()

    def finish(self, run_id: str) -> None:
        self._conn.execute("UPDATE runs SET finished_at = ? WHERE run_id = ?",
                           (datetime.now(timezone.utc).isoformat(), run_id))
        self._conn.commit()

    def record(self, run_id: str, step: str, outcome: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO step_outcomes VALUES (?,?,?,?)",
            (run_id, step, outcome, datetime.now(timezone.utc).isoformat()))
        self._conn.commit()

    def last_finished(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT MAX(finished_at) FROM runs WHERE finished_at IS NOT NULL"
        ).fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def hours_since_last_run(self, now: datetime | None = None) -> float | None:
        last = self.last_finished()
        if last is None:
            return None
        return ((now or datetime.now(timezone.utc)) - last).total_seconds() / 3600

    def health(self) -> list[StepHealth]:
        """Per-step outcome and how long it has been that way.

        The streak is what turns a log line into a finding. A step that failed
        once is weather; a step that has failed every run for a week is broken
        and nobody noticed, which is the condition `continue-on-error` creates.
        """
        steps = [r[0] for r in self._conn.execute(
            "SELECT DISTINCT step FROM step_outcomes ORDER BY step")]
        out: list[StepHealth] = []
        for step in steps:
            rows = [r["outcome"] for r in self._conn.execute(
                "SELECT outcome FROM step_outcomes WHERE step = ? "
                "ORDER BY recorded_at DESC LIMIT 30", (step,))]
            if not rows:
                continue
            latest = rows[0]
            streak = 0
            for outcome in rows:
                if outcome != latest:
                    break
                streak += 1
            out.append(StepHealth(step, latest, streak))
        return out

    def runs_recorded(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM runs").fetchone()[0])


def describe(log: RunLog, *, stale_after_hours: int = STALE_AFTER_HOURS,
             now: datetime | None = None) -> tuple[str, bool]:
    """Human-readable health, and whether it warrants failing the run.

    Returns (text, unhealthy). Failing the run is the point: a pipeline that
    reports success while a third of its steps have been broken for a week is
    worse than one that reports failure, because the green badge is read as
    evidence that the data is fine.
    """
    lines: list[str] = []
    unhealthy = False

    gap = log.hours_since_last_run(now)
    if gap is None:
        lines.append("no previous run recorded (first run, or state was reset)")
    elif gap > stale_after_hours:
        unhealthy = True
        lines.append(
            f"STALE: {gap:.0f}h since the last completed run, expected under "
            f"{stale_after_hours}h. Scheduled runs are delayed or dropped under "
            "load, and are disabled entirely after 60 days without repository "
            "activity -- neither of which raises a failure.")
    else:
        lines.append(f"last completed run {gap:.1f}h ago")

    health = log.health()
    if health:
        lines.append("")
        lines.append(f"{'step':<38}{'last':>10}{'streak':>8}")
        for h in sorted(health, key=lambda x: (x.last_outcome == "success", x.step)):
            flag = "  <-- ALERT" if h.alerting else ""
            lines.append(f"{h.step[:37]:<38}{h.last_outcome:>10}{h.streak:>8}{flag}")
        alerting = [h for h in health if h.alerting]
        if alerting:
            unhealthy = True
            lines.append("")
            lines.append(
                f"{len(alerting)} step(s) have failed {FAILURE_STREAK_ALERT}+ "
                "runs in a row. These carry continue-on-error, so the pipeline "
                "has been reporting success while not doing this work.")
    return "\n".join(lines), unhealthy
