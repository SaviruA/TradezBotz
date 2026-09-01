"""Geopolitical risk as a regime conditioner.

The observation this exists to serve: our sentiment work and a geopolitical
monitor are after the same thing from opposite ends. Sentiment asks "what is
being said about this company"; a geopolitical index asks "what is being said
about the world". For a universe of microcaps that news barely covers, the
second question is the answerable one -- and it does not need per-symbol
coverage, because it is one series conditioning every symbol.

That reframing is what makes it usable. `news sentiment` stays blocked because
no per-symbol history exists at any price. A macro regime has no such problem:
one number per day, and the whole history is published.

**Source: Caldara & Iacoviello, Measuring Geopolitical Risk (Fed IFDP 1222).**
A count of geopolitical-tension language across ten newspaper archives. The
daily series runs from 1985 and is free from the authors. Measured against our
labelling window: 3,896 daily observations from 2016, zero missing days, mean
115.0 and standard deviation 61.5 -- real variation, not a flat line. Its event
labels line up with the Iraq war, the London bombings, and Russia/Ukraine.

    https://www.matteoiacoviello.com/gpr.htm

**Why this is a conditioner and never a signal.** One world-level series cannot
distinguish two symbols on the same day. It can only ask whether a signal
behaves differently across regimes -- "does insider buying pay better when the
world looks dangerous?" -- which is a genuine question and exactly the shape
`backtest.all_of` exists for.

**The revision caveat, stated because it is the honest weakness.** GPR is
recomputed, and a methodology change revises the whole history. Using today's
file to evaluate a 2019 decision therefore risks the same back-door lookahead
that rules out Yahoo for fundamentals. Two things make it much milder here: the
index is a mechanical text count over fixed newspaper archives rather than a
judgement that can be restated, and every fetch is stamped so a revision is at
least detectable. It is not eliminated, and a result that depends on GPR should
be read with that in mind.

**Regime is a TRAILING percentile, never a full-sample one.** Ranking a 2016
day against the 1985-2026 distribution would use four decades of future data to
decide what counted as "high risk" at the time. `regime_at` uses only the days
before the one being asked about.
"""

from __future__ import annotations

import sqlite3
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

#: Trailing window for the regime percentile. Five years is long enough to span
#: more than one geopolitical cycle and short enough that "high risk" means high
#: relative to the recent world rather than to the Cold War.
REGIME_LOOKBACK_DAYS = 5 * 365

#: Minimum trailing observations before a percentile means anything. Below this
#: `regime_at` returns None rather than ranking against a handful of days.
MIN_REGIME_HISTORY = 250

#: Percentile boundaries for the regime bands.
HIGH_RISK_PERCENTILE = 0.80
LOW_RISK_PERCENTILE = 0.20

SCHEMA = """
CREATE TABLE IF NOT EXISTS gpr_daily (
    day         TEXT PRIMARY KEY,
    gprd        REAL NOT NULL,
    gprd_act    REAL,
    gprd_threat REAL,
    fetched_at  TEXT NOT NULL
);
"""


class MacroError(RuntimeError):
    pass


@dataclass(frozen=True)
class GprDay:
    day: date
    gprd: float
    acts: float | None = None
    threats: float | None = None


class MacroStore:
    """Daily geopolitical risk, with the fetch stamped.

    The stamp is not decoration. GPR is recomputed when its methodology moves,
    so knowing when a value was pulled is the only way to notice that the past
    changed underneath a result.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._days: list[date] | None = None
        self._values: list[float] | None = None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MacroStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def put_many(self, rows: Iterable[GprDay]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        payload = [(r.day.isoformat(), r.gprd, r.acts, r.threats, now)
                   for r in rows]
        self._conn.executemany(
            "INSERT OR REPLACE INTO gpr_daily VALUES (?,?,?,?,?)", payload)
        self._conn.commit()
        self._days = self._values = None
        return len(payload)

    def _load(self) -> tuple[list[date], list[float]]:
        if self._days is None:
            rows = self._conn.execute(
                "SELECT day, gprd FROM gpr_daily ORDER BY day").fetchall()
            self._days = [date.fromisoformat(r["day"]) for r in rows]
            self._values = [float(r["gprd"]) for r in rows]
        return self._days, self._values

    def count(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM gpr_daily").fetchone()[0])

    def span(self) -> tuple[date, date] | None:
        row = self._conn.execute(
            "SELECT MIN(day), MAX(day) FROM gpr_daily").fetchone()
        if not row or not row[0]:
            return None
        return date.fromisoformat(row[0]), date.fromisoformat(row[1])

    def fetched_at(self) -> str | None:
        row = self._conn.execute(
            "SELECT MAX(fetched_at) FROM gpr_daily").fetchone()
        return row[0] if row else None

    def value_at(self, day: date) -> float | None:
        """Most recent GPR reading strictly BEFORE `day`.

        Strictly before, because the day's own index is a count over that day's
        newspapers and is not available while the session is being traded.
        """
        days, values = self._load()
        i = bisect_left(days, day)
        return values[i - 1] if i > 0 else None

    def regime_at(self, day: date, *,
                  lookback_days: int = REGIME_LOOKBACK_DAYS) -> dict | None:
        """Where the latest reading sits in its own TRAILING distribution.

        Full-sample percentiles would be lookahead of the worst kind: they would
        use four decades of subsequent history to decide what counted as high
        risk in 2016, and every regime label would carry a little of the future.
        Only days strictly before `day` are considered.
        """
        days, values = self._load()
        end = bisect_left(days, day)
        if end == 0:
            return None
        floor_day = day - timedelta(days=lookback_days)
        start = bisect_left(days, floor_day)
        window = values[start:end]
        if len(window) < MIN_REGIME_HISTORY:
            return None

        current = values[end - 1]
        ordered = sorted(window)
        rank = bisect_left(ordered, current) / len(ordered)
        return {
            "gpr": current,
            "gpr_percentile": rank,
            "gpr_high": rank >= HIGH_RISK_PERCENTILE,
            "gpr_low": rank <= LOW_RISK_PERCENTILE,
            "gpr_observations": len(window),
        }


def parse_gpr_workbook(raw: bytes) -> list[GprDay]:
    """Read the published daily workbook into rows.

    Kept separate from fetching so the parser is testable without a network
    call, and so a change in the file's shape fails here with a clear message
    rather than somewhere downstream.
    """
    import io

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MacroError("pandas is required to read the GPR workbook") from exc

    try:
        frame = pd.read_excel(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise MacroError(
            "could not read the GPR workbook. It is a legacy .xls, which needs "
            "`pip install xlrd`."
        ) from exc

    if "date" not in frame.columns or "GPRD" not in frame.columns:
        raise MacroError(
            f"GPR workbook shape changed: expected 'date' and 'GPRD', got "
            f"{list(frame.columns)[:10]}"
        )

    out: list[GprDay] = []
    for _, row in frame.iterrows():
        when = row["date"]
        value = row["GPRD"]
        if pd.isna(when) or pd.isna(value):
            continue
        out.append(GprDay(
            day=pd.Timestamp(when).date(),
            gprd=float(value),
            acts=None if pd.isna(row.get("GPRD_ACT")) else float(row["GPRD_ACT"]),
            threats=(None if pd.isna(row.get("GPRD_THREAT"))
                     else float(row["GPRD_THREAT"])),
        ))
    return out


def fetch_gpr(url: str = GPR_DAILY_URL, session=None) -> list[GprDay]:
    """Download the daily geopolitical risk series."""
    import requests

    sess = session or requests.Session()
    resp = sess.get(url, timeout=180,
                    headers={"User-Agent": "tradezbotz research"})
    resp.raise_for_status()
    return parse_gpr_workbook(resp.content)
