"""Price data adapters.

`PriceSource` is the seam. Everything downstream depends on this protocol and
not on Massive specifically, so swapping in a paid source (or a second source
for cross-checking) is a constructor change rather than a rewrite.

Two properties the implementations must preserve:

* **Bars define the trading calendar.** We never assume which days were sessions;
  we read that off the returned bars. This avoids shipping a holiday calendar
  and silently disagreeing with the venue.
* **Absence is data.** A ticker whose bars stop mid-window has probably been
  delisted, and that is exactly the observation survivorship bias destroys. The
  source reports where the data ends rather than quietly returning a short list.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

import requests

MASSIVE_BASE = "https://api.massive.com"

#: Free tier allows 5 requests/minute and enforces it with HTTP 429.
DEFAULT_REQUESTS_PER_MINUTE = 5

#: Sessions of silence after which a ticker is suspicious enough to spend a
#: request confirming whether it still trades. Long weekends and holidays
#: make anything under ~4 days noisy.
STALE_BAR_DAYS = 5

#: Free tier serves roughly two years of daily history regardless of the range
#: requested. Recorded here so callers can reason about coverage gaps rather
#: than mistaking the cap for a delisting.
FREE_TIER_HISTORY_DAYS = 730


class PriceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bar:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Series:
    """Daily bars for one symbol, plus what the source knows about coverage."""

    symbol: str
    bars: tuple[Bar, ...]
    requested_start: date
    requested_end: date
    is_active: bool | None = None

    @property
    def first_day(self) -> date | None:
        return self.bars[0].day if self.bars else None

    @property
    def last_day(self) -> date | None:
        return self.bars[-1].day if self.bars else None

    def index_on_or_after(self, day: date) -> int | None:
        for i, bar in enumerate(self.bars):
            if bar.day >= day:
                return i
        return None


class PriceSource(Protocol):
    def daily_bars(self, symbol: str, start: date, end: date) -> Series: ...


class RateLimiter:
    """Sliding-window limiter. Blocks rather than dropping requests."""

    def __init__(self, per_minute: int = DEFAULT_REQUESTS_PER_MINUTE) -> None:
        self.per_minute = per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request slot is free. Returns seconds waited."""
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            waited = 0.0
            if len(self._calls) >= self.per_minute:
                waited = 60.0 - (now - self._calls[0]) + 0.25
                if waited > 0:
                    time.sleep(waited)
                    now = time.monotonic()
                    while self._calls and now - self._calls[0] >= 60.0:
                        self._calls.popleft()
            self._calls.append(time.monotonic())
            return waited


CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    day    TEXT NOT NULL,
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS fetches (
    symbol     TEXT PRIMARY KEY,
    start_day  TEXT NOT NULL,
    end_day    TEXT NOT NULL,
    is_active  INTEGER,
    fetched_at TEXT NOT NULL
);
"""


class PriceCache:
    """On-disk bar cache.

    Many events share a symbol, and the free tier allows 5 requests a minute, so
    refetching per event would turn a ten-hour backfill into weeks. The cache
    also makes the whole pipeline reproducible: a rerun reads the same bars
    rather than whatever the vendor serves today.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(CACHE_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def covered(self, symbol: str, start: date, end: date) -> bool:
        row = self._conn.execute(
            "SELECT start_day, end_day FROM fetches WHERE symbol = ?", (symbol,)
        ).fetchone()
        if not row:
            return False
        return row["start_day"] <= start.isoformat() and row["end_day"] >= end.isoformat()

    def get(self, symbol: str, start: date, end: date) -> Series:
        rows = self._conn.execute(
            "SELECT * FROM bars WHERE symbol = ? AND day BETWEEN ? AND ? ORDER BY day",
            (symbol, start.isoformat(), end.isoformat()),
        ).fetchall()
        meta = self._conn.execute(
            "SELECT is_active FROM fetches WHERE symbol = ?", (symbol,)
        ).fetchone()
        active = None if meta is None or meta["is_active"] is None else bool(meta["is_active"])
        return Series(
            symbol=symbol,
            bars=tuple(
                Bar(
                    day=date.fromisoformat(r["day"]),
                    open=r["open"], high=r["high"], low=r["low"],
                    close=r["close"], volume=r["volume"],
                )
                for r in rows
            ),
            requested_start=start,
            requested_end=end,
            is_active=active,
        )

    def put(self, series: Series) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)",
            [
                (series.symbol, b.day.isoformat(), b.open, b.high, b.low, b.close, b.volume)
                for b in series.bars
            ],
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO fetches VALUES (?,?,?,?,?)",
            (
                series.symbol,
                series.requested_start.isoformat(),
                series.requested_end.isoformat(),
                None if series.is_active is None else int(series.is_active),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def symbols(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT symbol FROM fetches ORDER BY symbol")]


class MassivePriceSource:
    """Massive (formerly Polygon.io) daily aggregates.

    Bars are requested split- and dividend-adjusted. Adjustment is applied by the
    vendor as of fetch time, which is why `PriceCache` matters: without it, a
    dividend paid tomorrow silently changes yesterday's backtest.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache: PriceCache | None = None,
        per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        session: requests.Session | None = None,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key or os.environ.get("MASSIVE_API_KEY", "")
        if not self.api_key:
            raise PriceError("MASSIVE_API_KEY is not set. See .env.example.")
        self.cache = cache
        self.limiter = RateLimiter(per_minute)
        self.session = session or requests.Session()
        self.max_retries = max_retries

    def _get(self, url: str, params: dict) -> dict:
        params = {**params, "apiKey": self.api_key}
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            resp = self.session.get(url, params=params, timeout=45)
            if resp.status_code == 429:
                # The limiter should prevent this; if the vendor disagrees, defer
                # to the vendor rather than hammering.
                time.sleep(min(60, 5 * 2**attempt))
                continue
            if resp.status_code == 403:
                raise PriceError(
                    f"403 from Massive for {url}. On the free tier this usually "
                    "means the resource requires a paid plan (Flat Files, extended "
                    "history), not that the key is invalid."
                )
            resp.raise_for_status()
            return resp.json()
        raise PriceError(f"Rate limited by Massive after {self.max_retries} attempts: {url}")

    def is_active(self, symbol: str) -> bool | None:
        """Whether the vendor still lists the ticker as trading.

        The reference endpoint defaults to `active=true`, so a delisted ticker
        comes back as *zero results* rather than `active: false`. Taking that at
        face value reports None, and the labeller then downgrades a real
        delisting to a mere coverage gap -- reintroducing survivorship bias at
        precisely the point we built this to catch it. So an empty active lookup
        is followed by an explicit inactive lookup before we give up.

        Returns True (listed), False (delisted), or None (vendor has no record).
        """
        def lookup(params: dict) -> list[dict]:
            try:
                body = self._get(f"{MASSIVE_BASE}/v3/reference/tickers", params)
            except (PriceError, requests.HTTPError):
                return []
            return body.get("results") or []

        if lookup({"ticker": symbol}):
            return True
        if lookup({"ticker": symbol, "active": "false"}):
            return False
        return None

    def daily_bars(self, symbol: str, start: date, end: date) -> Series:
        symbol = symbol.upper()
        if self.cache and self.cache.covered(symbol, start, end):
            return self.cache.get(symbol, start, end)

        body = self._get(
            f"{MASSIVE_BASE}/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}",
            {"adjusted": "true", "sort": "asc", "limit": 50000},
        )
        bars = tuple(
            Bar(
                day=datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc).date(),
                open=float(r["o"]), high=float(r["h"]), low=float(r["l"]),
                close=float(r["c"]), volume=float(r.get("v", 0.0)),
            )
            for r in (body.get("results") or [])
        )
        series = Series(
            symbol=symbol,
            bars=bars,
            requested_start=start,
            requested_end=end,
            is_active=self.is_active(symbol) if self._needs_status_check(bars, end) else None,
        )
        if self.cache:
            self.cache.put(series)
        return series

    @staticmethod
    def _needs_status_check(bars: Sequence[Bar], requested_end: date) -> bool:
        """Whether it is worth spending a request to ask if the ticker is alive.

        At 5 requests/minute the status lookup is not free -- doing it for every
        symbol roughly doubles a multi-hour backfill. A ticker whose bars run up
        to the present is self-evidently trading, so we only ask when the data
        stops early, which is the only case where the answer changes a label.

        Note `requested_end` is usually in the future (the labeller pads it past
        the longest horizon), so the comparison is against today, not the request.
        """
        if not bars:
            return False  # no bars means no evidence either way; do not spend a call
        effective_end = min(requested_end, date.today())
        return (effective_end - bars[-1].day).days > STALE_BAR_DAYS
