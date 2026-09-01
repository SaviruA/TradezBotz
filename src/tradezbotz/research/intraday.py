"""Intraday bars and trades, which volume profile and order flow require.

Daily bars cannot express either concept. A daily bar reports one volume for the
whole session, so there is no approximate point of control to recover; and order
flow needs trades classified by which side was the aggressor, which no OHLCV
field carries. This module is the fetch and store path that unblocks both.

**Why this is affordable.** Probing the account on 2026-08-30 showed the free
Alpaca plan serves the full consolidated tape (SIP), not just IEX -- 17 venues
and 209,697 shares in a minute where IEX showed one venue and 10,071. Minute
bars go back to 2016, `adjustment=all` genuinely back-adjusts them (NVDA's 10:1
split shows 1192.73 raw against 119.07 adjusted, with volume scaled to match),
and the multi-symbol endpoint returns many symbols per request. The only
restriction is that SIP data inside the last 15 minutes is refused, which never
binds on a backtest or on next-open entry.

**Why we do not store raw minute bars.** 636 symbols over 500 sessions is on the
order of 10^8 rows, which does not fit the encrypted CI state blob. Instead each
session is reduced once to a compact price histogram plus flow statistics
(`SessionProfile`), which is the form both strategies actually consume. Multi-day
profiles merge histograms rather than refetching. That is ~200 bytes per
symbol-day instead of ~40KB.

**Regular hours only, by default.** AAPL returns 1,618 minute bars for a session
that has 390 regular minutes; the rest is pre- and post-market. Extended-hours
prints are thin and wide, and including them drags a point of control toward
prices almost nobody traded at. `rth_only` defaults to True for that reason, and
the flag is recorded on the stored profile so a mixed store cannot go unnoticed.
"""

from __future__ import annotations

import os
import sqlite3
import struct
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence
from zoneinfo import ZoneInfo

import requests

from .prices import PriceError, RateLimiter

ALPACA_DATA_BASE = "https://data.alpaca.markets"

#: The free plan allows 200 requests/minute.
ALPACA_REQUESTS_PER_MINUTE = 200

#: Symbols per multi-symbol request. Alpaca accepts long lists; we stay well
#: under any undocumented cap because a rejected batch costs more than an extra
#: request does.
MAX_SYMBOLS_PER_REQUEST = 100

#: Alpaca refuses SIP data newer than this. Enforced client-side so the failure
#: is a clear message rather than an opaque 403 mid-backfill.
SIP_DELAY_MINUTES = 15

EASTERN = ZoneInfo("America/New_York")
RTH_OPEN, RTH_CLOSE = dtime(9, 30), dtime(16, 0)

#: Price buckets per stored session histogram. 40 is enough to locate a point of
#: control within a fraction of a percent on a typical daily range, and keeps a
#: stored session near 200 bytes.
PROFILE_BUCKETS = 40

#: Columns added to `session_profiles` after the table first shipped. An
#: existing database is migrated by adding whichever of these it lacks; a fresh
#: one gets them from PROFILE_SCHEMA. Kept as one list so the two paths cannot
#: drift apart and produce stores with different shapes.
PROFILE_TIMING_COLUMNS = (
    ("session_open", "REAL"),
    ("session_close", "REAL"),
    ("low_minute", "INTEGER"),
    ("high_minute", "INTEGER"),
    ("volume_after_low", "REAL"),
    ("volume_after_high", "REAL"),
)


class IntradayError(PriceError):
    pass


@dataclass(frozen=True)
class MinuteBar:
    ts: datetime          # timezone-aware, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trades: int = 0

    @property
    def typical(self) -> float:
        """Price at which this minute's volume is attributed.

        The bar's own VWAP when the source reports one -- Alpaca does -- since
        that is the actual volume-weighted price rather than a proxy for it.
        """
        if self.vwap is not None:
            return self.vwap
        return (self.high + self.low + self.close) / 3.0

    def in_regular_hours(self) -> bool:
        t = self.ts.astimezone(EASTERN).time()
        return RTH_OPEN <= t < RTH_CLOSE


class IntradaySource(Protocol):
    def minute_bars(self, symbols: Sequence[str], start: date, end: date
                    ) -> dict[str, list[MinuteBar]]: ...


def _parse_bar(raw: dict) -> MinuteBar:
    ts = datetime.fromisoformat(raw["t"].replace("Z", "+00:00")).astimezone(timezone.utc)
    return MinuteBar(
        ts=ts, open=raw["o"], high=raw["h"], low=raw["l"], close=raw["c"],
        volume=raw["v"], vwap=raw.get("vw"), trades=raw.get("n", 0),
    )


class AlpacaIntradaySource:
    """Minute bars and trades from Alpaca, consolidated feed.

    Requests are batched across symbols because the multi-symbol endpoint costs
    one rate-limit slot regardless of how many symbols it carries -- the
    difference between a backfill that fits in a CI job and one that does not.
    """

    def __init__(self, api_key: str | None = None, api_secret: str | None = None,
                 *, per_minute: int = ALPACA_REQUESTS_PER_MINUTE,
                 feed: str = "sip", session: requests.Session | None = None) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_PAPER_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_PAPER_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise IntradayError(
                "Alpaca credentials missing. Set ALPACA_PAPER_API_KEY and "
                "ALPACA_PAPER_API_SECRET. See .env.example."
            )
        self.feed = feed
        self.limiter = RateLimiter(per_minute)
        self.session = session or requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}

    def _end_param(self, end: date) -> str:
        """Clamp the window's end out of the SIP embargo.

        Date granularity is not enough. A request ending "today" is expanded by
        the API to cover today's session, which reaches inside the 15-minute
        embargo and returns 403 for the whole batch -- so an otherwise valid
        backfill of the previous month dies on its most recent day. Clamping to
        an explicit `now - 16m` timestamp keeps today's session available right
        up to the embargo instead of discarding it.
        """
        requested = datetime.combine(end, dtime(0, 0), tzinfo=timezone.utc)
        if self.feed != "sip":
            return requested.isoformat().replace("+00:00", "Z")
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=SIP_DELAY_MINUTES + 1)
        return min(requested, cutoff).isoformat().replace("+00:00", "Z")

    def minute_bars(self, symbols: Sequence[str], start: date, end: date,
                    *, rth_only: bool = True) -> dict[str, list[MinuteBar]]:
        """Minute bars for several symbols over a date range.

        `end` is exclusive, matching the daily adapter. Bars are split- and
        dividend-adjusted; an unadjusted intraday series spanning a split would
        place a session's volume at prices that never existed.
        """
        end_param = self._end_param(end)
        out: dict[str, list[MinuteBar]] = {s.upper(): [] for s in symbols}
        wanted = [s.upper() for s in symbols]
        for i in range(0, len(wanted), MAX_SYMBOLS_PER_REQUEST):
            chunk = wanted[i : i + MAX_SYMBOLS_PER_REQUEST]
            for symbol, bars in self._fetch_chunk(chunk, start, end_param):
                out.setdefault(symbol, []).extend(bars)
        if rth_only:
            out = {s: [b for b in bars if b.in_regular_hours()] for s, bars in out.items()}
        for bars in out.values():
            bars.sort(key=lambda b: b.ts)
        return out

    def _fetch_chunk(self, symbols: list[str], start: date, end: str
                     ) -> Iterator[tuple[str, list[MinuteBar]]]:
        page: str | None = None
        while True:
            self.limiter.acquire()
            params = {
                "symbols": ",".join(symbols),
                "timeframe": "1Min",
                "start": start.isoformat(),
                "end": end,
                "adjustment": "all",
                "feed": self.feed,
                "limit": 10000,
            }
            if page:
                params["page_token"] = page
            resp = self.session.get(f"{ALPACA_DATA_BASE}/v2/stocks/bars",
                                    params=params, headers=self._headers, timeout=60)
            if resp.status_code in (401, 403):
                raise IntradayError(
                    f"Alpaca rejected the request ({resp.status_code}): {resp.text[:200]}"
                )
            resp.raise_for_status()
            body = resp.json()
            for symbol, raw in (body.get("bars") or {}).items():
                yield symbol, [_parse_bar(r) for r in raw]
            page = body.get("next_page_token")
            if not page:
                return

    def trades(self, symbol: str, day: date, limit_pages: int = 0) -> list[dict]:
        """Raw trade prints for one session.

        Used to validate the minute-bar approximation of order flow against exact
        tick-level classification -- not in the backfill path. A full day of a
        liquid name is millions of prints; a small cap is hundreds (XELB returned
        739 for a whole session), which is why the check is affordable on the
        population insider buying actually concentrates in.
        """
        return self._paged(f"/v2/stocks/{symbol.upper()}/trades", "trades", day, limit_pages)

    def quotes(self, symbol: str, day: date, limit_pages: int = 0) -> list[dict]:
        """NBBO quotes for one session, for Lee-Ready classification.

        `limit_pages` caps the walk: quote traffic dwarfs trade traffic, and the
        validation sample does not need a whole session to measure how far the
        tick rule drifts from Lee-Ready.
        """
        return self._paged(f"/v2/stocks/{symbol.upper()}/quotes", "quotes", day, limit_pages)

    def _paged(self, path: str, key: str, day: date, limit_pages: int) -> list[dict]:
        out: list[dict] = []
        page: str | None = None
        pages = 0
        while True:
            self.limiter.acquire()
            params = {"start": day.isoformat(),
                      "end": self._end_param(day + timedelta(days=1)),
                      "feed": self.feed, "limit": 10000}
            if page:
                params["page_token"] = page
            resp = self.session.get(f"{ALPACA_DATA_BASE}{path}", params=params,
                                    headers=self._headers, timeout=60)
            if resp.status_code in (401, 403):
                raise IntradayError(f"Alpaca rejected the request ({resp.status_code}).")
            resp.raise_for_status()
            body = resp.json()
            out.extend(body.get(key) or [])
            page = body.get("next_page_token")
            pages += 1
            if not page or (limit_pages and pages >= limit_pages):
                return out


# --- stored session summaries -------------------------------------------------

@dataclass(frozen=True)
class SessionProfile:
    """One session reduced to what volume profile and order flow need.

    Deliberately lossy. The histogram fixes resolution at `PROFILE_BUCKETS`
    levels across the session range, which is enough to locate a point of control
    but not to reconstruct the session. Storing the reduction rather than the
    bars is what makes a multi-year, multi-hundred-symbol store fit at all.
    """

    symbol: str
    day: date
    low: float
    high: float
    volume: float
    vwap: float
    #: Volume in each of PROFILE_BUCKETS equal-width price bins from low to high.
    histogram: tuple[float, ...]
    #: Signed volume: buyer-initiated minus seller-initiated.
    delta: float
    #: Volume the classifier could not sign.
    unsigned_volume: float
    minute_count: int
    rth_only: bool = True
    #: Which classifier produced `delta`. Recorded because the two do not agree:
    #: measured against real prints on four small caps, minute-bar tick-rule
    #: delta matched tick-level Lee-Ready on sign only 1 time in 4. Mixing the
    #: two in one study would average a measurement against its own error, so
    #: the field exists to make a mixed store detectable rather than silent.
    flow_method: str = "tick_minute"

    # --- sequence within the session ----------------------------------------
    #
    # Everything above is order-free: min, max, sum, a histogram. That is what
    # makes the store small, and it is also what made a whole class of pattern
    # untestable, because the distinguishing content of those patterns IS the
    # order of events.
    #
    # A liquidity sweep is the clearest case. "The low pierced a prior level and
    # the close came back above it" describes both a stop run that reversed in
    # the first ten minutes and a genuine breakdown that happened to tick up at
    # 15:58. Those are opposite events and the fields above cannot tell them
    # apart. The daily bar cannot either, which is why the daily-bar version is
    # an upper bound on the population rather than a measurement of the pattern.
    #
    # Six numbers fix it, against a 40-float histogram -- the cost is noise. The
    # reason they had to be added *before* the backfill ran rather than when the
    # sweep test was written is that raw minute bars are never kept, so a field
    # missing at reduction time can only be recovered by refetching the session.
    #
    # All default to None so a session reduced before this change is detectable
    # rather than silently reading as "the low printed at minute zero".

    #: First bar's open and last bar's close. Not derivable from low/high, and
    #: without them the close's position in the session range is unknown.
    session_open: float | None = None
    session_close: float | None = None
    #: Minutes from the session's first bar to the bar carrying the extreme.
    #: Earliest bar wins on a tie.
    low_minute: int | None = None
    high_minute: int | None = None
    #: Volume that traded strictly after the extreme printed. This is the
    #: level-agnostic form of "how much session was left to reclaim in", and it
    #: is the reason these are volume shares rather than a reclaim timestamp: a
    #: reclaim is defined against a price level that is not known at reduction
    #: time, since it comes from the daily lookback.
    volume_after_low: float | None = None
    volume_after_high: float | None = None

    @property
    def has_timing(self) -> bool:
        """Whether this session carries the sequence fields.

        False for anything reduced before they existed. Check it rather than
        reading the fields, because a `None` handled as zero puts every old
        session's low at the opening bell.
        """
        return self.session_close is not None and self.low_minute is not None

    @property
    def share_after_low(self) -> float | None:
        """Fraction of session volume that traded after the low printed."""
        if self.volume_after_low is None or self.volume <= 0:
            return None
        return self.volume_after_low / self.volume

    @property
    def share_after_high(self) -> float | None:
        if self.volume_after_high is None or self.volume <= 0:
            return None
        return self.volume_after_high / self.volume

    @property
    def bucket_width(self) -> float:
        if self.high <= self.low or not self.histogram:
            return 0.0
        return (self.high - self.low) / len(self.histogram)

    def bucket_price(self, index: int) -> float:
        """Mid-price of a bucket."""
        return self.low + self.bucket_width * (index + 0.5)


PROFILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_profiles (
    symbol           TEXT NOT NULL,
    day              TEXT NOT NULL,
    low              REAL NOT NULL,
    high             REAL NOT NULL,
    volume           REAL NOT NULL,
    vwap             REAL NOT NULL,
    histogram        BLOB NOT NULL,
    delta            REAL NOT NULL,
    unsigned_volume  REAL NOT NULL,
    minute_count     INTEGER NOT NULL,
    rth_only         INTEGER NOT NULL,
    flow_method      TEXT NOT NULL DEFAULT 'tick_minute',
    -- Nullable on purpose: NULL means "reduced before these existed", which is
    -- a different thing from zero and has to stay distinguishable.
    session_open     REAL,
    session_close    REAL,
    low_minute       INTEGER,
    high_minute      INTEGER,
    volume_after_low REAL,
    volume_after_high REAL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS profile_fetches (
    symbol     TEXT NOT NULL,
    day        TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, day)
);
"""


def _pack(histogram: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(histogram)}f", *histogram)


def _unpack(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 4}f", blob)


class ProfileStore:
    """On-disk store of reduced sessions.

    Records *attempted* fetches separately from results, so a session that
    genuinely had no prints is distinguishable from one never requested. Without
    that split a backfill re-requests every empty day forever.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(PROFILE_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add the session-sequence columns to a store that predates them.

        `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a
        database written before those columns existed would silently keep the
        old shape and every insert naming them would fail. Adding them here is
        cheap: SQLite's ADD COLUMN does not rewrite rows, and existing sessions
        get NULL, which is exactly the "not recorded" they deserve.
        """
        have = {
            r["name"] for r in
            self._conn.execute("PRAGMA table_info(session_profiles)")
        }
        for name, kind in PROFILE_TIMING_COLUMNS:
            if name not in have:
                self._conn.execute(
                    f"ALTER TABLE session_profiles ADD COLUMN {name} {kind}"
                )

    def close(self) -> None:
        self._conn.close()

    def put(self, profile: SessionProfile) -> None:
        # Columns named explicitly. The positional form here broke the moment a
        # column was added: ADD COLUMN appends at the end, so a bare VALUES list
        # either fails on arity or -- worse, had the counts happened to match --
        # writes each value into the wrong column.
        self._conn.execute(
            "INSERT OR REPLACE INTO session_profiles ("
            "symbol, day, low, high, volume, vwap, histogram, delta, "
            "unsigned_volume, minute_count, rth_only, flow_method, "
            "session_open, session_close, low_minute, high_minute, "
            "volume_after_low, volume_after_high"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (profile.symbol, profile.day.isoformat(), profile.low, profile.high,
             profile.volume, profile.vwap, _pack(profile.histogram), profile.delta,
             profile.unsigned_volume, profile.minute_count, int(profile.rth_only),
             profile.flow_method,
             profile.session_open, profile.session_close,
             profile.low_minute, profile.high_minute,
             profile.volume_after_low, profile.volume_after_high),
        )
        self.mark_fetched(profile.symbol, profile.day)
        self._conn.commit()

    def mark_fetched(self, symbol: str, day: date) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO profile_fetches VALUES (?,?,?)",
            (symbol.upper(), day.isoformat(), datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def was_fetched(self, symbol: str, day: date) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM profile_fetches WHERE symbol = ? AND day = ?",
            (symbol.upper(), day.isoformat()),
        ).fetchone()
        return row is not None

    def get(self, symbol: str, day: date) -> SessionProfile | None:
        row = self._conn.execute(
            "SELECT * FROM session_profiles WHERE symbol = ? AND day = ?",
            (symbol.upper(), day.isoformat()),
        ).fetchone()
        return self._row(row) if row else None

    def range(self, symbol: str, start: date, end: date) -> list[SessionProfile]:
        rows = self._conn.execute(
            "SELECT * FROM session_profiles WHERE symbol = ? AND day BETWEEN ? AND ? "
            "ORDER BY day",
            (symbol.upper(), start.isoformat(), end.isoformat()),
        ).fetchall()
        return [self._row(r) for r in rows]

    def symbols(self) -> list[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT DISTINCT symbol FROM session_profiles ORDER BY symbol")]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0]

    def count_untimed(self) -> int:
        """Sessions reduced before the sequence fields existed.

        Reported by `status` because these are not repairable in place -- minute
        bars are not kept -- so the number is a refetch bill, and a bill that is
        invisible is one nobody pays until a study silently runs on half a store.
        """
        return self._conn.execute(
            "SELECT COUNT(*) FROM session_profiles WHERE low_minute IS NULL"
        ).fetchone()[0]

    @staticmethod
    def _row(r: sqlite3.Row) -> SessionProfile:
        return SessionProfile(
            symbol=r["symbol"], day=date.fromisoformat(r["day"]),
            low=r["low"], high=r["high"], volume=r["volume"], vwap=r["vwap"],
            histogram=_unpack(r["histogram"]), delta=r["delta"],
            unsigned_volume=r["unsigned_volume"], minute_count=r["minute_count"],
            rth_only=bool(r["rth_only"]), flow_method=r["flow_method"],
            session_open=r["session_open"], session_close=r["session_close"],
            low_minute=r["low_minute"], high_minute=r["high_minute"],
            volume_after_low=r["volume_after_low"],
            volume_after_high=r["volume_after_high"],
        )


def group_by_session(bars: Iterable[MinuteBar]) -> dict[date, list[MinuteBar]]:
    """Split a bar stream into trading sessions by Eastern calendar date.

    Grouping on UTC date would split a session at 20:00 ET and merge two halves
    of different days, which silently corrupts every profile built from it.
    """
    out: dict[date, list[MinuteBar]] = {}
    for bar in bars:
        out.setdefault(bar.ts.astimezone(EASTERN).date(), []).append(bar)
    for bars_in_day in out.values():
        bars_in_day.sort(key=lambda b: b.ts)
    return out
