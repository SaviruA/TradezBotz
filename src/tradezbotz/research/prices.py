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
from datetime import date, datetime, time as dtime, timedelta, timezone
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

#: Alpaca refuses consolidated-feed data newer than this. A request whose end
#: is today reaches into it and is refused wholesale, so windows are clamped.
SIP_DELAY_MINUTES = 15


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
    #: Volume-weighted average price for the session, when the source reports
    #: one. Alpaca returns it per bar; Massive does not always. Anchored VWAP is
    #: exact when this is present and falls back to the (H+L+C)/3 typical-price
    #: approximation when it is None.
    vwap: float | None = None


@dataclass(frozen=True)
class Series:
    """Daily bars for one symbol, plus what the source knows about coverage."""

    symbol: str
    bars: tuple[Bar, ...]
    requested_start: date
    requested_end: date
    is_active: bool | None = None
    #: Which adjustment basis these bars are on. Carried on the series rather
    #: than tracked by the caller because mixing bases inside one return
    #: calculation does not produce a small error -- it fabricates a return on
    #: every ex-dividend date.
    basis: str = "price"

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


#: Split-adjusted only. This is what a share actually printed at, and it is the
#: basis Massive and Yahoo serve. Never revised except by a split, so a cached
#: history stays reproducible.
BASIS_PRICE = "price"

#: Split *and* dividend adjusted -- a total-return series, which is what a
#: holder actually earned. Revised every time a distribution is paid, so a
#: cached history drifts from the vendor over time. Kept alongside rather than
#: instead of BASIS_PRICE: on a 5-day horizon the difference is ~0.03% for a
#: typical name but ~0.2% for a REIT or BDC, and that error is systematic within
#: dividend payers rather than random across them. Reporting both makes a result
#: that depends on the choice visible instead of silent.
BASIS_TOTAL = "total"

BASES = (BASIS_PRICE, BASIS_TOTAL)

#: Alpaca's `adjustment` parameter for each basis.
ALPACA_ADJUSTMENT = {BASIS_PRICE: "split", BASIS_TOTAL: "all"}

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    day    TEXT NOT NULL,
    basis  TEXT NOT NULL DEFAULT 'price',
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, day, basis)
);
CREATE TABLE IF NOT EXISTS fetches (
    symbol     TEXT NOT NULL,
    basis      TEXT NOT NULL DEFAULT 'price',
    start_day  TEXT NOT NULL,
    end_day    TEXT NOT NULL,
    is_active  INTEGER,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, basis)
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
        self._migrate()
        self._conn.executescript(CACHE_SCHEMA)
        self._conn.commit()

    def _migrate(self) -> None:
        """Add the `basis` dimension to a cache written before it existed.

        Everything already stored came from Massive, which serves a price-only
        (split-adjusted) series -- established by three-way comparison, where
        Massive and Yahoo agreed to 0.00% while Alpaca sat below both by the
        accumulated dividend. So existing rows are stamped BASIS_PRICE.

        SQLite cannot alter a primary key, so the tables are rebuilt. Worth the
        care: the CI cache holds ~2,200 symbols that would take many hours to
        refetch at Massive's 5 requests/minute.
        """
        have = {r[0] for r in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("bars", "fetches"):
            if table not in have:
                continue
            cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")]
            if "basis" in cols:
                continue
            keep = ", ".join(cols)
            self._conn.execute(f"ALTER TABLE {table} RENAME TO {table}_pre_basis")
            self._conn.executescript(CACHE_SCHEMA)
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} ({keep}, basis) "
                f"SELECT {keep}, ? FROM {table}_pre_basis",
                (BASIS_PRICE,),
            )
            self._conn.execute(f"DROP TABLE {table}_pre_basis")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def covered(self, symbol: str, start: date, end: date,
                basis: str = BASIS_PRICE) -> bool:
        row = self._conn.execute(
            "SELECT start_day, end_day FROM fetches WHERE symbol = ? AND basis = ?",
            (symbol, basis),
        ).fetchone()
        if not row:
            return False
        return row["start_day"] <= start.isoformat() and row["end_day"] >= end.isoformat()

    def get(self, symbol: str, start: date, end: date,
            basis: str = BASIS_PRICE) -> Series:
        rows = self._conn.execute(
            "SELECT * FROM bars WHERE symbol = ? AND basis = ? "
            "AND day BETWEEN ? AND ? ORDER BY day",
            (symbol, basis, start.isoformat(), end.isoformat()),
        ).fetchall()
        meta = self._conn.execute(
            "SELECT is_active FROM fetches WHERE symbol = ? AND basis = ?",
            (symbol, basis),
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
            basis=basis,
        )

    def put(self, series: Series) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO bars (symbol, day, basis, open, high, low, "
            "close, volume) VALUES (?,?,?,?,?,?,?,?)",
            [
                (series.symbol, b.day.isoformat(), series.basis,
                 b.open, b.high, b.low, b.close, b.volume)
                for b in series.bars
            ],
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO fetches (symbol, basis, start_day, end_day, "
            "is_active, fetched_at) VALUES (?,?,?,?,?,?)",
            (
                series.symbol,
                series.basis,
                series.requested_start.isoformat(),
                series.requested_end.isoformat(),
                None if series.is_active is None else int(series.is_active),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def symbols(self, basis: str | None = None) -> list[str]:
        """Symbols held. With no basis, any basis counts."""
        if basis is None:
            return [r[0] for r in self._conn.execute(
                "SELECT DISTINCT symbol FROM fetches ORDER BY symbol")]
        return [r[0] for r in self._conn.execute(
            "SELECT symbol FROM fetches WHERE basis = ? ORDER BY symbol", (basis,))]

    def bases(self, symbol: str) -> list[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT basis FROM fetches WHERE symbol = ? ORDER BY basis", (symbol,))]

    def earliest_day(self, basis: str | None = None) -> date | None:
        """The oldest bar held, which bounds what any backtest can measure.

        An event needs prior sessions before it can be enriched or costed --
        EDGE wants 21, a 200-day moving average wants 200 -- and no amount of
        care recovers history the vendor does not sell. Events closer to this
        date than the deepest indicator's lookback are therefore not
        measurable, and `measure` drops them explicitly rather than letting
        them silently charge a fallback constant.
        """
        sql = "SELECT MIN(day) FROM bars"
        params: tuple = ()
        if basis is not None:
            sql += " WHERE basis = ?"
            params = (basis,)
        row = self._conn.execute(sql, params).fetchone()
        return date.fromisoformat(row[0]) if row and row[0] else None

    def first_days(self, basis: str | None = None) -> dict[str, date]:
        """Each symbol's oldest bar, which is the only floor that means anything.

        A single global minimum is nearly useless here: cached symbols begin in
        2016, 2019, 2022, 2023 and 2024, so a floor set from the earliest of
        them lets a 2024 listing's first-week events through with no prior
        history at all -- and those are exactly the events that get charged a
        fallback cost constant instead of a measured spread.
        """
        sql = "SELECT symbol, MIN(day) FROM bars"
        params: tuple = ()
        if basis is not None:
            sql += " WHERE basis = ?"
            params = (basis,)
        sql += " GROUP BY symbol"
        return {r[0]: date.fromisoformat(r[1])
                for r in self._conn.execute(sql, params) if r[1]}


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


ALPACA_DATA_BASE = "https://data.alpaca.markets"

#: Alpaca's free Basic plan allows 200 requests/minute -- 40x Massive's budget.
ALPACA_REQUESTS_PER_MINUTE = 200


class AlpacaPriceSource:
    """Alpaca daily bars, intended as a CROSS-CHECK rather than a primary source.

    Attractive on paper: 7+ years of history against Massive's 2, and 200
    requests/minute against 5. But the free feed carries only IEX prints --
    roughly 2.5% of US volume -- and its documented failure mode is ghost prices
    on illiquid names, where IEX can show a last trade materially away from where
    the stock actually printed elsewhere.

    That failure lands squarely on this project's population: insider purchases
    skew small-cap, which is exactly where a single-venue feed is least
    trustworthy. So this is not a substitute for Massive and not a safe way to
    extend history.

    Its value is disagreement. Where two independent sources diverge on the same
    bar, that divergence is itself evidence the name is too thinly traded to
    trust -- information we otherwise would not have. See `crosscheck.compare`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        cache: PriceCache | None = None,
        per_minute: int = ALPACA_REQUESTS_PER_MINUTE,
        session: requests.Session | None = None,
        feed: str = "sip",
        basis: str = BASIS_PRICE,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_PAPER_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_PAPER_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise PriceError(
                "ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET are not set. "
                "See .env.example."
            )
        self.cache = cache
        self.limiter = RateLimiter(per_minute)
        self.session = session or requests.Session()
        self.feed = feed
        if basis not in ALPACA_ADJUSTMENT:
            raise PriceError(f"unknown basis {basis!r}; expected one of {BASES}")
        self.basis = basis

    def _end_param(self, end: date) -> str:
        """Clamp the window out of the SIP embargo.

        The consolidated feed refuses data inside the last 15 minutes, and a
        request whose end is *today* reaches into that window -- returning 403
        for the whole request, not just the embargoed part. Since callers
        routinely pass `date.today()`, without this every such call fails.

        Only applies to `sip`. IEX has no embargo, which is why this was not
        needed while IEX was the default.
        """
        requested = datetime.combine(end, dtime(0, 0), tzinfo=timezone.utc)
        if self.feed != "sip":
            return requested.isoformat().replace("+00:00", "Z")
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=SIP_DELAY_MINUTES + 1)
        return min(requested, cutoff).isoformat().replace("+00:00", "Z")

    def daily_bars(self, symbol: str, start: date, end: date) -> Series:
        symbol = symbol.upper()
        if self.cache and self.cache.covered(symbol, start, end, self.basis):
            return self.cache.get(symbol, start, end, self.basis)

        bars: list[Bar] = []
        page: str | None = None
        end_param = self._end_param(end)
        while True:
            self.limiter.acquire()
            params = {
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end_param,
                # The basis is explicit, never assumed. `split` is price-only
                # and matches what Massive and Yahoo serve; `all` adds dividend
                # adjustment. Defaulting to `all` while treating the result as
                # comparable to Massive is exactly the mistake that produced the
                # bogus "54% agree" finding.
                "adjustment": ALPACA_ADJUSTMENT[self.basis],
                "feed": self.feed,
                "limit": 10000,
            }
            if page:
                params["page_token"] = page
            resp = self.session.get(
                f"{ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars",
                params=params,
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                },
                timeout=45,
            )
            if resp.status_code in (401, 403):
                raise PriceError(f"Alpaca rejected the credentials ({resp.status_code}).")
            resp.raise_for_status()
            body = resp.json()
            for r in body.get("bars") or []:
                bars.append(
                    Bar(
                        day=datetime.fromisoformat(
                            r["t"].replace("Z", "+00:00")
                        ).date(),
                        open=float(r["o"]), high=float(r["h"]), low=float(r["l"]),
                        close=float(r["c"]), volume=float(r.get("v", 0.0)),
                    )
                )
            page = body.get("next_page_token")
            if not page:
                break

        # Alpaca exposes no delisting flag, so coverage is left unknown rather
        # than guessed. Massive remains the authority on whether a ticker lives.
        series = Series(
            symbol=symbol, bars=tuple(bars),
            requested_start=start, requested_end=end, is_active=None,
            basis=self.basis,
        )
        if self.cache:
            self.cache.put(series)
        return series


#: Yahoo, reached through OpenBB. Requested per call rather than held open, so a
#: missing optional dependency fails at use rather than at import.
OPENBB_PROVIDER = "yfinance"


class OpenBBPriceSource:
    """A third price opinion, via OpenBB's provider layer.

    Exists to settle a question two sources cannot. Massive and Alpaca disagree
    materially on corporate-action adjustment, and disagreement alone never says
    which one is wrong -- with two sources the best available answer is "one of
    these is broken." A third independent vendor turns that into an answer.

    It is not a candidate for system of record. Yahoo is a derived, unaudited,
    terms-of-service-grey feed and we would not build returns on it. Its value
    here is precisely that it is *independent* of the other two: it shares no
    infrastructure with either, so agreement between Yahoo and one vendor is
    real evidence about the other.

    OpenBB rather than raw yfinance because the same interface reaches nine
    providers for this endpoint. When one more reference is wanted, it is a
    provider string rather than another adapter.

    Optional dependency: `pip install openbb-yfinance`.
    """

    def __init__(self, provider: str = OPENBB_PROVIDER,
                 cache: "PriceCache | None" = None) -> None:
        self.provider = provider
        self.cache = cache

    def _fetcher(self):
        if self.provider != "yfinance":
            raise PriceError(
                f"provider {self.provider!r} needs its own OpenBB package and a "
                "fetcher mapping; only yfinance is wired up."
            )
        try:
            from openbb_yfinance.models.equity_historical import (
                YFinanceEquityHistoricalFetcher,
            )
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise PriceError(
                "OpenBB provider missing. Install it with:\n"
                "    pip install openbb-yfinance\n"
                "It is an optional extra: the pipeline runs without it, and only "
                "three-way crosscheck needs it."
            ) from exc
        return YFinanceEquityHistoricalFetcher

    def daily_bars(self, symbol: str, start: date, end: date) -> Series:
        import asyncio

        symbol = symbol.upper()
        fetcher = self._fetcher()
        rows = asyncio.run(
            fetcher.fetch_data(
                {
                    "symbol": symbol,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "interval": "1d",
                },
                {},
            )
        )
        bars = [
            Bar(
                day=r.date if isinstance(r.date, date) else r.date.date(),
                open=float(r.open), high=float(r.high), low=float(r.low),
                close=float(r.close), volume=float(r.volume or 0.0),
            )
            for r in rows
        ]
        bars.sort(key=lambda b: b.day)
        # Yahoo reports no delisting flag; leaving this unknown rather than
        # guessing keeps Massive the sole authority on whether a ticker lives.
        return Series(
            symbol=symbol, bars=tuple(bars),
            requested_start=start, requested_end=end, is_active=None,
        )


class DualBasisSource:
    """Fetches both adjustment bases for every symbol, caching each separately.

    Satisfies `PriceSource`, so `BackfillRunner` needs no change: it counts one
    symbol done when both bases have landed.

    **Why both rather than a choice.** Price-only is what a share printed at and
    is never revised; total return is what a holder actually earned. On a 5-day
    horizon they differ by ~0.03% for a typical name and ~0.2% for a REIT or BDC
    -- small, but systematic *within* dividend payers rather than random across
    them, which is the kind of error that looks like a finding. Storing both
    lets every result be reported on both, so a conclusion that depends on the
    choice is visible rather than silent. It is the same reasoning that makes
    `backtest` report winsorised returns beside raw ones.

    **Why this is affordable at all.** Two requests per symbol against Alpaca's
    200/minute is still roughly twenty times faster than one request against
    Massive's 5/minute -- the difference between a fourteen-hour backfill and a
    twenty-minute one.
    """

    def __init__(self, sources: dict[str, PriceSource],
                 primary: str = BASIS_PRICE) -> None:
        missing = [b for b in BASES if b not in sources]
        if missing:
            raise PriceError(f"no source supplied for basis {missing}")
        self.sources = sources
        self.primary = primary

    @classmethod
    def alpaca(cls, cache: PriceCache | None = None,
               per_minute: int = ALPACA_REQUESTS_PER_MINUTE,
               **kwargs) -> "DualBasisSource":
        """Both bases from Alpaca, sharing one rate limiter.

        The limiter is shared deliberately: two sources each self-limiting to
        200/min would together run at 400/min and be throttled.
        """
        limiter = RateLimiter(per_minute)
        sources: dict[str, PriceSource] = {}
        for basis in BASES:
            src = AlpacaPriceSource(cache=cache, basis=basis, **kwargs)
            src.limiter = limiter
            sources[basis] = src
        return cls(sources)

    def daily_bars(self, symbol: str, start: date, end: date) -> Series:
        """Fetch every basis; return the primary one.

        A failure on any basis propagates rather than being swallowed. Half a
        symbol is worse than none: the runner would mark it done and the missing
        basis would surface later as a silent coverage hole.
        """
        out: dict[str, Series] = {}
        for basis in BASES:
            out[basis] = self.sources[basis].daily_bars(symbol, start, end)
        return out[self.primary]


class CachedOnlySource:
    """Serves bars from the local cache and never reaches a vendor.

    Measurement must be reproducible and must not depend on network access,
    vendor credentials, or a rate limiter. A source that silently fetches on a
    miss would also make a backtest's coverage depend on when it was run, which
    turns "this strategy had 40% coverage" into a statement about the weather.

    A miss returns an empty `Series` rather than raising. That is the honest
    result: the labeller reads it as NO_DATA and the coverage report says how
    much of the population was unmeasurable, which is exactly the number a
    reader needs.
    """

    def __init__(self, cache: PriceCache, basis: str = BASIS_TOTAL) -> None:
        self.cache = cache
        self.basis = basis
        self.hits = 0
        self.misses = 0

    def daily_bars(self, symbol: str, start: date, end: date) -> Series:
        series = self.cache.get(symbol, start, end, basis=self.basis)
        if series.bars:
            self.hits += 1
        else:
            self.misses += 1
        return series

    def summary(self) -> str:
        total = self.hits + self.misses
        return (
            f"price cache ({self.basis}): {self.hits:,} of {total:,} symbols "
            f"served, {self.misses:,} absent"
        )
