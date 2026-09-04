"""CUSIP to ticker, the single mapping standing between us and 13F/13D.

**Why this exists.** A 13F information table identifies each position by CUSIP
and issuer name. It never carries a ticker. `Filing13F.to_events()` therefore
produces events with `symbol=None`, and `HoldingsJoin._index()` skips any event
without a symbol -- so every institutional holding we ingest is structurally
invisible to the backtest, however many of them we collect. The run that first
pulled real 13F data added 124,786 events in a single day and the join still
reported "0 symbols carry a disclosure".

**Why OpenFIGI.** It is free, public, and explicitly accepts CUSIP as an input
identifier. The SEC publishes no CUSIP-to-ticker map of its own: its quarterly
13(f) securities list carries CUSIP and issuer name only, and Form 4 carries
ticker and CIK but no CUSIP, so the two families cannot be joined from SEC data
alone. Note that OpenFIGI will not RETURN third-party identifiers (CUSIP, ISIN,
SEDOL) for licensing reasons -- submitting them as input is fine, which is the
direction we need.

**Why a persistent cache, and why resolution is a separate step from
measurement.** The same architecture as `CachedOnlySource`: a measurement that
reaches the network is not reproducible, its coverage depends on when it was
run, and "this strategy had 40% coverage" becomes a statement about the
weather. So `resolve-cusips` fills the cache as a pipeline step with its own
time budget, and the join reads the cache offline and never fetches.

**Unresolved stays unresolved.** A CUSIP the vendor does not know is recorded as
a miss and reported, never guessed at by issuer-name matching. Fuzzy name
matching across 13F issuer strings would silently attach one company's
disclosures to another company's returns, which is worse than missing data
because nothing downstream would say so.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import requests

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

#: Jobs per HTTP request. OpenFIGI allows 10 unauthenticated and 100 with a
#: key; the smaller figure is used unless a key is present because exceeding it
#: fails the WHOLE batch rather than the excess.
BATCH_UNAUTHENTICATED = 10
BATCH_AUTHENTICATED = 100

#: Requests per minute. Deliberately below the published ceilings -- the cost of
#: being throttled mid-run is a partially filled cache that looks like a vendor
#: gap, and the cache is permanent so there is no hurry.
RATE_UNAUTHENTICATED = 20
RATE_AUTHENTICATED = 200

#: Exchange codes we accept a ticker from. A CUSIP can map to listings on many
#: venues; taking the first row would sometimes return a foreign line whose
#: ticker collides with an unrelated US symbol.
US_EXCHANGES = {"US", "UN", "UQ", "UA", "UR", "UW", "UV", "UP"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cusip_map (
    cusip      TEXT PRIMARY KEY,
    symbol     TEXT,
    name       TEXT,
    resolved_at TEXT NOT NULL,
    -- A miss is RECORDED rather than retried forever. Most unresolved CUSIPs
    -- are delisted issuers or non-equity instruments the vendor genuinely does
    -- not carry, and re-asking nightly would spend the whole rate limit on
    -- questions already answered.
    found      INTEGER NOT NULL
);
"""


class CusipError(RuntimeError):
    pass


@dataclass(frozen=True)
class Resolution:
    cusip: str
    symbol: str | None
    name: str | None

    @property
    def found(self) -> bool:
        return bool(self.symbol)


class CusipCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CusipCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, cusip: str) -> str | None:
        row = self._conn.execute(
            "SELECT symbol FROM cusip_map WHERE cusip = ? AND found = 1",
            (cusip.upper(),)).fetchone()
        return row["symbol"] if row else None

    def mapping(self) -> dict[str, str]:
        """Every resolved CUSIP, for one-pass index building."""
        return {r["cusip"]: r["symbol"] for r in self._conn.execute(
            "SELECT cusip, symbol FROM cusip_map WHERE found = 1")}

    def known(self) -> set[str]:
        """Every CUSIP already asked about, hit or miss."""
        return {r[0] for r in self._conn.execute("SELECT cusip FROM cusip_map")}

    def record(self, resolutions: Iterable[Resolution]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        rows = [(r.cusip.upper(), r.symbol, r.name, now, 1 if r.found else 0)
                for r in resolutions]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT OR REPLACE INTO cusip_map "
            "(cusip, symbol, name, resolved_at, found) VALUES (?, ?, ?, ?, ?)",
            rows)
        self._conn.commit()
        return len(rows)

    def counts(self) -> tuple[int, int]:
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(found), 0) FROM cusip_map").fetchone()
        return int(row[0]), int(row[1])


def _pick_ticker(rows: Sequence[dict]) -> tuple[str | None, str | None]:
    """Choose the US listing, or nothing.

    A CUSIP commonly maps to several venues. Taking rows[0] would sometimes
    return a foreign line whose ticker collides with an unrelated US symbol --
    which attaches one company's disclosures to another company's returns, and
    nothing downstream would report it.
    """
    for row in rows:
        if row.get("exchCode") in US_EXCHANGES and row.get("ticker"):
            return row["ticker"].upper(), row.get("name")
    return None, None


class OpenFigiClient:
    def __init__(self, api_key: str | None = None,
                 session: requests.Session | None = None) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self.batch_size = BATCH_AUTHENTICATED if api_key else BATCH_UNAUTHENTICATED
        per_minute = RATE_AUTHENTICATED if api_key else RATE_UNAUTHENTICATED
        self._min_gap = 60.0 / per_minute
        self._last = 0.0

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key
        return headers

    def resolve(self, cusips: Sequence[str]) -> list[Resolution]:
        """Map one batch. Length must not exceed `batch_size`."""
        if not cusips:
            return []
        if len(cusips) > self.batch_size:
            raise CusipError(
                f"batch of {len(cusips)} exceeds the {self.batch_size} job "
                "limit; OpenFIGI fails the whole request rather than the excess"
            )
        gap = time.monotonic() - self._last
        if gap < self._min_gap:
            time.sleep(self._min_gap - gap)
        body = [{"idType": "ID_CUSIP", "idValue": c} for c in cusips]
        resp = self.session.post(OPENFIGI_URL, headers=self._headers(),
                                 data=json.dumps(body), timeout=30)
        self._last = time.monotonic()
        if resp.status_code == 429:
            raise CusipError("OpenFIGI rate limit hit; lower the request rate")
        resp.raise_for_status()

        payload = resp.json()
        out: list[Resolution] = []
        # Responses are POSITIONAL: result[i] answers body[i]. Zipping against
        # anything else silently attaches one issuer's ticker to another's
        # CUSIP, which is the single worst failure available here.
        for cusip, entry in zip(cusips, payload):
            rows = entry.get("data") or []
            symbol, name = _pick_ticker(rows) if rows else (None, None)
            out.append(Resolution(cusip=cusip, symbol=symbol, name=name))
        return out


def resolve_missing(cache: CusipCache, client: OpenFigiClient,
                    cusips: Iterable[str], *,
                    deadline: float | None = None,
                    on_progress=None) -> dict[str, int]:
    """Fill the cache for CUSIPs it has never been asked about.

    Already-known CUSIPs are skipped whether they hit or missed, because most
    misses are delisted issuers or non-equity instruments the vendor genuinely
    does not carry, and re-asking nightly would spend the entire rate limit on
    questions already answered.
    """
    known = cache.known()
    todo = sorted({c.upper() for c in cusips if c} - known)
    stats = {"asked": 0, "resolved": 0, "missed": 0, "remaining": len(todo)}
    for i in range(0, len(todo), client.batch_size):
        if deadline is not None and time.monotonic() > deadline:
            break
        batch = todo[i:i + client.batch_size]
        results = client.resolve(batch)
        cache.record(results)
        stats["asked"] += len(batch)
        stats["resolved"] += sum(1 for r in results if r.found)
        stats["missed"] += sum(1 for r in results if not r.found)
        stats["remaining"] = len(todo) - stats["asked"]
        if on_progress:
            on_progress(stats)
    return stats
