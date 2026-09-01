"""What each symbol in the universe actually is.

Until this existed, a symbol that returned no price data was indistinguishable
from a symbol that never existed. Both landed in the same bucket and both read
as "coverage gap", which made every coverage number a guess.

Alpaca's assets endpoint settles it, but only if both halves of it are used.
The bulk list carries 33,468 US equities including 19,187 inactive ones, and
that is *not* everything: it silently omits recently delisted names that the
per-symbol endpoint still answers for. Using the bulk list alone put 1,001 real
companies -- LEG, CMA, ALTR, ITCI -- into `unknown` and reported 99.5%
survivorship, which is an artefact of the gap and not a fact about the market.
`resolve_unknown` closes it. See `fetch_asset` for the evidence.

The catalog separates four genuinely different situations:

    listed          currently on NYSE / Nasdaq / ARCA / BATS / AMEX
    delisted        was on one of those, no longer trades
    otc             on the OTC market
    unknown         not a US equity Alpaca has ever heard of

**Why OTC matters here.** Measured by probe, not assumed: the `otc` feed returns
403 on this plan, and OTC symbols on the SIP feed return zero bars rather than an
error. So an OTC name does not fail the backfill -- it succeeds and returns
nothing, which is the quietest possible way for a symbol to disappear from a
study.

**Why OTC symbols are NOT simply dropped.** A company listed on Nasdaq in 2018
and relegated to OTC in 2022 is tagged `otc` today, but its 2018 price history is
real and we have it. Excluding on today's tag would delete history we legitimately
hold, which is a survivorship bias introduced by a cleanup step -- exactly the
kind of thing this module exists to prevent. The tag is recorded and reported;
whether to exclude is left to the caller and defaults to no.

**The delisted count is the survivorship number.** `rebalance.universe_warning`
wanted it and had no source. A universe where nothing ever delisted has selected
for survival, and now that can be checked rather than assumed.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import requests

ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"

#: Exchanges whose prints reach the consolidated tape. Everything else is OTC.
LISTED_EXCHANGES = frozenset({"NYSE", "NASDAQ", "ARCA", "BATS", "AMEX"})

LISTED = "listed"
DELISTED = "delisted"
OTC = "otc"
UNKNOWN = "unknown"

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    symbol     TEXT PRIMARY KEY,
    exchange   TEXT NOT NULL,
    status     TEXT NOT NULL,
    tradable   INTEGER NOT NULL,
    name       TEXT,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_exchange ON assets(exchange);
"""


SCHEMA_MISSING = """
CREATE TABLE IF NOT EXISTS assets_absent (
    symbol     TEXT PRIMARY KEY,
    checked_at TEXT NOT NULL
);
"""



class AssetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Asset:
    symbol: str
    exchange: str
    status: str
    tradable: bool
    name: str = ""

    @property
    def classification(self) -> str:
        if self.exchange not in LISTED_EXCHANGES:
            return OTC
        return LISTED if self.status == "active" else DELISTED


class AssetCatalog:
    """Local copy of Alpaca's equity asset list.

    One request returns every asset, so this is cheap to refresh and there is no
    reason to page or cache partially. Refreshed rather than accumulated: an
    asset's status changes over time and the current answer is the one wanted.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.executescript(SCHEMA_MISSING)
        self._conn.commit()
        self._memo: dict[str, str] | None = None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AssetCatalog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def put_many(self, assets: Iterable[Asset]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        rows = [(a.symbol.upper(), a.exchange, a.status, int(a.tradable),
                 a.name, now) for a in assets]
        self._conn.executemany(
            "INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?,?)", rows)
        self._conn.commit()
        self._memo = None
        return len(rows)

    def get(self, symbol: str) -> Asset | None:
        row = self._conn.execute(
            "SELECT * FROM assets WHERE symbol = ?", (symbol.upper(),)
        ).fetchone()
        if not row:
            return None
        return Asset(row["symbol"], row["exchange"], row["status"],
                     bool(row["tradable"]), row["name"] or "")

    def classify(self, symbol: str) -> str:
        """One symbol's classification, or UNKNOWN when absent.

        UNKNOWN is a real answer, not a failure: it means Alpaca has never
        carried this symbol as a US equity, which for a ticker taken off a Form
        4 usually means a parsing artefact or a foreign listing.
        """
        if self._memo is None:
            self._memo = {
                r["symbol"]: Asset(r["symbol"], r["exchange"], r["status"],
                                   bool(r["tradable"])).classification
                for r in self._conn.execute(
                    "SELECT symbol, exchange, status, tradable FROM assets")
            }
        return self._memo.get(symbol.upper(), UNKNOWN)

    def breakdown(self, symbols: Sequence[str]) -> dict[str, int]:
        """Count each classification across a set of symbols."""
        out = {LISTED: 0, DELISTED: 0, OTC: 0, UNKNOWN: 0}
        for s in symbols:
            out[self.classify(s)] += 1
        return out

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0])

    def mark_absent(self, symbol: str) -> None:
        """Record that a direct lookup 404'd, so it is not retried every run.

        Distinct from "not in the bulk list": that only means the list omitted
        it. Absent means the vendor has confirmed it does not know the symbol.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO assets_absent VALUES (?,?)",
            (symbol.upper(), datetime.now(timezone.utc).isoformat()))
        self._conn.commit()

    def known_absent(self) -> set[str]:
        return {r[0] for r in self._conn.execute(
            "SELECT symbol FROM assets_absent")}

    def fetched_at(self) -> str | None:
        row = self._conn.execute("SELECT MAX(fetched_at) FROM assets").fetchone()
        return row[0] if row else None


def fetch_assets(api_key: str | None = None, api_secret: str | None = None,
                 session: requests.Session | None = None) -> list[Asset]:
    """Pull the full US equity asset list in one request."""
    key = api_key or os.environ.get("ALPACA_PAPER_API_KEY", "")
    secret = api_secret or os.environ.get("ALPACA_PAPER_API_SECRET", "")
    if not key or not secret:
        raise AssetError(
            "ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET are not set. "
            "See .env.example."
        )
    sess = session or requests.Session()
    resp = sess.get(
        ASSETS_URL,
        params={"asset_class": "us_equity"},
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                 "Accept-Encoding": "gzip, deflate"},
        timeout=180,
    )
    resp.raise_for_status()
    return [
        Asset(symbol=a["symbol"], exchange=a.get("exchange") or "",
              status=a.get("status") or "", tradable=bool(a.get("tradable")),
              name=a.get("name") or "")
        for a in resp.json()
        if a.get("symbol")
    ]


def fetch_asset(symbol: str, api_key: str | None = None,
                api_secret: str | None = None,
                session: requests.Session | None = None) -> Asset | None:
    """Look one symbol up directly. Returns None on 404.

    **The list endpoint omits assets that this one returns**, which is a vendor
    inconsistency and not a small one. Verified: the list is internally
    consistent -- 14,281 active plus 19,187 inactive equals the 33,468 it
    returns, with nothing missing between them -- yet LEG, CMA, ALTR and ITCI
    are absent from all of it while `/v2/assets/LEG` answers NYSE, inactive.

    The pattern is recency. Those four were acquired during 2025-26; MDSO
    (2019) and GGP (2018) are in the bulk list. So the names the bulk list
    drops are the *recently delisted* ones -- precisely the bucket that decides
    whether a universe has selected for survival. Reading the bulk list alone
    put 1,001 real companies in `unknown` and reported 99.5% survivorship,
    which is not what a decade of microcaps looks like and was an artefact of
    this gap rather than a fact about the market.
    """
    key = api_key or os.environ.get("ALPACA_PAPER_API_KEY", "")
    secret = api_secret or os.environ.get("ALPACA_PAPER_API_SECRET", "")
    sess = session or requests.Session()
    resp = sess.get(
        f"{ASSETS_URL}/{symbol.upper()}",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=30,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    a = resp.json()
    if not a.get("symbol"):
        return None
    return Asset(symbol=a["symbol"], exchange=a.get("exchange") or "",
                 status=a.get("status") or "", tradable=bool(a.get("tradable")),
                 name=a.get("name") or "")


def describe(breakdown: dict[str, int]) -> str:
    """Human-readable universe composition, with the two numbers that matter."""
    total = sum(breakdown.values()) or 1
    lines = [
        f"  {LISTED:<10} {breakdown[LISTED]:>6,}  ({breakdown[LISTED]/total:.1%})"
        "  currently on a listed exchange",
        f"  {DELISTED:<10} {breakdown[DELISTED]:>6,}  ({breakdown[DELISTED]/total:.1%})"
        "  listed once, gone now -- history is real",
        f"  {OTC:<10} {breakdown[OTC]:>6,}  ({breakdown[OTC]/total:.1%})"
        "  no SIP data; the otc feed is 403 on this plan",
        f"  {UNKNOWN:<10} {breakdown[UNKNOWN]:>6,}  ({breakdown[UNKNOWN]/total:.1%})"
        "  never a US equity Alpaca carried",
    ]
    survivors = breakdown[LISTED]
    ever_listed = breakdown[LISTED] + breakdown[DELISTED]
    if ever_listed:
        share = survivors / ever_listed
        lines.append("")
        lines.append(f"  survivorship: {share:.1%} of ever-listed names are still "
                     "trading")
        if share > 0.95:
            lines.append("  WARNING: almost nothing in this universe delisted. "
                         "That is not what a decade of microcaps looks like, so "
                         "the universe has probably selected for survival.")
    return "\n".join(lines)


#: Alpaca allows 200 requests/minute on the free plan; 150 leaves headroom for
#: whatever else the run is doing.
RESOLVE_PER_MINUTE = 150


def resolve_unknown(catalog: AssetCatalog, symbols: Sequence[str], *,
                    session: requests.Session | None = None,
                    limit: int = 0, on_progress=None) -> dict[str, int]:
    """Look up every symbol the bulk list did not carry.

    This is the correction for the vendor gap documented on `fetch_asset`: the
    bulk list drops recently delisted names, and those are exactly the ones a
    survivorship check depends on. One request each, cached permanently, so it
    is a one-off cost that shrinks to nothing on later runs.

    A 404 is recorded rather than retried. "The vendor confirms it does not
    know this symbol" and "the bulk list happened not to include it" are
    different facts and only the first is stable.
    """
    import time

    sess = session or requests.Session()
    absent = catalog.known_absent()
    todo = [s for s in symbols
            if catalog.classify(s) == UNKNOWN and s.upper() not in absent]
    if limit:
        todo = todo[:limit]

    stats = {"checked": 0, "resolved": 0, "absent": 0, "failed": 0}
    gap = 60.0 / RESOLVE_PER_MINUTE
    for i, symbol in enumerate(todo):
        started = time.monotonic()
        try:
            asset = fetch_asset(symbol, session=sess)
        except Exception:  # noqa: BLE001 - one bad symbol must not end the pass
            stats["failed"] += 1
            continue
        stats["checked"] += 1
        if asset is None:
            catalog.mark_absent(symbol)
            stats["absent"] += 1
        else:
            catalog.put_many([asset])
            stats["resolved"] += 1
        if on_progress and (i + 1) % 100 == 0:
            on_progress(i + 1, len(todo), stats)
        wait = gap - (time.monotonic() - started)
        if wait > 0:
            time.sleep(wait)
    return stats
