"""Reddit mention snapshots from ApeWisdom.

https://apewisdom.io/api/ -- free, no authentication, covers r/wallstreetbets,
r/stocks, r/options and related communities.

**This source has no history.** Verified against the live API: `/ticker/{sym}`
and `/history/{sym}` both return empty lists, and the filter endpoint serves a
current snapshot with a single 24-hour comparison (`mentions_24h_ago`,
`rank_24h_ago`). Nothing older is retrievable at any price.

So sentiment cannot be backtested from this source -- it can only be
*accumulated*. Every day without a collector is a day of history that is gone
permanently, which is the whole argument for running this early even though the
signal cannot be evaluated for months.

The upside of building the archive ourselves is that the timestamps are
honest: `observed_at` is the moment we fetched, so a future backtest inherits
the point-in-time property that purchased sentiment history usually lacks.

Two cautions carried forward for whoever eventually tests this:

* **Reverse causality dominates.** NVDA has 303 mentions partly *because* it
  moved. Any evaluation must control for contemporaneous and lagged returns, or
  it will rediscover that returns predict returns.
* **Compute across the whole universe.** Scoring only the tickers that spiked is
  selection on the outcome.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Sequence

import requests

from .eventstore import Event

API_BASE = "https://apewisdom.io/api/v1.0/filter"

SOURCE = "apewisdom"
KIND = "sentiment_snapshot"

#: Communities worth recording. "all-stocks" aggregates the equity subreddits;
#: the individual ones are kept because a mention concentrated in one community
#: is a different signal from one spread across several.
DEFAULT_FILTERS: tuple[str, ...] = ("all-stocks", "wallstreetbets", "stocks")

#: No rate limit is published, so be conservative. A full pass is ~7 pages per
#: filter, and this runs a few times a day at most.
REQUEST_INTERVAL_SECONDS = 1.0

#: Pagination guard: the API reports its own page count, but a malformed
#: response must not spin forever.
MAX_PAGES = 25


class ApeWisdomError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mention:
    ticker: str
    name: str
    rank: int
    mentions: int
    upvotes: int
    rank_24h_ago: int | None
    mentions_24h_ago: int | None
    filter_name: str

    @property
    def mention_change(self) -> int | None:
        if self.mentions_24h_ago is None:
            return None
        return self.mentions - self.mentions_24h_ago


class ApeWisdomClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        interval: float = REQUEST_INTERVAL_SECONDS,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent", "TradezBotz research (https://github.com/SaviruA/TradezBotz)"
        )
        self.interval = interval
        self._last = 0.0

    def _get(self, url: str) -> dict:
        gap = time.monotonic() - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        resp = self.session.get(url, timeout=30)
        self._last = time.monotonic()
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise ApeWisdomError(f"unexpected payload shape from {url}")
        return body

    def mentions(self, filter_name: str) -> list[Mention]:
        """Every ranked ticker for one community, across all pages."""
        out: list[Mention] = []
        page = 1
        while page <= MAX_PAGES:
            url = f"{API_BASE}/{filter_name}"
            if page > 1:
                url = f"{url}/page/{page}"
            body = self._get(url)
            for row in body.get("results") or []:
                parsed = _parse_row(row, filter_name)
                if parsed:
                    out.append(parsed)
            total_pages = int(body.get("pages") or 1)
            if page >= total_pages:
                break
            page += 1
        return out


def _int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_row(row: dict, filter_name: str) -> Mention | None:
    ticker = str(row.get("ticker") or "").strip().upper()
    mentions = _int(row.get("mentions"))
    rank = _int(row.get("rank"))
    if not ticker or mentions is None or rank is None:
        return None
    import html

    return Mention(
        ticker=ticker,
        # Names arrive HTML-escaped, e.g. "SPDR S&amp;P 500 ETF Trust".
        name=html.unescape(str(row.get("name") or "")).strip(),
        rank=rank,
        mentions=mentions,
        upvotes=_int(row.get("upvotes")) or 0,
        rank_24h_ago=_int(row.get("rank_24h_ago")),
        mentions_24h_ago=_int(row.get("mentions_24h_ago")),
        filter_name=filter_name,
    )


def to_events(
    mentions: Sequence[Mention], observed_at: datetime | None = None
) -> Iterator[Event]:
    """Turn one poll into point-in-time events.

    `observed_at` is the fetch time, because that is genuinely when this became
    knowable to us. `occurred_at` is deliberately left unset: a mention count is
    a trailing aggregate over an undisclosed window, so claiming a moment for it
    would be inventing precision the source does not provide.

    Identity is bucketed to the hour so repeated polls within an hour dedupe
    rather than filling the store with near-identical rows.
    """
    observed = observed_at or datetime.now(timezone.utc)
    bucket = observed.strftime("%Y%m%dT%H")
    for m in mentions:
        yield Event(
            source=SOURCE,
            external_id=f"{m.filter_name}:{m.ticker}:{bucket}",
            kind=KIND,
            symbol=m.ticker,
            observed_at=observed,
            occurred_at=None,
            payload={
                "filter": m.filter_name,
                "name": m.name,
                "rank": m.rank,
                "mentions": m.mentions,
                "upvotes": m.upvotes,
                "rank_24h_ago": m.rank_24h_ago,
                "mentions_24h_ago": m.mentions_24h_ago,
                "mention_change": m.mention_change,
            },
        )
