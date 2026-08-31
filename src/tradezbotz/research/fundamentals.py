"""Company fundamentals from SEC XBRL, point-in-time.

The first signal family here that is not price or event driven.

**Why the SEC rather than a data vendor.** Every XBRL fact carries a `filed`
date -- the day it became public. That is the same guarantee `observed_at` gives
us, and it is what display sites cannot offer: Yahoo and Macrotrends show
*restated* figures, so a company's 2019 revenue as displayed today may have been
revised in 2021. Using that to evaluate a 2019 decision is lookahead through the
back door, and nothing in the number itself would reveal it. `as_of` filtering
here is not a nicety; it is the whole reason to use the primary source.

**Why price-to-sales rather than price-to-earnings.** Measured against the SEC
frames API for CY2024:

    all filers                    6,051   negative net income  3,056  (50.5%)
    revenue under $100M           1,056   negative net income    781  (74.0%)

P/E is mathematically undefined for three-quarters of the small-cap universe,
and insider buying concentrates in exactly those names. A P/E screen would
silently discard most of our population, and the survivors would be the
profitable minority -- a selection bias wearing a filter's clothing. Revenue
survives where earnings do not, which is the entire argument for P/S.

The known weakness of P/S is that it ignores margins: a 2% net margin business
and a 40% one look identical on it. That is why gross margin is computed
alongside rather than left implicit.

**On PEG.** The textbook version divides P/E by *analyst consensus* growth, and
is unusable here twice over: it inherits P/E's undefinedness, and point-in-time
analyst estimates are not available free, so any historical use would be
lookahead again. `value_growth_score` keeps the idea -- valuation relative to
growth -- with P/S over trailing revenue growth, both computable from filings
alone. It is a different number from PEG and is deliberately not called one.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

import requests

XBRL_BASE = "https://data.sec.gov/api/xbrl"

#: The SEC asks for no more than 10 requests/second and blocks clients that
#: ignore it. 8 leaves headroom, matching the EDGAR client.
MAX_REQUESTS_PER_SECOND = 8

#: Concepts are tried in order; filers tag the same economic quantity under
#: different names, and a missing tag is far more common than a wrong one.
#: `Revenues` alone misses most modern filers, who use the 606 tag.
REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
GROSS_PROFIT_CONCEPTS = ("GrossProfit",)
NET_INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
OPERATING_INCOME_CONCEPTS = ("OperatingIncomeLoss",)
SHARES_CONCEPTS = (
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)

#: A quarter's worth of days, used to assemble a trailing twelve months from
#: quarterly facts without assuming they arrive on a tidy schedule.
TTM_DAYS = 365


class FundamentalsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fact:
    """One reported value, with the date it became public."""

    value: float
    start: date | None
    end: date
    filed: date
    form: str
    #: Frame label such as CY2024Q1, when the SEC assigns one. Absent for many
    #: small filers, so never rely on it being present.
    frame: str | None = None

    @property
    def duration_days(self) -> int | None:
        return (self.end - self.start).days if self.start else None

    @property
    def is_quarterly(self) -> bool:
        d = self.duration_days
        return d is not None and 60 <= d <= 120

    @property
    def is_annual(self) -> bool:
        d = self.duration_days
        return d is not None and 300 <= d <= 400


class XbrlClient:
    """Fetches company facts, self-limiting to the SEC's published rate."""

    def __init__(self, user_agent: str | None = None,
                 session: requests.Session | None = None) -> None:
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT", "")
        if not self.user_agent or "@" not in self.user_agent:
            raise FundamentalsError(
                "SEC_USER_AGENT must be set to a contact string containing a "
                "real email address. The SEC blocks clients without one."
            )
        self.session = session or requests.Session()
        self._last = 0.0

    def _throttle(self) -> None:
        gap = 1.0 / MAX_REQUESTS_PER_SECOND
        wait = gap - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def company_facts(self, cik: str | int) -> dict:
        self._throttle()
        cik_padded = f"{int(cik):010d}"
        resp = self.session.get(
            f"{XBRL_BASE}/companyfacts/CIK{cik_padded}.json",
            headers={"User-Agent": self.user_agent,
                     "Accept-Encoding": "gzip, deflate"},
            timeout=60,
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()


def extract(facts: dict, concepts: Sequence[str],
            unit: str = "USD") -> list[Fact]:
    """Pull every reported observation for the first concept that exists.

    Concepts are tried in order rather than merged. Two tags for the same
    quantity are not guaranteed to agree -- a filer switching from `Revenues` to
    the 606 tag may restate -- and silently concatenating them would produce a
    series that never existed on any filing.
    """
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    dei = (facts.get("facts") or {}).get("dei") or {}
    for concept in concepts:
        node = gaap.get(concept) or dei.get(concept)
        if not node:
            continue
        units = node.get("units") or {}
        rows = units.get(unit) or units.get("shares") or []
        if not rows:
            continue
        out: list[Fact] = []
        for r in rows:
            try:
                out.append(Fact(
                    value=float(r["val"]),
                    start=date.fromisoformat(r["start"]) if r.get("start") else None,
                    end=date.fromisoformat(r["end"]),
                    filed=date.fromisoformat(r["filed"]),
                    form=r.get("form", ""),
                    frame=r.get("frame"),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        if out:
            return sorted(out, key=lambda f: (f.end, f.filed))
    return []


def visible(facts: Sequence[Fact], as_of: date) -> list[Fact]:
    """Only what had actually been filed by `as_of`.

    The single most important function in this module. Without it every ratio
    below is contaminated by figures that did not exist at the decision point,
    including restatements of periods long past.
    """
    return [f for f in facts if f.filed <= as_of]


def latest(facts: Sequence[Fact], as_of: date) -> Fact | None:
    """Most recently *ending* period visible at `as_of`.

    Ties on `end` are broken by the later `filed`, which is the restatement:
    if a period was reported twice, the version knowable at `as_of` is the most
    recent one filed by then.
    """
    seen = visible(facts, as_of)
    return max(seen, key=lambda f: (f.end, f.filed)) if seen else None


def trailing_twelve_months(facts: Sequence[Fact], as_of: date) -> float | None:
    """Sum the last four quarters, or take the last annual figure.

    Annual filings are preferred when one covers a period ending within the last
    year: they are audited, and stitching four quarters together across a
    restatement can double-count.
    """
    seen = visible(facts, as_of)
    if not seen:
        return None

    annual = [f for f in seen if f.is_annual]
    if annual:
        newest = max(annual, key=lambda f: (f.end, f.filed))
        if (as_of - newest.end).days <= TTM_DAYS:
            return newest.value

    quarters: dict[date, Fact] = {}
    for f in seen:
        if not f.is_quarterly:
            continue
        # Keep the latest filing for each period end -- the restated value is
        # what was knowable most recently.
        prior = quarters.get(f.end)
        if prior is None or f.filed > prior.filed:
            quarters[f.end] = f
    if len(quarters) < 4:
        return None
    last_four = sorted(quarters.values(), key=lambda f: f.end)[-4:]
    if (as_of - last_four[-1].end).days > TTM_DAYS:
        return None
    return sum(f.value for f in last_four)


# --- ratios ---------------------------------------------------------------------

@dataclass(frozen=True)
class Snapshot:
    """What was knowable about a company's fundamentals on one date."""

    cik: str
    as_of: date
    revenue_ttm: float | None
    gross_profit_ttm: float | None
    net_income_ttm: float | None
    operating_income_ttm: float | None
    shares_outstanding: float | None
    revenue_prior_ttm: float | None = None

    @property
    def gross_margin(self) -> float | None:
        """The margin P/S ignores. A 2% and a 40% net margin business look
        identical on price-to-sales, which is why this travels with it."""
        if not self.revenue_ttm or self.gross_profit_ttm is None:
            return None
        return self.gross_profit_ttm / self.revenue_ttm if self.revenue_ttm > 0 else None

    @property
    def revenue_growth(self) -> float | None:
        """Trailing year-over-year revenue growth, as a fraction."""
        if not self.revenue_ttm or not self.revenue_prior_ttm:
            return None
        if self.revenue_prior_ttm <= 0:
            return None
        return self.revenue_ttm / self.revenue_prior_ttm - 1.0

    @property
    def profitable(self) -> bool | None:
        if self.net_income_ttm is None:
            return None
        return self.net_income_ttm > 0

    def price_to_sales(self, price: float) -> float | None:
        """Market capitalisation over trailing revenue.

        Needs shares outstanding, which small filers tag inconsistently -- hence
        several fallback concepts and a None rather than a guess.
        """
        if not self.revenue_ttm or not self.shares_outstanding:
            return None
        if self.revenue_ttm <= 0 or self.shares_outstanding <= 0 or price <= 0:
            return None
        return (price * self.shares_outstanding) / self.revenue_ttm

    def value_growth_score(self, price: float) -> float | None:
        """P/S divided by trailing revenue growth, in percent. Lower is cheaper
        per unit of growth.

        Deliberately **not** called PEG. PEG uses P/E over *forward analyst*
        growth: undefined for the 74% of small caps with negative earnings, and
        unavailable point-in-time at any price we would pay. This keeps the idea
        and changes both inputs, so it is a different number and should not be
        compared to published PEG thresholds like "below 1.0 is cheap".

        Returns None on non-positive growth: a shrinking company has no
        meaningful price-per-unit-of-growth, and dividing by a negative would
        rank the worst names as the cheapest.
        """
        ps = self.price_to_sales(price)
        growth = self.revenue_growth
        if ps is None or growth is None or growth <= 0:
            return None
        return ps / (growth * 100.0)


def snapshot(client: XbrlClient, cik: str | int, as_of: date,
             facts: dict | None = None) -> Snapshot:
    """Everything knowable about one company's fundamentals on one date."""
    raw = facts if facts is not None else client.company_facts(cik)
    if not raw:
        return Snapshot(str(cik), as_of, None, None, None, None, None)

    revenue = extract(raw, REVENUE_CONCEPTS)
    prior_cut = date(as_of.year - 1, as_of.month, as_of.day) if as_of.month != 2 or as_of.day != 29 else date(as_of.year - 1, 2, 28)

    shares_facts = extract(raw, SHARES_CONCEPTS, unit="shares")
    latest_shares = latest(shares_facts, as_of)

    return Snapshot(
        cik=str(cik),
        as_of=as_of,
        revenue_ttm=trailing_twelve_months(revenue, as_of),
        gross_profit_ttm=trailing_twelve_months(extract(raw, GROSS_PROFIT_CONCEPTS), as_of),
        net_income_ttm=trailing_twelve_months(extract(raw, NET_INCOME_CONCEPTS), as_of),
        operating_income_ttm=trailing_twelve_months(
            extract(raw, OPERATING_INCOME_CONCEPTS), as_of),
        shares_outstanding=latest_shares.value if latest_shares else None,
        revenue_prior_ttm=trailing_twelve_months(revenue, prior_cut),
    )


def margin_compressing(client: XbrlClient, cik: str | int, as_of: date,
                       quarters: int = 4, facts: dict | None = None) -> bool:
    """Whether gross margin has declined over the last `quarters` reports.

    The bear-case signal: deteriorating unit economics before it reaches the
    headline numbers. Strictly monotonic decline is required -- a noisy series
    that happens to end lower is not compression, and treating it as such would
    fire on most companies most of the time.
    """
    raw = facts if facts is not None else client.company_facts(cik)
    if not raw:
        return False
    rev = {f.end: f for f in visible(extract(raw, REVENUE_CONCEPTS), as_of)
           if f.is_quarterly}
    gp = {f.end: f for f in visible(extract(raw, GROSS_PROFIT_CONCEPTS), as_of)
          if f.is_quarterly}
    ends = sorted(set(rev) & set(gp))[-quarters:]
    if len(ends) < quarters:
        return False
    margins = [gp[e].value / rev[e].value for e in ends if rev[e].value > 0]
    if len(margins) < quarters:
        return False
    return all(b < a for a, b in zip(margins, margins[1:]))
