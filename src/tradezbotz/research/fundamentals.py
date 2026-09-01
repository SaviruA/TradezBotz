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

**The large-cap multiples, and why they are here.** P/E, P/FCF and EV/EBITDA are
built despite the argument above, because they are the right tools on a large-cap
universe and that universe is now in scope. The argument above is not withdrawn
-- it is scoped. P/E is undefined for 77% of filers under $100M of revenue and
for 14% of those over $10B, so the same metric is a selection bias at one end of
the market and the standard measure at the other.

`COVERAGE_BY_BAND` below carries the measured availability for each, and
`size_band` / `guard_single_band` exist so a result cannot silently pool bands
whose transaction costs differ by an order of magnitude.

**Forward P/E is not here and cannot be.** It is the one member of the standard
five that no choice of universe unlocks: it needs analyst consensus estimates,
which are not available free at a point in time, and using today's estimates to
judge a past decision is the same back-door lookahead that rules out Yahoo and
Macrotrends for reported figures. Large caps have plenty of analyst coverage;
we have no point-in-time record of what that coverage said. Those are different
problems and only the first one is fixed by moving up the size distribution.
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

#: EBITDA is not a GAAP tag and never will be -- it is a non-GAAP construct, so
#: it has to be assembled from operating income plus depreciation and
#: amortisation. Two D&A tags because filers split roughly 3,034 to 357 between
#: them and neither alone is sufficient.
DA_CONCEPTS = (
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "DepreciationAndAmortization",
)

#: Free cash flow inputs. Operating cash flow is the best-tagged number in all
#: of XBRL -- 98.8% of small filers report it, better than net income -- and
#: capex is the constraint at 65.2%.
OPERATING_CASH_FLOW_CONCEPTS = ("NetCashProvidedByUsedInOperatingActivities",
                                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
CAPEX_CONCEPTS = ("PaymentsToAcquirePropertyPlantAndEquipment",
                  "PaymentsToAcquireProductiveAssets")

#: Enterprise value inputs. Both are balance-sheet quantities, so they are read
#: with `latest` rather than summed over twelve months.
CASH_CONCEPTS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
)
DEBT_CONCEPTS = ("LongTermDebt", "LongTermDebtNoncurrent")
DEBT_CURRENT_CONCEPTS = ("LongTermDebtCurrent", "DebtCurrent")
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
    #: EBITDA and FCF inputs. All None-able, because the tags they need are the
    #: least reliably reported ones in XBRL and a guess here would propagate
    #: into a ratio that looks like a measurement.
    depreciation_amortisation_ttm: float | None = None
    operating_cash_flow_ttm: float | None = None
    capex_ttm: float | None = None
    #: Balance-sheet quantities, read at a point in time rather than summed.
    cash: float | None = None
    debt: float | None = None
    #: True when at least one debt tag was found. Distinguishes "this company
    #: reported no debt" from "this company did not tag its debt", which XBRL
    #: itself does not distinguish and which `debt = 0.0` would silently
    #: conflate -- understating enterprise value for exactly the filers whose
    #: tagging is weakest.
    debt_reported: bool = False

    @property
    def ebitda_ttm(self) -> float | None:
        """Operating income plus D&A.

        The conventional construction, and the reason EV/EBITDA is expensive to
        compute here: EBITDA is non-GAAP, so both halves must be tagged. Only
        39.8% of small filers and 52.9% of the largest have both.
        """
        if self.operating_income_ttm is None or self.depreciation_amortisation_ttm is None:
            return None
        return self.operating_income_ttm + self.depreciation_amortisation_ttm

    @property
    def free_cash_flow_ttm(self) -> float | None:
        """Operating cash flow minus capital expenditure.

        Capex is reported as a positive outflow in XBRL, so it is subtracted.
        Getting that sign backwards would turn a cash-burning company into the
        cheapest name in the screen.
        """
        if self.operating_cash_flow_ttm is None or self.capex_ttm is None:
            return None
        return self.operating_cash_flow_ttm - abs(self.capex_ttm)

    def market_cap(self, price: float) -> float | None:
        if not self.shares_outstanding or price <= 0:
            return None
        return price * self.shares_outstanding

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

    def price_to_earnings(self, price: float) -> float | None:
        """Trailing P/E. Undefined on negative earnings, deliberately.

        Returning None rather than a negative number is the whole point: a
        negative P/E sorts *below* every cheap profitable company, so a naive
        "lowest P/E" screen fills up with loss-makers. On our microcap universe
        that is 77% of the population; on filers over $10B of revenue it is
        14%, which is why this only becomes usable at the large end.
        """
        if not self.net_income_ttm or not self.shares_outstanding:
            return None
        if self.net_income_ttm <= 0 or self.shares_outstanding <= 0 or price <= 0:
            return None
        return (price * self.shares_outstanding) / self.net_income_ttm

    def price_to_free_cash_flow(self, price: float) -> float | None:
        """Market cap over trailing free cash flow.

        The best-covered of the cash-based multiples -- operating cash flow is
        tagged more reliably than net income at every size band -- and the
        hardest of them to manage, since accruals cannot move it.

        None on non-positive FCF for the same reason as P/E: a cash-burning
        company must not sort as the cheapest.
        """
        cap = self.market_cap(price)
        fcf = self.free_cash_flow_ttm
        if cap is None or fcf is None or fcf <= 0:
            return None
        return cap / fcf

    def enterprise_value(self, price: float) -> float | None:
        """Market cap plus debt minus cash.

        Requires `debt_reported`. Treating an untagged balance sheet as
        debt-free would understate EV for precisely the filers least likely to
        tag it, producing a screen that rewards poor disclosure.
        """
        cap = self.market_cap(price)
        if cap is None or not self.debt_reported or self.cash is None:
            return None
        return cap + (self.debt or 0.0) - self.cash

    def ev_to_ebitda(self, price: float) -> float | None:
        """The multiple with the strongest evidence behind it.

        Loughran & Wellman (JFQA 2011) build an enterprise-multiple factor
        earning 5.28% a year; Gray & Vogel (JPM 2012) race the metrics over
        forty years and EBITDA/TEV wins, beating P/E, book-to-market and
        FCF/TEV.

        The evidence is not the constraint here -- coverage is. Measured on the
        SEC frames API for CY2024, this is computable for 13.6% of filers under
        $100M of revenue and 41.9% of those over $10B. Even at the top of the
        market it is a minority, because the largest filers include banks and
        insurers for whom operating income and capex do not mean what this
        formula assumes.
        """
        ev = self.enterprise_value(price)
        ebitda = self.ebitda_ttm
        if ev is None or ebitda is None or ebitda <= 0 or ev <= 0:
            return None
        return ev / ebitda


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

    # Balance-sheet quantities: a position on a date, not a flow over a period,
    # so `latest` rather than `trailing_twelve_months`.
    latest_cash = latest(extract(raw, CASH_CONCEPTS), as_of)
    debt_long = latest(extract(raw, DEBT_CONCEPTS), as_of)
    debt_short = latest(extract(raw, DEBT_CURRENT_CONCEPTS), as_of)
    debt_total = None
    if debt_long is not None or debt_short is not None:
        debt_total = (debt_long.value if debt_long else 0.0) + \
                     (debt_short.value if debt_short else 0.0)

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
        depreciation_amortisation_ttm=trailing_twelve_months(
            extract(raw, DA_CONCEPTS), as_of),
        operating_cash_flow_ttm=trailing_twelve_months(
            extract(raw, OPERATING_CASH_FLOW_CONCEPTS), as_of),
        capex_ttm=trailing_twelve_months(extract(raw, CAPEX_CONCEPTS), as_of),
        cash=latest_cash.value if latest_cash else None,
        debt=debt_total,
        debt_reported=debt_total is not None,
    )


# --- size bands ------------------------------------------------------------------
#
# The five-multiple toolkit is calibrated on large caps and mostly undefined
# below them, so the universe a ratio is computed on is part of the result
# rather than a detail of it. These bands exist so that fact stays in the data.

#: Market-cap floors, in dollars. Conventional US definitions; the boundaries
#: are arbitrary in the way all such boundaries are, which is why the band
#: travels with the result rather than being hardcoded into a filter.
MICRO_CAP_CEILING = 300_000_000
SMALL_CAP_CEILING = 2_000_000_000
MID_CAP_CEILING = 10_000_000_000
LARGE_CAP_CEILING = 200_000_000_000

BAND_MICRO = "micro"
BAND_SMALL = "small"
BAND_MID = "mid"
BAND_LARGE = "large"
BAND_MEGA = "mega"

#: Where each multiple is computable, measured on the SEC frames API for CY2024
#: by revenue band (a proxy for size, since frames carries no market cap).
#: Recorded here because the numbers are the argument and they should not have
#: to be rediscovered.
#:
#:     band              n     P/E defined   FCF    EBITDA   EV/EBITDA
#:     revenue <$100M    1,991     23.4%    61.9%    38.2%     13.6%
#:     $100M - $1B       1,217     49.1%    73.8%    56.2%     33.4%
#:     $1B - $10B        1,217     73.3%    69.2%    62.5%     48.0%
#:     over $10B           427     85.7%    65.8%    52.9%     41.9%
#:
#: Two things in that table are worth more than the headline. First, moving up
#: the size distribution fixes P/E dramatically (23% to 86%) and does NOT fix
#: EV/EBITDA (14% to 42%) -- it remains a minority even among the largest
#: filers, because that band is thick with banks and insurers for whom
#: operating income and capex do not mean what the formula assumes. Second,
#: three of the four metrics PEAK in the $1B-$10B band rather than at the top,
#: so "bigger is better" is false past mid-cap.
COVERAGE_BY_BAND = {
    "revenue <$100M": {"pe": 0.234, "fcf": 0.619, "ebitda": 0.382, "ev_ebitda": 0.136},
    "$100M-$1B": {"pe": 0.491, "fcf": 0.738, "ebitda": 0.562, "ev_ebitda": 0.334},
    "$1B-$10B": {"pe": 0.733, "fcf": 0.692, "ebitda": 0.625, "ev_ebitda": 0.480},
    "over $10B": {"pe": 0.857, "fcf": 0.658, "ebitda": 0.529, "ev_ebitda": 0.419},
}


def size_band(market_cap: float | None) -> str | None:
    """Which size band a market capitalisation falls in."""
    if market_cap is None or market_cap <= 0:
        return None
    if market_cap < MICRO_CAP_CEILING:
        return BAND_MICRO
    if market_cap < SMALL_CAP_CEILING:
        return BAND_SMALL
    if market_cap < MID_CAP_CEILING:
        return BAND_MID
    if market_cap < LARGE_CAP_CEILING:
        return BAND_LARGE
    return BAND_MEGA


def guard_single_band(bands: Sequence[str | None]) -> None:
    """Refuse a study that silently mixes size bands.

    Not pedantry. Transaction costs differ by more than an order of magnitude
    across this range -- our own measured median round trip is 93bps on the
    microcap end against roughly 5bps at the top, and published implementation
    shortfall runs 110.8bps for US small caps against 31.7bps for large. A
    result pooled across bands is a weighted average of two different
    economies, and the weighting is an artefact of which names happened to have
    tags rather than of anything decided.

    The same argument as `_require_one_method` for order-flow classifiers: the
    fix for heterogeneous inputs is to refuse them, not to average them.
    """
    present = {b for b in bands if b is not None}
    if len(present) > 1:
        raise FundamentalsError(
            f"study mixes size bands {sorted(present)}. Costs differ by an "
            "order of magnitude across these, so a pooled result is an average "
            "of different economies weighted by tag availability. Filter to one "
            "band before measuring."
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
