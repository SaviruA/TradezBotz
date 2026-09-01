"""The backlog, as code.

Every hypothesis we have decided is worth measuring lives here, including the
ones that cannot run yet. That second part is the load-bearing half: a strategy
omitted from a sweep leaves no trace in the output, so the only defence against
silent attrition is a list where removing something is an edit somebody has to
make on purpose.

Three rules govern what goes in:

**An opinion goes in `prior`, never in the decision to include.** "This probably
does not work" is a reason to write it down and be checked against the result,
not a reason to skip the test.

**`blocked_by` is mechanical only.** Data that does not exist yet, a join not
written yet. Never "looks weak". Each blocked entry names the specific missing
thing, so the report says what would have to be true to run it.

**Combinations are first-class.** A signal with no standalone edge can still
carry information conditioned on another; testing only singles would drop those
before they were ever measured. The pairs below all condition on an open-market
insider buy, because that is the population this system actually observes.
"""

from __future__ import annotations

from .backtest import Selector, all_of, everything, field_equals, threshold
from .sweep import Candidate

#: Form 4 transaction code for an open-market purchase. Read alongside
#: acquired/disposed because the code alone does not distinguish direction on
#: every filing.
CODE_OPEN_MARKET_BUY = "P"

#: Notional above which a purchase counts as a conviction-sized one. $100k is
#: the conventional line in the insider-trading literature and is well above the
#: token director purchase made for optics.
LARGE_BUY_NOTIONAL = 100_000.0


def _buy(payload, label) -> bool:
    return (
        payload.get("transaction_code") == CODE_OPEN_MARKET_BUY
        and payload.get("acquired_disposed") == "A"
    )


def _true(key: str) -> Selector:
    """Feature selector. False when the feature is absent, which is correct:
    an event we could not compute features for did not meet the condition, and
    treating unknown as met would fabricate trades."""
    return field_equals(key, True)


#: Payload-only hypotheses -- these need nothing but the Form 4 itself.
def insider_candidates() -> list[Candidate]:
    return [
        Candidate(
            "baseline: every event", everything,
            "Trade every labelled event. Any real signal must beat this, and "
            "without it a positive mean looks like a finding when it is the "
            "population's drift.",
            prior="small positive drift, not significant after costs",
            controlled=False,
        ),
        Candidate(
            "open-market buy", _buy,
            "Form 4 code P with shares acquired. The one insider transaction "
            "type that costs the filer money and cannot be explained by a "
            "vesting schedule or a tax election.",
            prior="the strongest single payload filter we have; still expected "
                  "to lose to costs on the smallest names",
        ),
        Candidate(
            "officer buy",
            all_of(_buy, field_equals("is_officer", True)),
            "Operational visibility. An officer sees the numbers before the "
            "board does.",
            prior="better than the director cut",
        ),
        Candidate(
            "director buy",
            all_of(_buy, field_equals("is_director", True)),
            "Board-level visibility, but directors also buy for signalling "
            "reasons that have nothing to do with information.",
            prior="weaker than officers; possibly indistinguishable from the "
                  "population",
        ),
        Candidate(
            "10% holder buy",
            all_of(_buy, field_equals("is_ten_percent", True)),
            "Large outside holders file Form 4 too. Different motive entirely: "
            "position building rather than information.",
            prior="no edge -- this is closer to a flows signal than a "
                  "disclosure signal",
        ),
        Candidate(
            "large buy",
            all_of(_buy, threshold("notional", LARGE_BUY_NOTIONAL)),
            f"Open-market purchase above ${LARGE_BUY_NOTIONAL:,.0f}. Size as a "
            "proxy for conviction.",
            prior="stronger mean, far fewer trades; likely fails the trade-count "
                  "gate on some horizons",
        ),
    ]


#: Indicator hypotheses. Each needs `features.FeatureBuilder` to have run, and
#: each is measured twice: alone, and conditioned on an insider buy. The pair is
#: the interesting one, but without the standalone we cannot tell whether the
#: pair adds anything or is just the insider filter wearing a hat.
FEATURE_HYPOTHESES: tuple[tuple[str, str, str], ...] = (
    ("near_high",
     "Within 10% of the 52-week high. The dominant feature in the microcap "
     "insider-purchase literature at 36% of model importance.",
     "the single most likely feature to survive; direction is confirmation, "
     "not reversion"),
    ("gain_over_10",
     "Up more than 10% over the trailing month. The same trend-confirmation "
     "claim as near_high but a distinct condition.",
     "works, and overlaps near_high enough that both surviving means one "
     "finding rather than two"),
    ("rsi_oversold",
     "RSI(14) at or below 30. The textbook mean-reversion entry.",
     "no edge standalone; this is the most-tested indicator in existence and "
     "an edge would have been arbitraged"),
    ("bb_below_lower",
     "Close below the lower Bollinger band.",
     "no edge; Bollinger himself says a band tag is not a signal"),
    ("bb_squeeze",
     "Bandwidth in the lowest decile of its own history -- compression before "
     "expansion, with no directional claim of its own.",
     "no standalone edge by construction; only interesting paired"),
    ("ttm_squeeze",
     "Bollinger bands inside the Keltner channels. A different definition of "
     "compression from bb_squeeze, included so the two can be compared rather "
     "than assumed equivalent.",
     "agrees with bb_squeeze often enough that a disagreement is the finding"),
    ("macd_cross",
     "MACD histogram turning positive on the bar.",
     "no edge"),
    ("donchian_breakout",
     "Close above the 20-session high.",
     "weak positive, eaten by costs on this universe"),
    ("swept_low",
     "Took out the prior 20-session low on conviction volume, then closed back "
     "inside. The failed breakdown.",
     "expected to fail, and the direction may be backwards. Osler (JIMF 2005) "
     "found stop-loss clusters PROPAGATE trends rather than reverse them -- it "
     "is take-profit clusters that reverse -- and that the effect is "
     "significant for hours, not days. Short-term reversal is real and is "
     "strongest in exactly our microcaps, but Avramov/Chordia/Goyal and de "
     "Groot/Huij/Zhou both find costs eat it there specifically. Kept as a "
     "control, not as a hope"),
    ("swept_high",
     "The bull-trap inverse, and the direct control for donchian_breakout: a "
     "bar piercing the prior high either holds it or does not, so the two "
     "partition the same events.",
     "negative for a long strategy -- which is the point of including it. Its "
     "real job is to stop a breakout result being read as a breakout result "
     "when it is a sweep result"),
    ("above_ma_200",
     "Close above the 200-day average. A regime filter, not an entry.",
     "raises the mean of whatever it is paired with, and does nothing alone"),
    ("engulfing_bull",
     "Bullish engulfing candle.",
     "no edge; single candle patterns do not survive costs"),
    ("connors_rsi2",
     "RSI(2) below 10 while above the 200-day average. Connors' mean reversion "
     "setup with its trend filter.",
     "high hit rate, small average win -- expected to die on the cost gate "
     "rather than the edge gate"),
    ("engulfing_reversal",
     "An open reconstruction of the GainzAlgo signal shape from its published "
     "description. Says nothing about the vendor's implementation.",
     "no edge; and whatever the result, it cannot be read as a verdict on the "
     "product, whose trial count is unknowable"),
)


def feature_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for key, rationale, prior in FEATURE_HYPOTHESES:
        out.append(Candidate(key, _true(key), rationale, prior=prior))
        out.append(Candidate(
            f"buy + {key}",
            all_of(_buy, _true(key)),
            f"Open-market insider purchase conditioned on: {rationale}",
            prior=f"pair test -- standalone prior was: {prior}",
        ))
    return out


#: Hypotheses served by `joins.py` rather than `features.py`. Same shape --
#: a boolean written into the payload point-in-time -- but sourced from the
#: intraday store, the holdings disclosures and the XBRL facts cache.
#:
#: These were the blocked list until the joins existed. Nothing about the
#: hypotheses changed; the wiring did.
JOIN_HYPOTHESES: tuple[tuple[str, str, str], ...] = (
    ("above_poc",
     "Price above the point of control -- the price at which the most volume "
     "traded over the prior window. The Market Profile claim is that price is "
     "drawn back toward accepted value.",
     "no standalone edge; the interesting version is the interaction with an "
     "insider buy, since 'insider bought while price sits above accepted "
     "value' is a different statement from either half"),
    ("below_value_area",
     "Price below the band holding 70% of prior volume. The classic Market "
     "Profile mean-reversion setup.",
     "the one microstructure claim with a mechanism rather than a chart "
     "pattern behind it, and still expected to lose to costs"),
    ("in_low_volume_node",
     "Price sitting where little volume traded. Profile theory says price "
     "moves quickly through thin prices.",
     "unknown. If it works it should work as a conditioner on speed of move, "
     "not on direction"),
    ("positive_delta",
     "Net buyer-initiated volume over the prior window, at least 10% of total.",
     "minute-bar delta agreed with tick-level Lee-Ready on sign only 1 time in "
     "4, so a result here is about the proxy as much as about order flow"),
    ("swept_low_intraday",
     "Took out the prior 20-session low intraday and reclaimed it with real "
     "session left -- the version of the sweep that the daily bar cannot "
     "express.",
     "expected to fail on the Osler evidence, and this is the only form of "
     "the claim worth the trial: hours, not days, is the horizon it operates "
     "on"),
    ("congress_bought",
     "A House member disclosed a purchase in the same name within 90 days, "
     "matched on the FILING date, not the transaction date.",
     "the individual-trade result in the literature was reversed twice "
     "(Eggers & Hainmueller 2013, Belmont 2022); significance survives only "
     "in an aggregate trade-weighted portfolio, which this is not"),
    ("activist_stake",
     "A 13D was filed on the name within 90 days. 13D means control intent, "
     "unlike a passive 13G.",
     "the strongest of the holdings signals if any of them work: 5 business "
     "days of disclosure lag against 45 for a 13F"),
    ("institution_added",
     "A 13F disclosed a position change within 90 days.",
     "no edge. The 45-day lag is most of a quarter, and whatever the filer "
     "knew is public by the time we see it"),
    ("profitable",
     "Positive trailing twelve-month net income, from XBRL filtered on `filed`.",
     "a quality conditioner rather than a signal. 74% of small filers fail it, "
     "so it mostly selects the larger end of our universe"),
)


def join_candidates() -> list[Candidate]:
    """Hypotheses reachable only once `joins.py` has enriched the payloads."""
    out: list[Candidate] = []
    for key, rationale, prior in JOIN_HYPOTHESES:
        out.append(Candidate(key, _true(key), rationale, prior=prior))
        out.append(Candidate(
            f"buy + {key}",
            all_of(_buy, _true(key)),
            f"Open-market insider purchase conditioned on: {rationale}",
            prior=f"pair test -- standalone prior was: {prior}",
        ))
    # Cheapness is a numeric cut rather than a flag, so these do not fit the
    # loop above. Thresholds are conventional rather than fitted: a fitted
    # threshold is a search, and a search that does not register its trials is
    # the thing the whole apparatus exists to prevent.
    out.append(Candidate(
        "buy + cheap on sales",
        all_of(_buy, lambda p, l: 0 < (p.get("price_to_sales") or 0) < 1.0),
        "Open-market purchase where the company trades below one times "
        "trailing revenue.",
        prior="P/S below 1 on a microcap usually means distress rather than "
              "value, so this may well be negative",
    ))
    out.append(Candidate(
        "buy + growing revenue",
        all_of(_buy, lambda p, l: (p.get("revenue_growth") or 0) > 0.10),
        "Open-market purchase at a company growing revenue over 10% "
        "year-on-year, both figures point-in-time from XBRL.",
        prior="the most plausible fundamental pairing: an insider buying into "
              "growth is a different statement from one buying a decline",
    ))
    return out


#: Hypotheses we intend to measure and cannot yet. Each names the missing
#: mechanism, not a judgement. These print in the report under "NOT measured --
#: untested, not rejected", which is the only place the distinction survives.
def blocked_candidates() -> list[Candidate]:
    return [
        Candidate(
            "order flow imbalance", everything,
            "Aggressor-side imbalance from trades and quotes.",
            prior="unknown; minute-bar delta and Lee-Ready agree on sign only "
                  "1 time in 4, so only the --exact path can test this",
            blocked_by="exact order flow is fetched per symbol and covers a "
                       "fraction of the universe; not enough events carry it "
                       "to clear the 30-trade floor",
        ),
        Candidate(
            "anchored VWAP from the filing", everything,
            "Whether buyers since the Form 4 disseminated are collectively up.",
            prior="informative about exits, not entries",
            blocked_by="anchors on the event itself, so every value describes "
                       "the post-entry path. Not an entry condition and cannot "
                       "be made into one without lookahead",
        ),
        Candidate(
            "news sentiment", everything,
            "FinBERT score on headlines within the disclosure window.",
            prior="the signal the operator is most enthusiastic about, which is "
                  "itself a reason to hold it to the same bar as everything else",
            blocked_by="sentiment accumulates forward only -- ApeWisdom serves "
                       "no history at any price, and news coverage of this "
                       "universe is near zero. Not backfillable",
        ),
        Candidate(
            "13F filer persistence ranking", everything,
            "Follow only those filers whose position changes persist across "
            "quarters, rather than every 13F filer.",
            prior="the 45-day lag probably removes whatever was there; the "
                  "ranked version is the only one worth the trial",
            blocked_by="`institution_added` is now measurable, but RANKING "
                       "filers needs `holdings.persistence` run across "
                       "consecutive quarters, and the store does not yet hold "
                       "enough of them",
        ),
        Candidate(
            "large cap: forward P/E", everything,
            "The one member of the standard five that no choice of universe "
            "unlocks.",
            prior="untestable rather than unpromising, and those must not be "
                  "reported the same way",
            blocked_by="needs point-in-time analyst consensus. Large caps have "
                       "abundant coverage; we have no historical record of what "
                       "it said, and using today's estimates on a past date is "
                       "the same lookahead that rules out Yahoo for reported "
                       "figures. A vendor problem, not a data-availability one",
        ),
    ]


def all_candidates(*, with_features: bool = True,
                   with_joins: bool = True) -> list[Candidate]:
    """The full backlog.

    `with_features=False` drops the indicator hypotheses and `with_joins=False`
    the intraday/holdings/fundamentals ones, for runs where that enrichment was
    skipped. Dropping them is not the same as leaving them in to report zero
    trades: a zero-trade row reads as "measured, nothing there", and these were
    never measured.
    """
    out = insider_candidates()
    if with_features:
        out += feature_candidates()
    if with_joins:
        out += join_candidates()
    out += blocked_candidates()
    return out

#: The cross-sectional valuation track. These run against the population from
#: `rebalance.py` -- one row per (symbol, month-end) with that date's multiples
#: and the symbol's quantile -- NOT against the insider event population.
#:
#: They are kept in their own function for that reason. Sweeping them beside the
#: insider candidates would pool two universes whose transaction costs differ by
#: an order of magnitude, and `fundamentals.guard_single_band` exists to refuse
#: exactly that.
VALUE_HYPOTHESES: tuple[tuple[str, str, str], ...] = (
    ("cheapest_ev_to_ebitda",
     "Cheapest quintile on the enterprise multiple. Loughran & Wellman (JFQA "
     "2011) build a factor from it earning 5.28%/yr; Gray & Vogel (JPM 2012) "
     "race it against P/E, book-to-market and FCF/TEV over forty years and it "
     "wins.",
     "the best-evidenced multiple we have found, run on the most arbitraged "
     "segment of the market. Those two cancel to an unknown, which is the "
     "honest prior"),
    ("cheapest_price_to_free_cash_flow",
     "Cheapest quintile on price to free cash flow. Cash is harder to manage "
     "than net income, and operating cash flow is the best-tagged number in "
     "XBRL at every size band.",
     "weaker than EV/EBITDA on the Gray & Vogel evidence, better covered in "
     "the data"),
    ("cheapest_price_to_earnings",
     "Cheapest quintile on trailing P/E. Usable only because the universe "
     "moved: defined for 86% of filers over $10B of revenue against 23% under "
     "$100M.",
     "no edge. The single most-screened number in existence, on the most "
     "analysed segment of the market"),
    ("cheapest_price_to_sales",
     "Cheapest quintile on price to sales. The one multiple that works at "
     "both ends of the size distribution, included here so the large-cap and "
     "microcap tracks share a comparable measurement.",
     "the bridge between the two universes; a disagreement between them on "
     "this metric is more interesting than either result alone"),
)


def value_candidates() -> list[Candidate]:
    """Cross-sectional hypotheses for the rebalance population.

    Deliberately NOT included in `all_candidates`: they need a different event
    population and a different universe, so mixing them into the insider sweep
    would produce results pooled across two economies.
    """
    out = [Candidate(
        "baseline: hold the whole universe", everything,
        "Every symbol in the cohort at every rebalance. The cheapest quintile "
        "has to beat this, not merely be positive -- in a rising market every "
        "quintile is positive.",
        prior="positive, and the number every value result must be read "
              "against",
        controlled=False,
    )]
    for key, rationale, prior in VALUE_HYPOTHESES:
        out.append(Candidate(key, _true(key), rationale, prior=prior))
    return out

