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
     "the daily-bar version is a loose upper bound on the real pattern; a "
     "positive result argues for building the intraday test, not for trading it"),
    ("swept_high",
     "The bull-trap inverse, and the direct control for donchian_breakout.",
     "negative for a long strategy -- which is the point of including it"),
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


#: Hypotheses we intend to measure and cannot yet. Each names the missing
#: mechanism, not a judgement. These print in the report under "NOT measured --
#: untested, not rejected", which is the only place the distinction survives.
def blocked_candidates() -> list[Candidate]:
    return [
        Candidate(
            "volume profile: POC rejection", everything,
            "Price rejected from the session's point of control, where the most "
            "volume traded.",
            prior="unknown -- this is the one microstructure claim with a "
                  "mechanism rather than a chart pattern behind it",
            blocked_by="intraday profiles exist in profiles.db but are not "
                       "joined to labelled events; needs a FeatureBuilder "
                       "equivalent reading ProfileStore",
        ),
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
            "congressional copy", everything,
            "Buy alongside a disclosed House PTR purchase.",
            prior="the aggregate trade-weighted result in the literature is "
                  "weak and the individual-trade result was reversed twice",
            blocked_by="House PTRs are ingested but not joined to the labelled "
                       "event population; needs the disclosure-lag-aware join",
        ),
        Candidate(
            "13F filer persistence", everything,
            "Follow filers whose position changes persist across quarters.",
            prior="the 45-day lag probably removes whatever was there",
            blocked_by="persistence gate has not been run over enough ingested "
                       "quarters to rank filers",
        ),
        Candidate(
            "P/S with growth", everything,
            "Sales multiple against sales growth, on XBRL facts filtered by "
            "their `filed` date.",
            prior="value screens on microcaps mostly select distress",
            blocked_by="XBRL facts are not joined to events; needs a "
                       "point-in-time join on `filed`",
        ),
    ]


def all_candidates(*, with_features: bool = True) -> list[Candidate]:
    """The full backlog. `with_features=False` drops the indicator hypotheses,
    for a run where feature enrichment was skipped -- they would otherwise all
    report zero trades, which reads like a measurement and is not one."""
    out = insider_candidates()
    if with_features:
        out += feature_candidates()
    out += blocked_candidates()
    return out
