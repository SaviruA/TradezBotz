# Strategy backlog

Nothing is dropped before it is tested. A signal with no standalone edge can
still carry information in combination, so the unit of record here is the
*hypothesis*, and every hypothesis gets a row whether or not it looks promising.

Status values:

| status | meaning |
| --- | --- |
| `built` | function exists, unit-tested, ready to register as a trial |
| `blocked` | cannot be computed from data we hold; the blocker is named |
| `planned` | agreed, not yet written |

## This table is now executable

`research/candidates.py` is the machine-readable half of this document, and
`python -m tradezbotz measure` runs it. Every row that reaches a backtest has a
`Candidate` there carrying its rationale, its recorded prior, and — where it
cannot run — the named blocker, which prints in the report under *"NOT measured
— untested, not rejected"*.

Keeping the two in sync is manual and that is a known weakness. The mitigation
is that the code side is the one that runs: a hypothesis in this table with no
`Candidate` is never measured, while a `Candidate` missing from this table is
still measured and still reported.

**`built` never meant reachable.** For most of this project's life, every
function in the tables below was written and unit-tested, and *no code path
called any of them from a backtest*. Twenty-odd indicators, a cost model, a
clustering correction and a trial registry all worked in isolation and were
never once run together. The gap was invisible precisely because each piece was
individually green. `measure` is the wiring; the honest reading of every earlier
"nothing has been measured yet" is that there was nothing to measure *with*.

**Indicators reach a selector through the payload.** A `Selector` is
`(payload, label) -> bool` and never sees bars, so `research/features.py`
computes each indicator once per (symbol, entry day) and writes the answer into
the payload. It evaluates on bars strictly **before** the entry session, because
the entry session's own close is not knowable when we buy its open — reading
indicators on the entry bar would produce a strong, entirely fictitious edge on
every momentum feature at once.

**Every row costs trial budget.** Each entry below, individually and in each
pair, is one trial in the Deflated Sharpe denominator. Roughly 20 singles plus
their pairs is several hundred trials, and our own noise simulation produced a
best Sharpe of 2.33 from 1,000 pure-noise trials. That is the honest price of
testing everything, and it is the right price to pay — but it means a single
strong-looking result proves much less than it would in isolation, and the
control group matters more than the candidate.

---

## Price / volume structure

| hypothesis | status | where | notes |
| --- | --- | --- | --- |
| Bollinger squeeze | built | `indicators.bollinger_squeeze` | bandwidth in lowest decile of own history |
| Bollinger lower-band tag | built | `indicators.bollinger_below_lower` | Bollinger himself: a band tag alone is not a signal |
| RSI oversold | built | `indicators.rsi_oversold` | Wilder smoothing, period 14 |
| MACD bullish cross | built | `indicators.macd_bullish_cross` | a crossing, not a state |
| Donchian breakout | built | `indicators.donchian_breakout` | prior range excludes current bar |
| Trend filter (above 200MA) | built | `indicators.above_ma` | mostly a conditioner for other signals |
| 12-1 momentum | built | `indicators.momentum` | skips recent month for short-term reversal |
| **Liquidity sweep (bullish)** | built | `indicators.swept_low` (daily), `microstructure.swept_low_intraday` | evidence points the other way — see caveat below |
| **Liquidity sweep (bearish)** | built | `indicators.swept_high` (daily), `microstructure.swept_high_intraday` | exact control for Donchian breakout; that is its real job |
| **Anchored VWAP** | built | `indicators.anchored_vwap` | anchors to real events, not arbitrary dates |
| Relative volume | built | `indicators.relative_volume` | median-based; gates the sweeps |
| **Volume profile (POC / value area)** | blocked | `microstructure.above_poc` etc | pipeline built; blocker is now the join from profiles.db to labelled events |
| **Order flow (signed volume / delta)** | blocked | `microstructure.lee_ready` | pipeline built; exact flow covers too few events to clear the 30-trade floor |
| VWAP reversion (session) | planned | — | needs intraday |
| Opening range breakout | planned | — | needs intraday |

## From awesome-systematic-trading

Mined 2026-08-30. The list is a curated index, not code, so these are candidate
hypotheses rather than dependencies. Only entries that work on data we hold are
listed; crypto-only and broker-API entries were skipped.

| hypothesis | status | why it earns a row |
| --- | --- | --- |
| Hurst exponent regime filter | planned | classifies a series as mean-reverting / random walk / trending. Most useful as a *conditioner*: Bollinger reversion should work in mean-reverting regimes and fail in trending ones, and that split is testable |
| Yang-Zhang volatility | planned | uses the full OHLC bar rather than closes, so it is far more efficient than close-to-close on the same data. Drop-in better input wherever we currently use stdev |
| Garman-Klass / Parkinson volatility | planned | same family; worth testing against Yang-Zhang since each handles gaps differently |
| Keltner channel | planned | the textbook squeeze is Bollinger bands *inside* Keltner channels. We implemented the squeeze as a bandwidth percentile instead, so this is the conventional definition as a control |
| Microprice (Stoikov) | planned | order-book fair-value estimator. Newly reachable: the intraday pipeline now pulls NBBO quotes |
| Pairs / cointegration | planned | the one genuinely different structure on the list -- relative rather than directional, so it pairs with everything else |
| Heikin-Ashi | planned | smoothed candles; a trend filter with a different lag profile from an MA |
| Dual Thrust | planned | open-range breakout with a volatility-scaled band |
| Parabolic SAR | planned | trailing stop rule; more interesting as an *exit* than an entry, and we have no exit rules yet |
| **TTM squeeze** (Bollinger inside Keltner) | built | `indicators.ttm_squeeze` | the conventional squeeze; LazyBear's ~76k-like open indicator. Control for our percentile version |
| **Connors RSI(2)** | built | `indicators.connors_rsi2` | 2-period RSI oversold, above the 200MA. High published win rate -- see the caveat |
| **Engulfing reversal** | built | `indicators.engulfing_reversal` | open reconstruction of the GainzAlgo signal shape -- see the caveat |
| **Darvas Box** | planned | event-anchored box from a new 12-month high; see the caveat below |
| Awesome Oscillator | planned | momentum via 5/34 median-price MAs; overlaps MACD, so worth testing as a substitute rather than an addition |

**Two tools rather than strategies**, both worth stealing:

- **honest-signals** scores chart patterns against a pattern-free baseline for
  the same market and timeframe, reporting lift with *cluster-robust confidence
  intervals*. That is exactly the unresolved problem in our own results: every
  backtest so far flags `CLUSTERED` because overlapping events on one symbol are
  not independent draws. Cluster-robust standard errors are the standard answer
  and we should adopt them.
- **alphalens** does factor IC and quantile-return analysis. Our engine measures
  one hypothesis at a time; cross-sectional ranking is a different and
  complementary question.

## Event signals

| hypothesis | status | where | notes |
| --- | --- | --- | --- |
| Insider buy (Form 4, code P) | built | `backtest` selectors | baseline event |
| Opportunistic vs routine insider | built | `classify.RoutineClassifier` | Cohen/Malloy/Pomorski split |
| Reddit sentiment rank | built | `apewisdom` | forward-only; no historical backfill yet |
| **8-K material event (by item)** | built | `filings.is_results_8k` etc | item codes carry the signal; raw filing *frequency* is a NEGATIVE signal |
| **424B offering / dilution** | built | `filings.is_immediate_dilution` | negative signal; offerings drop small caps 20-30% |
| **Distance from 52-week high** | built | `indicators.near_high` | 36% of model importance on microcap insider buys; strength, not weakness |
| 13D/13G activist stake | planned | — | 5-business-day disclosure, structured since Dec 2024 |
| 13F quarterly holdings | planned | — | 45-day lag; bulk archives exist |
| Congressional PTR | planned | — | Capitol Trades-style trailing-return ranking |

## Transaction costs gate every result now

`BacktestResult.survives_costs` is the gate, and it is False when no cost model
was supplied -- an unmeasured cost is not a passed test.

Measured across 232 cached symbols with the EDGE estimator: median round trip
**93bps**, p90 **251bps**. Sub-dollar names run past 1,000bps and are almost
certainly untradeable whatever the signal says. A 5-day strategy must clear
~0.93% per trade at the median name just to break even, against a published
microcap insider CAR of ~6.3%.

## Fundamentals, from SEC XBRL

A signal family orthogonal to everything above, which is all price- and
event-driven. Sourced from `data.sec.gov/api/xbrl`, not from a data vendor:
**every observation carries a `filed` date**, which is point-in-time correctness
for free and structurally the same guarantee as our own `observed_at`.

Verified available: 528 us-gaap concepts for XELB (a microcap), with GrossProfit
at 140 observations and OperatingIncomeLoss at 168. The frames endpoint returns
2,095 companies for one concept in one quarter.

| hypothesis | status | notes |
| --- | --- | --- |
| **Non-preplanned insider selling** | planned | Form 4 `aff10b5One` = false. The sell-side mirror of the routine/opportunistic classifier. **Coverage constraint below.** |
| Margin compression (gross + operating) | built | 4 consecutive quarters of decline; GrossProfit / OperatingIncomeLoss / Revenues |
| Price/Sales + gross margin | built | `fundamentals.Snapshot` -- P/E is undefined for 74% of small caps |
| **Price/FCF** | built | `Snapshot.price_to_free_cash_flow` — best-covered cash multiple at every size band |
| **EV/EBITDA** | built | `Snapshot.ev_to_ebitda` — best-evidenced multiple in the literature; a large-cap tool, 13.6% computable on microcaps |
| **Trailing P/E** | built | `Snapshot.price_to_earnings` — usable only because the universe moved: 86% defined over $10B revenue vs 23% under $100M |
| Forward P/E | blocked | needs point-in-time analyst consensus. **The one member of the five that no choice of universe unlocks** — a vendor problem, not a coverage one |
| PEG (textbook) | blocked | rejected on the same grounds, plus Damodaran's: dividing P/E by growth does not neutralise growth, ignores risk, and double-counts the first year |
| YoY revenue growth | built | conditioner rather than a signal on its own |
| Value/Growth score (P/S ÷ growth) | built | a sales-based PEG variant; low = most growth per dollar of valuation |
| Customer concentration > 25% | blocked | 10-K narrative text, not tagged in XBRL |
| GAAP vs non-GAAP gap, guidance cuts | blocked | narrative text |

### Which valuation multiple, and can we compute it?

Prompted by a "Five Numbers That Tell You What A Stock Actually Costs" guide
(P/S, forward P/E, PEG, EV/EBITDA, P/FCF). Two separate questions, and they give
opposite answers.

**Which multiple predicts returns?** The literature is fairly clear and it does
not favour the ones retail guides lead with. [Loughran & Wellman (JFQA
2011)](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/new-evidence-on-the-relation-between-the-enterprise-multiple-and-average-stock-returns/5CD22A12A06AFCDC5233E477757FB659)
build an enterprise-multiple factor earning **5.28% a year**, reading EV/EBITDA
as a proxy for the discount rate. [Gray & Vogel (JPM
2012)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1970693) race the
metrics against each other over 40 years and find **EBITDA/TEV wins**, beating
P/E, book-to-market and FCF/TEV. So the best-evidenced multiple is the one the
guide gives fourth billing, and PEG — which it calls a "cheat code" — is the one
[Damodaran](https://pages.stern.nyu.edu/~adamodar/pdfiles/eqnotes/peg.pdf)
dismantles: dividing P/E by growth does not neutralise growth, it entangles it
further, ignores risk entirely, and double-counts the first year.

**Can we compute it on our universe?** Measured against the SEC frames API for
CY2024, over the 1,815 filers reporting revenue under $100M:

| input | filers | share |
| --- | --- | --- |
| operating cash flow | 1,794 | 98.8% |
| net income | 1,754 | 96.6% |
| operating income | 1,620 | 89.3% |
| capex | 1,183 | 65.2% |
| D&A (both tags) | 831 | 45.8% |
| **FCF computable** (op cash flow − capex) | **1,175** | **64.7%** |
| **EBITDA computable** (op income + D&A) | 723 | 39.8% |
| ...+ cash | 665 | 36.6% |
| **EV/EBITDA computable** (+ any debt tag) | **261** | **14.4%** |

So the best multiple in the literature is computable for one small filer in
seven. **And the missing six are not missing at random** — they are the smallest
and least-resourced filers, which is precisely where insider buying
concentrates. An EV/EBITDA screen here would not be a valuation filter; it would
be a filter for having a full accounting department, correlated with size,
survival and analyst attention. That is a selection bias wearing a valuation
filter's clothing, the same objection that ruled out P/E.

A further trap specific to EV: a filer with no debt tag may have no debt, or may
simply not have tagged it. XBRL does not distinguish those, and treating absent
as zero would systematically understate enterprise value for the least
well-tagged names.

**P/FCF is the one to build.** 64.7% coverage, cash is materially harder to
manage than net income, and it is genuinely new information — every fundamental
we hold today derives from the income statement. It still needs a coverage gate,
because 35% missing is not nothing.

**Forward P/E is a non-starter** twice over: it needs analyst consensus, which
[microcaps largely do not have](https://www.osam.com/Commentary/a-true-microcap-strategy),
and point-in-time historical estimates are not available free — using today's
estimates to judge a 2019 decision is the same lookahead that ruled out Yahoo
and Macrotrends for fundamentals.

**A structural caution about the source.** Every worked example in that guide is
a mega-cap — SpaceX, Amazon, JPMorgan, Apple — and so are its thresholds
("under 15 reads as cheap", "PEG under one"). Those are calibrated on a
population where earnings exist. In ours, 74% of small filers have negative net
income, so three of its five numbers are undefined for most of the universe
before any judgement about their usefulness is made. The document is a
newsletter lead magnet with a portfolio CTA, which is not itself a reason to
discount the content, but it is a reason to expect thresholds chosen for
readability rather than fitted to anything.

**Where it agrees with us, it agrees for the right reason:** its case for P/S is
that it works on unprofitable companies with no earnings to divide by. That is
exactly why `fundamentals.py` leads with P/S.

### Decision: build the five, and scope them to large caps

Directed 2026-09-01. The objection above was not that the multiples are bad — it
was that they are calibrated on a population we were not trading. Moving the
universe answers that, so P/E, P/FCF and EV/EBITDA are now built in
`fundamentals.py` alongside P/S.

**Re-measured across the whole size distribution before building**, because the
microcap number alone does not say whether moving up fixes anything:

| band | n | P/E defined | FCF | EBITDA | EV/EBITDA |
| --- | --- | --- | --- | --- | --- |
| revenue < $100M | 1,991 | 23.4% | 61.9% | 38.2% | 13.6% |
| $100M – $1B | 1,217 | 49.1% | 73.8% | 56.2% | 33.4% |
| $1B – $10B | 1,217 | 73.3% | 69.2% | 62.5% | **48.0%** |
| over $10B | 427 | **85.7%** | 65.8% | 52.9% | 41.9% |

Two results that changed the plan:

**Size fixes P/E and does not fix EV/EBITDA.** P/E definedness goes 23% → 86%,
which is the whole case for moving universe. EV/EBITDA only reaches 42%, so the
best-evidenced multiple in the literature is still a minority even among the
largest filers — the top band is thick with banks and insurers for whom
operating income and capex do not mean what the formula assumes.

**"Bigger is better" is false past mid-cap.** FCF, EBITDA and EV/EBITDA all peak
in the **$1B–$10B** band and decline above it. If one band is to be picked for
the valuation track, the data says mid-cap, not mega-cap.

**Forward P/E stays blocked, and this is the correction to the instruction.** It
is the one member of the five that no choice of universe unlocks. Large caps
have abundant analyst coverage; we have no *point-in-time record of what that
coverage said*, and using today's estimates to judge a past date is the same
back-door lookahead that rules out Yahoo for reported figures. That is a vendor
problem, not a coverage problem, and market cap does not touch it.

**What is still missing is the question's shape, not the ratios.** Everything
else in this backlog is an event study — something happened on a date, measure
what followed. A valuation multiple is cross-sectional: rank the universe, hold
the cheapest slice, rebalance. The engine can express that, but only once a
**rebalance-date population** exists: one row per (symbol, month-end) carrying
that date's multiples and the symbol's quintile within its band. That single
piece blocks all four large-cap candidates, and it is the same piece for each.

`size_band` and `guard_single_band` exist so the two universes cannot be pooled
by accident. Costs differ by more than an order of magnitude across them — our
measured 93bps median on microcaps against roughly 5bps at the top, and
published implementation shortfall of 110.8bps for US small caps against 31.7bps
for large — so a pooled result is an average of two economies, weighted by
whichever filers happened to have tags.

### The 10b5-1 coverage constraint

Measured directly, not assumed. The structured `<aff10b5One>` element arrives
with the December 2022 Form 4 amendment:

    2018-06-14   structured 0/15    footnote mentions 2
    2021-06-15   structured 0/15    footnote mentions 1
    2023-06-15   structured 15/15   footnote mentions 0
    2026-08-26   structured 60/60   -- 9 true (15%), 51 false (85%)

Before 2023 the flag is mostly **absent**, not merely unstructured: only one or
two filings in fifteen mention 10b5-1 at all, in free-text footnotes. So this
signal covers roughly the most recent third of our ten-year window, and any
pre-2023 event must be labelled UNKNOWN rather than assumed discretionary --
the same discipline the routine/opportunistic classifier already applies.

The base rate where it does exist is clean and usable: 15% preplanned, 85%
discretionary.

## Where LLM research may and may not be used

Prompt-driven company research (deep dive, peer comparison, bear case) is
**forbidden as a backtest input** and permitted only forward-only, at execution
time, on a name the pipeline has already flagged.

The reason is the one recorded in `TOOLING.md`: a model asked to assess a 2019
company already knows how 2019 turned out. That leakage lives in the weights and
defeats `observed_at`, the trial registry, the DSR and the locked holdout at
once. There is no version of it that survives a historical evaluation.

Forward-only use is legitimate, with one honest cost: a discretionary veto
cannot be backtested, so it is an **unmeasured intervention** with unknown sign.
Every override should be logged so it can be assessed later rather than
disappearing into the result.

A related trap in the prompts as written: they suggest sourcing fundamentals
from Yahoo Finance and Macrotrends. Those display **restated** figures. A
company's 2019 revenue as shown today may have been restated in 2021, so using
it to evaluate a 2019 decision is lookahead by the back door. XBRL's `filed`
field is exactly what prevents that, and is why the table above sources from the
primary filing rather than a display layer.

## Required pairings

The user requirement is explicit: test individually **and** paired. The engine
already supports this via `backtest.all_of`, so pairing costs no new machinery,
only trial budget.

- each price/volume signal × insider buy
- each price/volume signal × sentiment rank
- anchored VWAP (anchored to the filing) × insider buy — the most motivated
  pair on this list: the anchor is the event itself, so the question becomes
  "are buyers since the insider filed in profit?", which is a different and
  sharper question than either signal alone
- liquidity sweep × relative volume (already coupled inside the function)
- liquidity sweep × Donchian breakout — these partition the same bars, so
  running both prevents reading a breakout result that is really a sweep result
- **insider buy x near 52-week high** -- the paper's strongest cut, and the one
  that inverts the naive "buy the dip" reading
- **insider buy x NO recent 424B** -- an insider buying into a company about to
  run an ATM is a different trade from one that is not
- **insider buy x 8-K item 1.01** -- insiders are documented to buy ahead of new
  customer and supplier agreements, which is what 1.01 discloses
- **insider buy x NO recent non-preplanned insider selling** -- an officer
  buying while another is discretionarily selling is a different signal from a
  clean one, and both sides come from the same form we already parse
- **insider buy x margin compression** -- the bear-case pairing: does insider
  conviction survive deteriorating fundamentals, or is it the stronger tell?
- Bollinger squeeze × Donchian breakout (compression then expansion)

---

## The blockers are cleared

Both were unblocked on 2026-08-30 by building the intraday fetch and store path
(`research/intraday.py`, `research/microstructure.py`, `tradezbotz
backfill-intraday`). Sessions are reduced once to a compact price histogram plus
flow statistics and stored at ~600 bytes each, so the full universe projects to
roughly 110MB rather than the ~10^8 raw minute bars it would otherwise take.

Verified end to end against live data: 60 sessions across four small caps, with
exact Lee-Ready flow classification on every one.

## The order flow warning that must not be lost

There are two ways to sign volume, and **they do not agree**:

| | source | cost | correct? |
| --- | --- | --- | --- |
| `tick_minute` | direction of minute closes | free, comes with the bars | no |
| `lee_ready` | prints against the prevailing NBBO | one request per symbol-day | yes |

Measured against real prints on four small caps for 2026-08-24:

| symbol | minute-bar delta | Lee-Ready delta | agree on sign? |
| --- | --- | --- | --- |
| XELB | +0.0123 | -0.1133 | no |
| RCG | -0.0078 | +0.3649 | no |
| GNSS | +0.0396 | -0.1118 | no |
| IMTE | -0.0564 | -0.3519 | yes (6x magnitude) |

**One in four.** The cheap classifier is not a lower-resolution version of the
expensive one; it measures something else. A minute containing 500 buys and 500
sells nets to whatever its close did, and the tick rule sees one signed number.

Consequences, all enforced in code:

1. `backfill-intraday --exact` is what produces usable order flow. Without it
   the stored delta is labelled `tick_minute`.
2. `SessionProfile.flow_method` records which classifier ran, and
   `delta_ratio` / `cumulative_delta` **raise** on a window mixing the two
   rather than averaging a measurement against its own error.
3. `tick_minute` delta is still worth testing -- as its own hypothesis, named
   honestly as "direction of minute closes", not as order flow.

Affordability is why exact classification is realistic at all: insider buying
concentrates in small caps, and a full session of those is 127-1,227 prints. A
mega-cap session is millions and is not attempted.

## Two caveats on the community indicators

**Connors RSI(2)** publishes win rates of 75-79% over long backtests, and that
is the thing to be careful about rather than the thing to be encouraged by. A
high hit rate with a small average win and an uncapped loss is the classic shape
that looks excellent until it does not. `outlier_dependent`, `survives_costs`
and the winsorised mean are the checks that matter here; the win rate is close
to uninformative on its own.

**`engulfing_reversal` is a reconstruction, not a product.** It was assembled
from GainzAlgo's own published description -- EMA trend, RSI momentum, ATR
bands, engulfing candles, ATR-scaled targets -- because the product itself is
closed and therefore cannot be evaluated honestly at all: the Deflated Sharpe is
meaningless without a trial count, and the number of variants tried before
release is unknowable from outside.

So we test the *idea* in the open, with canonical parameters and no sweeping,
like everything else here. A result says something about
engulfing-plus-oversold-plus-trend. It says nothing about the vendor's
implementation and must never be reported as though it did.

## The Darvas Box caveat

Bulkowski's test -- 104 ETFs and 557 stocks, 2001-2010, across two bulls and two
bears -- found **average gains near 0% on daily data** for both ETFs and stocks.
The only configuration that worked was weekly bars with a 52-week lookback and
~297-day holds: 49% win rate, +10.5% per trade.

Our horizons are (0, 1, 5, 20) on daily bars, which is the configuration that
measured zero. At a 93bps median round trip a 0% gross strategy is firmly
negative, so **testing it standalone at current horizons is close to a known
answer.** Bulkowski's diagnosed failure mode is that on daily scales the system
"opens a trade just before price peaks".

Most of the rule set already exists: a new 12-month high is `near_high`, the
breakout is `donchian_breakout`, the volume condition is `relative_volume`, and
the fall-back-into-the-box exit is exactly `swept_high`. Three things are
genuinely new -- the box is anchored to an *event* rather than a rolling window,
it requires 3-day confirmation, and it pyramids into higher boxes (we have no
position sizing at all).

**What makes it worth a trial anyway** is the half of Darvas's method that gets
forgotten. He called himself a *techno-fundamentalist* and filtered to companies
with new products and earnings growth before he looked at a single box. We have a
machine-readable, point-in-time version of that filter now: 8-K item 2.02
(results) and 1.01 (material agreement), plus insider buying. Box breakout
conditioned on those is a coherent reconstruction of the actual method rather
than of the indicator.

**Prerequisite:** a longer horizon. The working version held ~10 months and
nothing in `DEFAULT_HORIZONS` reaches past 20 days. This only became testable at
all after the labelling window widened from 2 years to 10.

## The liquidity sweep caveat

`swept_low` / `swept_high` are honest daily-bar approximations, and the
approximation is load-bearing. From a daily bar we know the low pierced a prior
level and the close recovered above it; we do not know the sequence within the
session, nor whether the reclaim was immediate or took six hours. The pattern
traders actually describe is the intraday one, which is materially stricter.

So a positive daily-bar result is an **upper bound on the population**, not a
measurement of the pattern. Treat it as a reason to build the intraday test, not
as confirmation of the idea.

### The literature points the other way, and that is recorded in the prior

Researched 2026-09-01, before any measurement:

- **Osler (JIMF 2005, "Stop-loss orders and price cascades")** is the paper
  everyone cites for this pattern, and it finds the opposite of what the pattern
  claims. Stop-loss clusters *propagate trends*; it is take-profit clusters that
  reverse. The retail narrative attached itself to that evidence with the sign
  flipped.
- The same paper: results are **"statistically significant for hours, although
  not for days"**. Our daily horizons are 1, 5 and 20 sessions — the documented
  effect has decayed before the shortest one closes.
- Short-term reversal after extreme moves is real, and **strongest in small
  illiquid stocks**. Avramov/Chordia/Goyal (2006) and de Groot/Huij/Zhou (2011)
  both find trading costs consume it in exactly that segment; what makes it
  survive is restricting to large caps, which we cannot do — the insider signal
  lives in microcaps, at our measured 93bps median round trip.
- **Bulkowski's busted-pattern data is the one genuinely supportive source**:
  downward breakouts bust at a 40% median rate against 24% for upward, and
  single-busted patterns move 23% (up) / 53% (down) against 42% / 15% for
  non-busted ones.
- SMC/ICT community backtests reporting 50–65% win rates are not usable
  evidence. The rule sets have subjective components — displacement, inducement,
  "major" POI — so two researchers get different signals from the same data, and
  no knowable trial count means no Deflated Sharpe. The same structural problem
  as GainzAlgo.

**The conclusion is not to drop it.** `swept_high` is the exact partition
complement of `donchian_breakout` — a bar piercing the prior high either holds it
or does not — so running both is what stops a breakout result being read as a
breakout result when it is a sweep result. That value holds regardless of whether
the sweep works. Both priors now say "expected to fail" rather than staying
hopeful.

### The intraday version, and why the fields had to be added first

`swept_low_intraday` / `swept_high_intraday` in `research/microstructure.py`
test the actual claim: they require the reclaim to leave real session behind it
(≥10% of session volume after the extreme printed), which is what separates a
stop run from a breakdown that ticked up at 15:58. The daily bar reports those
two identically.

That needed six fields `SessionProfile` did not have — `session_open`,
`session_close`, `low_minute`, `high_minute`, `volume_after_low`,
`volume_after_high`. Everything the reduction stored before them was order-free
(min, max, sum, a histogram), which is what makes the store small and is also
exactly why this class of pattern was untestable.

**They had to be added before the intraday backfill ran at scale.** Raw minute
bars are never kept, so a field missing at reduction time is only recoverable by
refetching the session. `profiles.db` was empty when this landed, so the cost was
zero; `status` reports any untimed sessions as an outstanding refetch bill, and
`backfill-intraday --refresh-untimed` pays it. `require_timing` refuses a mixed
store rather than silently skipping old rows — skipping would make the sample
definition a fact about deployment history rather than about the market.
