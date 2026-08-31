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
| **Liquidity sweep (bullish)** | built | `indicators.swept_low` | daily-bar approximation — see caveat below |
| **Liquidity sweep (bearish)** | built | `indicators.swept_high` | exact control for Donchian breakout |
| **Anchored VWAP** | built | `indicators.anchored_vwap` | anchors to real events, not arbitrary dates |
| Relative volume | built | `indicators.relative_volume` | median-based; gates the sweeps |
| **Volume profile (POC / value area)** | blocked | — | needs intraday pipeline, see below |
| **Order flow (signed volume / delta)** | blocked | — | needs trade+quote pipeline, see below |
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
| Margin compression (gross + operating) | planned | 4 consecutive quarters of decline; GrossProfit / OperatingIncomeLoss / Revenues |
| Price/Sales, Price/FCF, EV/EBITDA | planned | XBRL plus our own prices; no vendor needed |
| YoY revenue growth | planned | conditioner rather than a signal on its own |
| Value/Growth score (P/S ÷ growth) | planned | a sales-based PEG variant; low = most growth per dollar of valuation |
| Customer concentration > 25% | blocked | 10-K narrative text, not tagged in XBRL |
| GAAP vs non-GAAP gap, guidance cuts | blocked | narrative text |

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
