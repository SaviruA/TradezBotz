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
| 13D/13G activist stake | planned | — | 5-business-day disclosure, structured since Dec 2024 |
| 13F quarterly holdings | planned | — | 45-day lag; bulk archives exist |
| Congressional PTR | planned | — | Capitol Trades-style trailing-return ranking |

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

## The liquidity sweep caveat

`swept_low` / `swept_high` are honest daily-bar approximations, and the
approximation is load-bearing. From a daily bar we know the low pierced a prior
level and the close recovered above it; we do not know the sequence within the
session, nor whether the reclaim was immediate or took six hours. The pattern
traders actually describe is the intraday one, which is materially stricter.

So a positive daily-bar result is an **upper bound on the population**, not a
measurement of the pattern. Treat it as a reason to build the intraday test, not
as confirmation of the idea.
