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

## The two blockers, precisely

Volume profile and order flow are not blocked by cost or by API access. They are
blocked by the fact that the pipeline fetches and stores **daily bars only**
(`prices.PriceSource.daily_bars`, `timeframe=1Day`), and both concepts are
defined on intraday data:

- **Volume profile** needs volume bucketed by price *within* a session. A daily
  bar reports one volume for the whole day, so the point of control and value
  area cannot be recovered from it at all — not approximately, not badly. There
  is no daily-bar version of this indicator.
- **Order flow** needs trades classified by aggressor side (buyer- or
  seller-initiated). That requires trade prints and the prevailing quote, then a
  tick-rule or Lee-Ready classification. Nothing in a daily OHLCV bar carries it.

Both are unblocked by the same piece of work: an intraday fetch and store path.
See `docs/INTRADAY.md` for what the data probe found.

## The liquidity sweep caveat

`swept_low` / `swept_high` are honest daily-bar approximations, and the
approximation is load-bearing. From a daily bar we know the low pierced a prior
level and the close recovered above it; we do not know the sequence within the
session, nor whether the reclaim was immediate or took six hours. The pattern
traders actually describe is the intraday one, which is materially stricter.

So a positive daily-bar result is an **upper bound on the population**, not a
measurement of the pattern. Treat it as a reason to build the intraday test, not
as confirmation of the idea.
